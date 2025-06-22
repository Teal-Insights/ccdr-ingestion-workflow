# TODO: Store the file in S3 and catpure the actual storage url

from PIL import Image
import base64
import io
import os
import asyncio
from functools import partial
from litellm import completion, Choices
from litellm.files.main import ModelResponse
import pydantic
from tenacity import retry, stop_after_attempt, wait_exponential
import pymupdf
from pathlib import Path
from typing import List, Dict, Any, Optional, cast
from .models import ImageBlock, BlocksDocument, Block

# Global semaphore to limit concurrent API calls
_api_semaphore = asyncio.Semaphore(2)  # Allow up to 2 concurrent API calls

class ImageDescription(pydantic.BaseModel):
    label: str
    description: str

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def describe_image_with_vlm(
    image: Image.Image,
    api_key: str
) -> ImageDescription | None:
    """Use Gemini to describe the image with async retry logic."""
    # Set environment variable for LiteLLM
    os.environ["GEMINI_API_KEY"] = api_key

    prompt = """Describe the image in detail. Include the following:
    - A label from: "chart", "graph", "diagram", "map", "photo", "table", or "text_box"
    - A description of what the image shows/communicates"""

    # Convert PIL Image to base64
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Create message content with text and image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                }
            ]
        }
    ]

    # Use semaphore to limit concurrent API calls
    async with _api_semaphore:
        # Convert synchronous completion call to async using run_in_executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(
                completion,
                model="gemini/gemini-2.5-flash-preview-05-20",
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object", "response_schema": ImageDescription.model_json_schema()}
            )
        )
    
    # Parse JSON response into Pydantic model
    if response and isinstance(response, ModelResponse) and isinstance(response.choices[0], Choices) and response.choices[0].message.content:
        description = ImageDescription.model_validate_json(response.choices[0].message.content)
        return description
    else:
        raise Exception("No valid response from Gemini")

def extract_images_from_page(page: pymupdf.Page, page_number: int, output_dir: str) -> List[Dict[str, Any]]:
    """Extract and save images from a page, returning info for each image"""
    images_info: List[Dict[str, Any]] = []
    
    # Get image list from page
    image_list = page.get_images(full=True)
    
    print(f"    PyMuPDF detected {len(image_list)} images on page {page_number}")
    
    for img_index, img in enumerate(image_list):
        print(f"    Processing image {img_index + 1}: xref={img[0]}")
        # Extract image
        xref = img[0]  # xref is the first element
        pix = pymupdf.Pixmap(page.parent, xref)

        image_filename = f"page_{page_number}_image_{img_index + 1}.png"
        image_path = os.path.join(output_dir, image_filename)

        if pix.n - pix.alpha < 4:  # Can convert to PNG directly
            pix.save(image_path)
        else:
            # Convert CMYK to RGB first
            pix1 = pymupdf.Pixmap(pymupdf.csRGB, pix)
            pix1.save(image_path)
            pix = pix1  # Use the converted pixmap for info extraction
            pix1 = None

        # Get image bounding box
        bbox = page.get_image_bbox(img)
        if isinstance(bbox, pymupdf.Rect):
            image_info = {
                "filename": image_filename,
                "storage_path": image_path,
                "xref": xref,
                "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                "width": pix.width,
                "height": pix.height,
                "page_number": page_number
            }
            images_info.append(image_info)
        else:
            print(f"    Error: bbox is not a pymupdf.Rect object: {bbox}")
        
        pix = None
    
    return images_info

async def extract_images_from_pdf(
    pdf_path: str, 
    output_filename: str,
    api_key: Optional[str] = None,
    images_dir: Optional[str] = None
) -> str:
    """
    Extract images from PDF, describe them with VLM, and save as JSON blocks.
    
    Args:
        pdf_path: Path to the PDF file
        output_filename: Full path to the output JSON file
        api_key: Gemini API key for image descriptions (optional)
        images_dir: Directory to save extracted images (optional, defaults to 'images' subdir of output file's parent)
    
    Returns:
        Path to the output JSON file
    """
    # Setup directories
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if images_dir is None:
        images_dir_path = output_path.parent / "images"
    else:
        images_dir_path = Path(images_dir)
    
    images_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Open PDF
    try:
        doc = pymupdf.open(pdf_path)
        print(f"Processing PDF: {pdf_path}")
        print(f"Number of pages: {len(doc)}")
        
        # First pass: Extract all images from all pages
        all_image_blocks = []
        
        print("Extracting images from all pages...")
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_number = page_num + 1
            
            print(f"  Processing page {page_number}...")
            
            # Extract images from this page
            page_images = extract_images_from_page(page, page_number, str(images_dir_path))
            
            # Create blocks for all images on this page
            for image_info in page_images:
                # Load the saved image for description
                image_path = image_info["storage_path"]
                
                # Create the block structure as dict temporarily for processing
                block = {
                    "block_type": "image",
                    "page_number": page_number,
                    "bbox": image_info["bbox"],
                    "storage_url": image_path,  # Local file path for now
                    "description": "Image description not available",  # Default
                    "image_info": image_info  # Temporary field for processing
                }
                all_image_blocks.append(block)
        
        # Second pass: Describe all images concurrently across all pages
        if api_key and all_image_blocks:
            print(f"Describing {len(all_image_blocks)} images concurrently across all pages...")
            
            async def describe_block_image(block_data):
                try:
                    pil_image = Image.open(block_data["storage_url"])
                    description_result = await describe_image_with_vlm(pil_image, api_key)
                    if description_result:
                        return f"{description_result.label}: {description_result.description}"
                    else:
                        return "Failed to generate description"
                except Exception as e:
                    print(f"    Error describing image {block_data['image_info']['filename']}: {e}")
                    return f"Error generating description: {str(e)}"
            
            # Create description tasks for all images
            description_tasks = [describe_block_image(block) for block in all_image_blocks]
            
            # Run all description tasks concurrently
            descriptions = await asyncio.gather(*description_tasks, return_exceptions=True)
            
            # Update blocks with descriptions
            for block, description in zip(all_image_blocks, descriptions):
                if isinstance(description, Exception):
                    block["description"] = f"Error generating description: {str(description)}"
                else:
                    block["description"] = description
                # Remove temporary field
                del block["image_info"]
        else:
            # No API key provided or no images
            for block in all_image_blocks:
                block["description"] = "No API key provided for description"
                if "image_info" in block:
                    del block["image_info"]
        
        # Store page count before closing document
        total_pages = len(doc)
        doc.close()
        
        # Convert processed dicts to Pydantic models
        image_blocks: List[ImageBlock] = []
        for block_data in all_image_blocks:
            # Remove temporary field
            if "image_info" in block_data:
                del block_data["image_info"]
            # Create ImageBlock from cleaned data
            image_block = ImageBlock(**block_data)
            image_blocks.append(image_block)
        
        # Create output document using Pydantic model
        output_data = BlocksDocument(
            pdf_path=pdf_path,
            total_pages=total_pages,
            total_blocks=len(image_blocks),
            blocks=cast(List[Block], image_blocks)
        )
        
        # Save the results to JSON
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(output_data.model_dump_json(indent=2, exclude_none=True))
        
        print(f"Extracted {len(image_blocks)} images from {total_pages} pages")
        print(f"Results saved to: {output_filename}")
        print(f"Images saved to: {images_dir_path}")
        
        return str(output_path.absolute())
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise


if __name__ == "__main__":
    import dotenv
    import tempfile
    import sys

    dotenv.load_dotenv(override=True)
    
    if len(sys.argv) < 2:
        print("Usage: uv run extract_images.py <pdf_file>")
        print("Example: uv run extract_images.py document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Create a real temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="images_test_")
    output_filename = os.path.join(temp_dir, "images.json")
    images_dir = os.path.join(temp_dir, "images")
    
    try:
        output_path = asyncio.run(extract_images_from_pdf(
            pdf_path=pdf_path,
            output_filename=output_filename,
            api_key=os.getenv("GEMINI_API_KEY"),
            images_dir=images_dir
        ))
        print(f"Images extracted successfully!")
        print(f"Output file: {output_path}")
        print(f"Temporary directory: {temp_dir}")
        print(f"Note: Clean up temporary directory when done: rm -rf {temp_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)