# TODO: Address case where Google Gemini can't differentiate white from transparent pixels

import asyncio
import base64
import io
import json
import os
from functools import partial
from pathlib import Path
from typing import Optional, Union
import tempfile

import pydantic
from PIL import Image
from litellm import completion, Choices
from litellm.files.main import ModelResponse
from tenacity import retry, stop_after_attempt, wait_exponential
import cairosvg

from .models import BlocksDocument, ImageBlock, SvgBlock, TextBlock

# Global semaphore to limit concurrent API calls to 2
_api_semaphore = asyncio.Semaphore(2)


class ImageDescription(pydantic.BaseModel):
    label: str
    description: str


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def describe_image_with_vlm(
    image: Image.Image, api_key: str
) -> ImageDescription | None:
    """Use Gemini to describe the image with async retry logic."""
    # Set environment variable for LiteLLM
    os.environ["GEMINI_API_KEY"] = api_key

    prompt = """Describe the image in detail. Include the following:
    - A label from: "chart", "graph", "diagram", "map", "photo", "table", or "text_box"
    - A description of what the image shows/communicates"""

    # Convert PIL Image to base64
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    # Create message content with text and image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                },
            ],
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
                model="gemini/gemini-2.5-flash",
                messages=messages,
                temperature=0.0,
                response_format={
                    "type": "json_object",
                    "response_schema": ImageDescription.model_json_schema(),
                },
            ),
        )

    # Parse JSON response into Pydantic model
    if (
        response
        and isinstance(response, ModelResponse)
        and isinstance(response.choices[0], Choices)
        and response.choices[0].message.content
    ):
        description = ImageDescription.model_validate_json(
            response.choices[0].message.content
        )
        return description
    else:
        raise Exception("No valid response from Gemini")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def describe_svg_as_text(svg_content: str, api_key: str) -> ImageDescription | None:
    """Use Gemini to describe SVG content as text."""

    prompt = f"""Analyze the following SVG content and describe what it represents. Include the following:
    - A label from: "chart", "graph", "diagram", "map", "photo", "table", or "text_box"
    - A description of what the SVG shows/communicates

SVG Content:
{svg_content}"""

    messages = [
        {
            "role": "user",
            "content": prompt,
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
                model="gemini/gemini-2.5-flash",
                messages=messages,
                temperature=0.0,
                response_format={
                    "type": "json_object",
                    "response_schema": ImageDescription.model_json_schema(),
                },
                api_key=api_key
            ),
        )

    # Parse JSON response into Pydantic model
    if (
        response
        and isinstance(response, ModelResponse)
        and isinstance(response.choices[0], Choices)
        and response.choices[0].message.content
    ):
        description = ImageDescription.model_validate_json(
            response.choices[0].message.content
        )
        return description
    else:
        raise Exception("No valid response from Gemini")


def convert_svg_to_image(svg_path: str, output_format: str = "PNG") -> Image.Image:
    """
    Convert an SVG file to a PIL Image.
    
    Args:
        svg_path: Path to the SVG file
        output_format: Format to convert to ("PNG" or "JPEG")
    
    Returns:
        PIL Image object
    """
    # Create a temporary file for the converted image
    with tempfile.NamedTemporaryFile(suffix=f".{output_format.lower()}", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Convert SVG to the specified format
        if output_format.upper() == "PNG":
            cairosvg.svg2png(url=svg_path, write_to=temp_path)
        elif output_format.upper() == "JPEG":
            cairosvg.svg2png(url=svg_path, write_to=temp_path + ".png")
            # Convert PNG to JPEG since cairosvg doesn't directly support JPEG
            png_image = Image.open(temp_path + ".png")
            rgb_image = png_image.convert("RGB")
            rgb_image.save(temp_path, "JPEG")
            os.unlink(temp_path + ".png")  # Clean up temporary PNG
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Load and return the PIL Image
        image = Image.open(temp_path)
        # Create a copy to ensure we can delete the temp file
        image_copy = image.copy()
        image.close()
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return image_copy
    
    except Exception as e:
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e


async def describe_images_in_json(
    json_file_path: str,
    images_dir: str,
    api_key: str,
    output_file_path: Optional[str] = None,
    svg_as_text: bool = False,
) -> str:
    """
    Load a JSON file with blocks, describe all image and SVG blocks, and save updated JSON.
    
    Args:
        json_file_path: Path to the JSON file containing blocks
        images_dir: Directory containing the extracted images/SVGs
        api_key: Gemini API key for image descriptions
        output_file_path: Path to save the updated JSON (defaults to overwriting input file)
        svg_as_text: If True, submit SVG content as text; if False, convert to image first
    
    Returns:
        Path to the output JSON file with descriptions
    """
    if output_file_path is None:
        output_file_path = json_file_path
    
    # Load the existing JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse into BlocksDocument
    blocks_doc = BlocksDocument.model_validate(data)
    
    # Filter blocks by type
    text_blocks = [block for block in blocks_doc.blocks if isinstance(block, TextBlock)]
    image_blocks = [block for block in blocks_doc.blocks if isinstance(block, ImageBlock)]
    svg_blocks = [block for block in blocks_doc.blocks if isinstance(block, SvgBlock)]
    
    total_describable = len(image_blocks) + len(svg_blocks)
    
    print(f"Found {len(text_blocks)} text blocks (will be skipped)")
    print(f"Found {len(image_blocks)} image blocks to describe")
    print(f"Found {len(svg_blocks)} SVG blocks to describe")
    if svg_blocks:
        svg_method = "text analysis" if svg_as_text else "image conversion"
        print(f"SVG processing method: {svg_method}")
    
    if total_describable == 0:
        print("No image or SVG blocks found for description")
        return output_file_path
    
    print(f"Processing {total_describable} blocks for description...")
    
    async def describe_block(block: Union[ImageBlock, SvgBlock]):
        try:
            # Construct the full file path
            file_path = Path(images_dir) / Path(block.storage_url).name
            
            if not file_path.exists():
                print(f"    Warning: File not found: {file_path}")
                return f"Error: File not found at {file_path}"
            
            # Handle different block types
            if isinstance(block, ImageBlock):
                # Load image directly and describe via vision
                pil_image = Image.open(file_path)
                description_result = await describe_image_with_vlm(pil_image, api_key)
                pil_image.close()
                
            elif isinstance(block, SvgBlock):
                if svg_as_text:
                    # Read SVG content as text and describe via text analysis
                    print(f"    Analyzing SVG as text: {file_path.name}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        svg_content = f.read()
                    description_result = await describe_svg_as_text(svg_content, api_key)
                else:
                    # Convert SVG to image first and describe via vision
                    print(f"    Converting SVG to PNG: {file_path.name}")
                    pil_image = convert_svg_to_image(str(file_path), "PNG")
                    description_result = await describe_image_with_vlm(pil_image, api_key)
                    pil_image.close()
            else:
                return f"Error: Unsupported block type: {type(block)}"
            
            if description_result:
                return f"{description_result.label}: {description_result.description}"
            else:
                return "Failed to generate description"
                
        except Exception as e:
            print(f"    Error describing {block.storage_url}: {e}")
            return f"Error generating description: {str(e)}"
    
    # Create description tasks for all describable blocks
    describable_blocks = image_blocks + svg_blocks
    description_tasks = [describe_block(block) for block in describable_blocks]
    
    # Run all description tasks concurrently
    descriptions = await asyncio.gather(*description_tasks, return_exceptions=True)
    
    # Update blocks with descriptions
    for block, description in zip(describable_blocks, descriptions):
        if isinstance(description, Exception):
            block.description = f"Error generating description: {str(description)}"
        else:
            block.description = str(description)
    
    # Save the updated JSON
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(blocks_doc.model_dump_json(indent=2, exclude_none=True))
    
    print(f"Updated {len(describable_blocks)} block descriptions")
    print(f"Results saved to: {output_file_path}")
    
    return output_file_path


if __name__ == "__main__":
    import dotenv
    import sys

    dotenv.load_dotenv(override=True)

    if len(sys.argv) < 3:
        print("Usage: uv run -m transform.describe_images <json_file> <images_dir> [output_file] [--svg-as-text]")
        print("Example: uv run -m transform.describe_images blocks.json ./images")
        print("Example: uv run -m transform.describe_images blocks.json ./images output.json --svg-as-text")
        print("Note: Processes ImageBlocks and SvgBlocks, skips TextBlocks")
        print("      --svg-as-text: Analyze SVG content as text instead of converting to image")
        sys.exit(1)

    json_file_path = sys.argv[1]
    images_dir = sys.argv[2]
    
    # Parse remaining arguments
    output_file_path = None
    svg_as_text = False
    
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "--svg-as-text":
            svg_as_text = True
        elif not output_file_path and not arg.startswith("--"):
            output_file_path = arg

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    try:
        output_path = asyncio.run(
            describe_images_in_json(
                json_file_path=json_file_path,
                images_dir=images_dir,
                api_key=api_key,
                output_file_path=output_file_path,
                svg_as_text=svg_as_text,
            )
        )
        print("Image descriptions completed successfully!")
        print(f"Output file: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
