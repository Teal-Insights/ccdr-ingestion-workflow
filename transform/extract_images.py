# TODO: Store the file in S3 and capture the actual storage url

import os
import pymupdf
from pathlib import Path
from typing import List, Dict, Any, Optional, cast, Tuple
from .models import ImageBlock, BlocksDocument, Block


def extract_images_from_page(
    page: pymupdf.Page, page_number: int, output_dir: str
) -> List[Dict[str, Any]]:
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
                "page_number": page_number,
            }
            images_info.append(image_info)
        else:
            print(f"    Error: bbox is not a pymupdf.Rect object: {bbox}")

        pix = None

    return images_info


def extract_images_from_pdf(
    pdf_path: str,
    output_filename: str,
    images_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Extract images from PDF and save as JSON blocks.

    Args:
        pdf_path: Path to the PDF file
        output_filename: Full path to the output JSON file
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

        # Extract all images from all pages
        all_image_blocks = []

        print("Extracting images from all pages...")
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_number = page_num + 1

            print(f"  Processing page {page_number}...")

            # Extract images from this page
            page_images = extract_images_from_page(
                page, page_number, str(images_dir_path)
            )

            # Create blocks for all images on this page
            for image_info in page_images:
                # Create the block structure
                block = {
                    "block_type": "image",
                    "page_number": page_number,
                    "bbox": image_info["bbox"],
                    "storage_url": image_info["storage_path"],  # Local file path for now
                    "description": None,  # No description initially
                }
                all_image_blocks.append(block)

        # Store page count before closing document
        total_pages = len(doc)
        doc.close()

        # Convert processed dicts to Pydantic models
        image_blocks: List[ImageBlock] = []
        for block_data in all_image_blocks:
            # Create ImageBlock from data
            image_block = ImageBlock(**block_data)
            image_blocks.append(image_block)

        # Create output document using Pydantic model
        output_data = BlocksDocument(
            pdf_path=pdf_path,
            total_pages=total_pages,
            total_blocks=len(image_blocks),
            blocks=cast(List[Block], image_blocks),
        )

        # Save the results to JSON
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(output_data.model_dump_json(indent=2, exclude_none=True))

        print(f"Extracted {len(image_blocks)} images from {total_pages} pages")
        print(f"Results saved to: {output_filename}")
        print(f"Images saved to: {images_dir_path}")

        return str(output_path.absolute()), str(images_dir_path.absolute())

    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise


if __name__ == "__main__":
    import tempfile
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run -m transform.extract_images <pdf_file>")
        print("Example: uv run -m transform.extract_images document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Create a real temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="images_test_")
    output_filename = os.path.join(temp_dir, "images.json")
    images_dir = os.path.join(temp_dir, "images")

    try:
        output_path, images_dir = extract_images_from_pdf(
            pdf_path=pdf_path,
            output_filename=output_filename,
            images_dir=images_dir,
        )
        print("Images extracted successfully!")
        print(f"Output file: {output_path}")
        print(f"Temporary directory: {temp_dir}")
        print(f"Note: Clean up temporary directory when done: rm -rf {temp_dir}")
        print(f"To add descriptions, run: uv run -m transform.describe_images {output_filename} {images_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
