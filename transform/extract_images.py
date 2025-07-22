import os
import pymupdf
from transform.models import ContentBlockBase, ContentBlock, BlockType
from utils.aws import upload_image_to_s3
from typing import cast


def extract_images_from_pdf(
    content_blocks: list[ContentBlockBase],
    pdf_path: str,
    temp_dir: str,
    document_id: int
) -> list[ContentBlock]:
    """Extract and save images from a page, returning info for each image"""
    # Extract images from the PDF, save to S3, and capture storage url
    content_blocks_with_images: list[ContentBlock] = []
    pdf = pymupdf.open(pdf_path)
    for i, content_block in enumerate(content_blocks):
        if content_block.block_type != BlockType.PICTURE:
            content_blocks_with_images.append(
                cast(ContentBlock, content_block)
            )
            continue

        # Use positional data to get pixmap for the page and crop to the bbox
        page: pymupdf.Page = pdf[content_block.positional_data.page_pdf - 1]
        image = page.get_pixmap(clip=content_block.positional_data.bbox)

        # Save image to S3
        storage_url = upload_image_to_s3(temp_dir, (document_id, i))

        # Save image to local temp dir, creating the images directory if it doesn't exist
        os.makedirs(os.path.join(temp_dir, "images"), exist_ok=True)
        image_path = os.path.join(temp_dir, "images", f"doc_{document_id}_{i}.png")
        with open(image_path, "wb") as f:
            f.write(image)

        # Create content block
        content_block_with_image = cast(ContentBlock, content_block)
        content_block_with_image.storage_url = storage_url
        content_blocks_with_images.append(content_block_with_image)

    pdf.close()

    return content_blocks_with_images

