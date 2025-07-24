import os
import pymupdf
from transform.models import ContentBlockBase, ContentBlock, BlockType
from utils.aws import upload_image_to_s3


def extract_images_from_pdf(
    content_blocks: list[ContentBlockBase],
    pdf_path: str,
    temp_dir: str,
    publication_id: int,
    document_id: int
) -> list[ContentBlock]:
    """Extract and save images from a page, returning info for each image"""
    # Extract images from the PDF, save to S3, and capture storage url
    content_blocks_with_images: list[ContentBlock] = []
    pdf = pymupdf.open(pdf_path)
    for i, base_content_block in enumerate(content_blocks):
        # Use Pydantic's model_copy to "upgrade" the model with additional fields
        content_block = ContentBlock.model_validate(base_content_block.model_dump())

        if content_block.block_type != BlockType.PICTURE:
            content_blocks_with_images.append(content_block)
            continue

        # Use positional data to get pixmap for the page and crop to the bbox
        page: pymupdf.Page = pdf[content_block.positional_data.page_pdf - 1]
        rect_like_bbox = (
            content_block.positional_data.bbox["x1"],
            content_block.positional_data.bbox["y1"],
            content_block.positional_data.bbox["x2"],
            content_block.positional_data.bbox["y2"]
        )
        image = page.get_pixmap(clip=rect_like_bbox)

        # Save image to local temp dir, creating the images directory if it doesn't exist
        os.makedirs(os.path.join(temp_dir, "images"), exist_ok=True)
        image_path = os.path.join(temp_dir, "images", f"doc_{document_id}_{i}.webp")
        image.pil_save(image_path, format="WebP", optimize=True, quality=85)

        # Save image to S3
        storage_url = upload_image_to_s3(temp_dir, (publication_id, document_id), i, "webp")

        content_block.storage_url = storage_url
        content_blocks_with_images.append(content_block)

    pdf.close()

    return content_blocks_with_images

if __name__ == "__main__":
    import json
    with open(os.path.join("artifacts", "doc_601_content_blocks.json"), "r") as f:
        content_blocks = json.load(f)
        content_blocks = [ContentBlock.model_validate(block) for block in content_blocks]
    print(f"Loaded {len(content_blocks)} content blocks, of which {len([block for block in content_blocks if block.block_type == BlockType.PICTURE])} are pictures")
    content_blocks_with_images = extract_images_from_pdf(content_blocks, "artifacts/wkdir/doc_601.pdf", "artifacts", 242, 601)
    with open(os.path.join("artifacts", "doc_601_content_blocks_with_images.json"), "w") as f:
        json.dump([block.model_dump() for block in content_blocks_with_images], f, indent=2)
    print("Images extracted successfully!")