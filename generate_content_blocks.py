"""CCDR Content Blocks Generation

Generates content blocks from PDF documents for the CCDR ingestion pipeline.
This is the first phase of the pipeline that processes PDFs through layout extraction,
image extraction, and styling to produce structured content blocks.

Usage:
    uv run generate_content_blocks.py

The script processes documents in batches (configurable LIMIT) and saves content blocks
as JSON files for the next phase of processing.
"""

import dotenv
import json
import os
import requests
import asyncio
from pathlib import Path
from typing import Sequence
from sqlmodel import Session, select
from litellm import Router
from transform.extract_layout import extract_layout
from transform.map_page_numbers import add_logical_page_numbers
from transform.reclassify_blocks import reclassify_block_types
from utils.models import ExtractedLayoutBlock, BlockType, LayoutBlock, ContentBlockBase, ContentBlock
from transform.extract_images import extract_images_from_pdf
from transform.describe_images import describe_images_with_vlm
from transform.style_text_blocks import style_text_blocks
from utils.db import engine, check_schema_sync
from utils.schema import Document, Node
from utils.aws import (
    download_pdf_from_s3, upload_json_to_s3, verify_environment_variables, sync_folder_to_s3, sync_s3_to_folder
)
from utils.litellm_router import create_router


async def get_content_blocks(
    document_id: int, publication_id: int, storage_url: str,
    download_url: str, working_dir: str, use_s3: bool,
    router: Router
) -> list[ContentBlock]:
    layout_extractor_api_key: str = os.getenv("LAYOUT_EXTRACTOR_API_KEY", "")
    assert layout_extractor_api_key, "LAYOUT_EXTRACTOR_API_KEY is not set"
    layout_extractor_api_url: str = os.getenv("LAYOUT_EXTRACTOR_API_URL", "")
    assert layout_extractor_api_url, "LAYOUT_EXTRACTOR_API_URL is not set"
    
    # 1. Download the PDF from S3 to a local temp file and save it to the temp directory
    pdf_path: str | None
    layout_path: str | None = None
    if use_s3 and storage_url:
        pdf_path, layout_path = download_pdf_from_s3(working_dir, (publication_id, document_id), storage_url)
    else:
        # Fall back to downloading directly from the World Bank
        pdf_path = os.path.join(working_dir, "pdfs", f"{document_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(requests.get(download_url).content)
        print(f"Downloaded PDF to {pdf_path}")
    assert pdf_path, "PDF path is None"

    # 2. Extract layout JSON from the PDF using the Layout Extractor API
    if not layout_path:
        layout_path = extract_layout(
            pdf_path,
            os.path.join(working_dir, f"doc_{document_id}.json"),
            layout_extractor_api_url,
            layout_extractor_api_key
        )
        print(f"Extracted layout to {layout_path}")
        upload_json_to_s3(working_dir, (publication_id, document_id))
    else:
        print(f"Skipping layout extraction and using downloaded layout at {layout_path}")

    # Load the layout JSON from the file
    with open(layout_path, "r") as f:
        extracted_layout_blocks: list[ExtractedLayoutBlock] = [
            ExtractedLayoutBlock.model_validate(block) for block in json.load(f)
        ]

    # 3. Use page numbers to label all blocks with logical page numbers, then discard page header and footer blocks
    layout_blocks: list[LayoutBlock] = await add_logical_page_numbers(
        extracted_layout_blocks, 
        router,
        pdf_path
    )
    filtered_layout_blocks: list[LayoutBlock] = [
        block for block in layout_blocks if block.type not in [BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER]
    ]
    print(f"Added logical page numbers to {len(layout_blocks)} blocks")

    # 4. Re-label text blocks that are actually images or figures by detecting if there's an image or geometry in the bbox
    content_blocks: list[ContentBlockBase] = await reclassify_block_types(
        filtered_layout_blocks, pdf_path, router
    )
    print(f"Re-labeled {len(content_blocks)} blocks")

    # 5. Extract images from the PDF
    content_blocks_with_images: list[ContentBlock] = extract_images_from_pdf(
        content_blocks, pdf_path, working_dir, publication_id, document_id
    )
    print("Images extracted successfully!")

    # 6. Describe the images with a VLM (e.g., Gemini)
    content_blocks_with_descriptions: list[ContentBlock] = await describe_images_with_vlm(
        content_blocks_with_images, working_dir, document_id, router
    )
    print("Images described successfully!")

    # 7. For spans in the pymupdf dict that have formatting flags, substring match to
    # the LayoutLM-detected text content and add style tags to the matched substrings
    styled_text_blocks: list[ContentBlock] = style_text_blocks(
        content_blocks_with_descriptions, pdf_path, working_dir
    )

    return styled_text_blocks


async def main() -> None:
    dotenv.load_dotenv(override=True)

    # Configure
    LIMIT: int = 200
    USE_S3: bool = True
    working_dir: str = "./data"
    content_blocks_dir: str = "./data/content_blocks"
    os.makedirs(content_blocks_dir, exist_ok=True)
    print(f"Using working directory: {working_dir}")

    # Fail fast if required env vars are not set or DB schema is out of sync
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    assert gemini_api_key, "GEMINI_API_KEY is not set"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    assert openai_api_key, "OPENAI_API_KEY is not set"
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    assert openrouter_api_key, "OPENROUTER_API_KEY is not set"
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    assert deepseek_api_key, "DEEPSEEK_API_KEY is not set"
    verify_environment_variables()
    check_schema_sync()
    
    router: Router = create_router(gemini_api_key, openai_api_key, deepseek_api_key, openrouter_api_key)

    # Get Documents from the database where the count of child nodes is 0 (meaning nodes have not been uploaded yet)
    with Session(engine) as session:
        get_missing = (
                select(Document)
                .outerjoin(Node)
                .where(Node.id == None)  # noqa: E711
            )
        missing_documents: Sequence[Document] = session.exec(get_missing).all()

    # Sync local files with S3 if USE_S3 is True
    if USE_S3:
        sync_folder_to_s3(content_blocks_dir, "content_blocks")

    # Get document ids for which we don't already have a content blocks file locally
    unproc_documents: list[Document] = [
        document for document in missing_documents
        if not (Path(content_blocks_dir) / f"doc_{document.id}_content_blocks.json").exists()
    ]

    # Loop over the unprocessed documents and process them
    counter: int = 0
    for document in unproc_documents:
        if counter >= LIMIT:
            break
        counter += 1

        assert document.id, "Document ID is required"
        assert document.publication_id, "Publication ID is required"
        assert document.storage_url, "Storage URL is required"
        assert document.download_url, "Download URL is required"

        print(f"Processing document {document.id}...")

        # Get the content blocks for the document
        content_blocks: list[ContentBlock] = await get_content_blocks(
            document.id, document.publication_id, document.storage_url, 
            document.download_url, working_dir, USE_S3, router
        )

        # Save the content blocks to a file
        output_file = Path(content_blocks_dir) / f"doc_{document.id}_content_blocks.json"
        with open(output_file, "w") as f:
            json.dump([block.model_dump() for block in content_blocks], f)
        
        print(f"Saved content blocks for document {document.id} to {output_file}")

    # Upload to S3 if USE_S3 is True
    if USE_S3:
        sync_s3_to_folder("content_blocks", content_blocks_dir)

    print(f"Content blocks generation completed! Output in: {content_blocks_dir}")


if __name__ == "__main__":
    asyncio.run(main())
