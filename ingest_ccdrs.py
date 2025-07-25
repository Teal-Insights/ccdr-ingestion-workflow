"""
1. Get document records from the database and identify which ones we need to process
2. For each document, download the PDF from S3 to a local temp file
3. Send the PDF to the Layout Extractor API to get bounding boxes and labels for each element on each page
4. Ignore headers and footers based on their labels
5. Many boxes are mislabeled as text; use heuristics to re-label them as images or figures
6. Extract images from the PDF and save them to S3
7. Describe the images with a VLM (e.g., Gemini)
8. Style the text blocks with the descriptions of the images
9. Preliminary HTML conversion, 1 p or img tag per block
10. Detect the top-level structure of the document
11. Recursively detect the nested structure of the document
12. Convert HTML to graph and ingest into the database
"""

import dotenv
import json
import os
import requests
import asyncio
from typing import Sequence
from sqlmodel import Session, select
from transform.extract_layout import extract_layout
from transform.map_page_numbers import add_logical_page_numbers
from transform.reclassify_blocks import reclassify_block_types
from transform.models import ExtractedLayoutBlock, BlockType, LayoutBlock, ContentBlockBase, ContentBlock
from transform.extract_images import extract_images_from_pdf
from transform.describe_images import describe_images_with_vlm
# from transform.extract_text_blocks import extract_text_blocks
from transform.style_text_blocks import style_text_blocks
from transform.detect_top_level_structure import detect_top_level_structure
from transform.detect_nested_structure import detect_nested_structure
from utils.db import engine, check_schema_sync
from utils.schema import Document, Node, TagName
from utils.aws import download_pdf_from_s3, upload_json_to_s3, verify_environment_variables

dotenv.load_dotenv(override=True)

# Configure
LIMIT: int = 1
USE_S3: bool = True

# Fail fast if required env vars are not set or DB schema is out of sync
gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
assert gemini_api_key, "GEMINI_API_KEY is not set"
deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
assert deepseek_api_key, "DEEPSEEK_API_KEY is not set"
layout_extractor_api_key: str = os.getenv("LAYOUT_EXTRACTOR_API_KEY", "")
assert layout_extractor_api_key, "LAYOUT_EXTRACTOR_API_KEY is not set"
layout_extractor_api_url: str = os.getenv("LAYOUT_EXTRACTOR_API_URL", "")
assert layout_extractor_api_url, "LAYOUT_EXTRACTOR_API_URL is not set"
verify_environment_variables()
assert check_schema_sync(), "DB schema is out of sync."

# Create a temporary working directory for the entire pipeline
# temp_dir: str = tempfile.mkdtemp(prefix="pdf_parsing_pipeline_")
# Temporarily use a permanent dir instead
temp_dir: str = "./artifacts/wkdir"
print(f"Using temporary directory: {temp_dir}")

# 1. Get Documents from the database where the count of child nodes is 0 (meaning they're unprocessed)
with Session(engine) as session:
    get_unprocessed = (
            select(Document.id, Document.publication_id, Document.storage_url, Document.download_url)
            .outerjoin(Node)
            .where(Node.id == None)  # noqa: E711
        )
    unproc_document_ids: Sequence[tuple[int | None, int | None, str | None, str]] = session.exec(get_unprocessed).all()

# Loop over the unprocessed documents and process them
counter: int = 0
for document_id, publication_id, storage_url, download_url in unproc_document_ids:
    if counter >= LIMIT:
        break
    counter += 1

    if not document_id or not publication_id:
        raise ValueError("Document ID and publication ID are required")

    # 2. Download the PDF from S3 to a local temp file and save it to the temp directory
    pdf_path: str | None
    layout_path: str | None = None
    if USE_S3 and storage_url:
        pdf_path, layout_path = download_pdf_from_s3(temp_dir, (publication_id, document_id), storage_url)
    else:
        # Fall back to downloading directly from the World Bank
        pdf_path = os.path.join(temp_dir, "pdfs", f"{document_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(requests.get(download_url).content)
        print(f"Downloaded PDF to {pdf_path}")
    assert pdf_path, "PDF path is None"

    # 3. Extract layout JSON from the PDF using the Layout Extractor API
    if not layout_path:
        layout_path = extract_layout(pdf_path, os.path.join(temp_dir, f"doc_{document_id}.json"))
        print(f"Extracted layout to {layout_path}")
        upload_json_to_s3(temp_dir, (publication_id, document_id))
    else:
        print(f"Skipping layout extraction and using downloaded layout at {layout_path}")

    # Load the layout JSON from the file
    with open(layout_path, "r") as f:
        extracted_layout_blocks: list[ExtractedLayoutBlock] = [
            ExtractedLayoutBlock.model_validate(block) for block in json.load(f)
        ]

    # 4. Use page numbers to label all blocks with logical page numbers, then discard headers footer, and page number blocks
    layout_blocks: list[LayoutBlock] = add_logical_page_numbers(extracted_layout_blocks, deepseek_api_key)
    filtered_layout_blocks: list[LayoutBlock] = [
        block for block in layout_blocks if block.type not in [BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER]
    ]
    print(f"Added logical page numbers to {len(layout_blocks)} blocks")

    # 5. Re-label text blocks that are actually images or figures by detecting if there's an image or geometry in the bbox
    content_blocks: list[ContentBlockBase] = asyncio.run(
        reclassify_block_types(filtered_layout_blocks, pdf_path)
    )
    print(f"Re-labeled {len(content_blocks)} blocks")

    # 6. Extract and describe images with a VLM (e.g., Gemini)
    content_blocks_with_images: list[ContentBlock] = extract_images_from_pdf(content_blocks, pdf_path, temp_dir, document_id)
    print("Images extracted successfully!")

    # 7. Describe the images with a VLM (e.g., Gemini)
    content_blocks_with_descriptions: list[ContentBlock] = asyncio.run(describe_images_with_vlm(
        content_blocks_with_images, gemini_api_key, temp_dir, document_id
    ))
    print("Images described successfully!")

    # 8. For spans in the pymupdf dict that have formatting flags, substring match to
    # the LayoutLM-detected text content and add style tags to the matched substrings
    styled_text_blocks: list[ContentBlock] = style_text_blocks(
        content_blocks_with_descriptions, pdf_path, temp_dir
    )

    # 9. Detect the top-level structure of the document
    top_level_structure: list[tuple[TagName, list[ContentBlock]]] = asyncio.run(
        detect_top_level_structure(
            styled_text_blocks, api_key=gemini_api_key
        )
    )
    print("Structure detected successfully!")

    # 10. Recursively detect the nested structure of the document
    nested_structure: list[tuple[TagName, list[ContentBlock] | list[tuple]]] = asyncio.run(
        detect_nested_structure(
            top_level_structure, api_key=gemini_api_key
        )
    )
    print("Nested structure detected successfully!")

    # 11. Transform the cleaned HTML document into a graph matching our schema and ingest it into our database
    


    # TODO: 12. Enrich the database records by generating relations from anchor tags

print(f"Pipeline completed! All outputs in: {temp_dir}")
