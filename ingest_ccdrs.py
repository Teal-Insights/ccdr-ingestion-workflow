# TODO: Coordinate router creation and use more parallelization at the top level of the pipeline?

"""CCDR Ingestion Pipeline

Main orchestration script for the CCDR (Country and Climate Development Reports) ingestion workflow.
Transforms World Bank PDF documents into a structured graph database format suitable for semantic search.

Usage:
    uv run ingest_ccdrs.py

The script processes documents in batches (configurable LIMIT) and outputs intermediate
artifacts to a working directory for debugging and pipeline inspection.
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
from utils.models import ExtractedLayoutBlock, BlockType, LayoutBlock, ContentBlockBase, ContentBlock, StructuredNode
from transform.extract_images import extract_images_from_pdf
from transform.describe_images import describe_images_with_vlm
from transform.style_text_blocks import style_text_blocks
from transform.restructure_with_CC import restructure_with_claude_code
from transform.classify_section_types import classify_section_types
from transform.upload_to_db import upload_structured_nodes_to_db
from transform.generate_embeddings import generate_embeddings
from utils.db import engine, check_schema_sync
from utils.schema import Document, Node
from utils.aws import download_pdf_from_s3, upload_json_to_s3, verify_environment_variables
from utils.litellm_router import create_router


async def main() -> None:
    dotenv.load_dotenv(override=True)

    # Configure
    LIMIT: int = 1
    USE_S3: bool = True

    # Fail fast if required env vars are not set or DB schema is out of sync
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    assert gemini_api_key, "GEMINI_API_KEY is not set"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    assert openai_api_key, "OPENAI_API_KEY is not set"
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    assert openrouter_api_key, "OPENROUTER_API_KEY is not set"
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    assert deepseek_api_key, "DEEPSEEK_API_KEY is not set"
    layout_extractor_api_key: str = os.getenv("LAYOUT_EXTRACTOR_API_KEY", "")
    assert layout_extractor_api_key, "LAYOUT_EXTRACTOR_API_KEY is not set"
    layout_extractor_api_url: str = os.getenv("LAYOUT_EXTRACTOR_API_URL", "")
    assert layout_extractor_api_url, "LAYOUT_EXTRACTOR_API_URL is not set"
    verify_environment_variables()
    assert check_schema_sync(), "DB schema is out of sync."

    router: Router = create_router(gemini_api_key, openai_api_key, deepseek_api_key, openrouter_api_key)

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
            layout_path = extract_layout(
                pdf_path,
                os.path.join(temp_dir, f"doc_{document_id}.json"),
                layout_extractor_api_url,
                layout_extractor_api_key
            )
            print(f"Extracted layout to {layout_path}")
            upload_json_to_s3(temp_dir, (publication_id, document_id))
        else:
            print(f"Skipping layout extraction and using downloaded layout at {layout_path}")

        # Load the layout JSON from the file
        with open(layout_path, "r") as f:
            extracted_layout_blocks: list[ExtractedLayoutBlock] = [
                ExtractedLayoutBlock.model_validate(block) for block in json.load(f)
            ]

        # 4. Use page numbers to label all blocks with logical page numbers, then discard page header and footer blocks
        layout_blocks: list[LayoutBlock] = await add_logical_page_numbers(
            extracted_layout_blocks, 
            router,
            pdf_path
        )
        filtered_layout_blocks: list[LayoutBlock] = [
            block for block in layout_blocks if block.type not in [BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER]
        ]
        print(f"Added logical page numbers to {len(layout_blocks)} blocks")

        # 5. Re-label text blocks that are actually images or figures by detecting if there's an image or geometry in the bbox
        content_blocks: list[ContentBlockBase] = await reclassify_block_types(
            filtered_layout_blocks, pdf_path, router
        )
        print(f"Re-labeled {len(content_blocks)} blocks")

        # 6. Extract images from the PDF
        content_blocks_with_images: list[ContentBlock] = extract_images_from_pdf(
            content_blocks, pdf_path, temp_dir, publication_id, document_id
        )
        print("Images extracted successfully!")

        # 7. Describe the images with a VLM (e.g., Gemini)
        content_blocks_with_descriptions: list[ContentBlock] = await describe_images_with_vlm(
            content_blocks_with_images, temp_dir, document_id, router
        )
        print("Images described successfully!")

        # 8. For spans in the pymupdf dict that have formatting flags, substring match to
        # the LayoutLM-detected text content and add style tags to the matched substrings
        styled_text_blocks: list[ContentBlock] = style_text_blocks(
            content_blocks_with_descriptions, pdf_path, temp_dir
        )

        # 9. Restructure the document with Claude Code
        nested_structure: list[StructuredNode] = restructure_with_claude_code(
            styled_text_blocks,
            "output.html",
            Path(f"./artifacts/doc_{document_id}")
        )
        print("Nested structure detected successfully!")

        # 10. Enrich the HTML sections with sectionType classifications
        nested_structure: list[StructuredNode] = await classify_section_types(router, nested_structure, "")
        print("Section types classified successfully!")

        # 11. Convert the Pydantic models to our schema and ingest it into our database
        upload_structured_nodes_to_db(nested_structure, document_id)
        print("Structured nodes uploaded to database successfully!")

        # 12. Enrich the database records by generating relations from anchor tags
        # TODO: Implement this

    # 13. Generate embeddings for each ContentData record
    generate_embeddings()
    # TODO: For tables, explore embedding the table node's entire html content (add embeddingType for this?)

    print(f"Pipeline completed! All outputs in: {temp_dir}")


if __name__ == "__main__":
    asyncio.run(main())