"""CCDR Content Blocks Processing

Processes content blocks into structured nodes and uploads them to the database.
This is the second phase of the pipeline that takes JSON content blocks and converts
them into the final database schema with section classification and embeddings.

Usage:
    uv run process_content_blocks.py

The script loads content blocks from JSON files and processes them through restructuring,
classification, database upload, and embedding generation.
"""

import logging
import dotenv
import json
import os
import asyncio
from pathlib import Path
from typing import Sequence
from sqlmodel import Session, select
from litellm import Router
from utils.models import ContentBlock, StructuredNode
from utils.html import create_nodes_from_html
from nodes_maker.classify_section_types import classify_section_types
from nodes_maker.upload_to_db import upload_structured_nodes_to_db
from nodes_maker.generate_embeddings import generate_embeddings
from utils.db import engine, check_schema_sync
from utils.schema import Document, Node
from utils.aws import verify_environment_variables, sync_folder_to_s3
from utils.litellm_router import create_router
from utils.file_editor import file_starts_with

logger = logging.getLogger(__name__)


async def main() -> None:
    dotenv.load_dotenv(override=True)

    # Configure
    LIMIT: int = 50
    USE_S3: bool = True
    content_blocks_dir: str = "./data/content_blocks"
    html_dir: str = "./data/html"
    
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

    # Sync with S3 if USE_S3 is True
    if USE_S3:
        sync_folder_to_s3("content_blocks", content_blocks_dir)

    # Get documents that have content blocks and approved html files but haven't been processed to database yet
    processable_documents: list[Document] = [
        document for document in missing_documents
        if (Path(content_blocks_dir) / f"doc_{document.id}_content_blocks.json").exists()
        and (Path(html_dir) / f"doc_{document.id}" / "output.html").exists()
        and file_starts_with((Path(html_dir) / f"doc_{document.id}" / "output.html"), "<!-- Approved -->")
    ]

    if not processable_documents:
        print("No documents with content blocks found to process. Run generate_content_blocks.py and generate_html.py first.")
        return

    # Process the documents
    counter: int = 0
    failures: list[int] = []
    for document in processable_documents:
        if counter >= LIMIT:
            break
        counter += 1

        assert document.id, "Document ID is required"
        
        print(f"Processing content blocks for document {document.id}...")

        # Load the content blocks from the file
        content_blocks_file = Path(content_blocks_dir) / f"doc_{document.id}_content_blocks.json"
        with open(content_blocks_file, "r") as f:
            content_blocks_data = json.load(f)
            content_blocks: list[ContentBlock] = [
                ContentBlock.model_validate(block) for block in content_blocks_data
            ]

        # Load the html from the file
        html_file = Path(html_dir) / f"doc_{document.id}" / "output.html"
        with open(html_file, "r") as f:
            html = f.read()

        # Process the html into structured nodes
        if "<body>" in html and "</body>" in html:
            html = html.split("<body>")[1].split("</body>")[0]
    
        # Parse into structured nodes
        nodes = create_nodes_from_html(html, content_blocks)
        logger.info(f"Successfully restructured HTML with {len(nodes)} top-level nodes")
    
        # 9. Enrich the HTML sections with sectionType classifications
        nested_structure: list[StructuredNode] = await classify_section_types(router, nodes, "")
        logger.info(f"Document {document.id}: Section types classified successfully!")

        # 10. Convert the Pydantic models to our schema and ingest it into our database
        try:
            upload_structured_nodes_to_db(nested_structure, document.id)
            print(f"Document {document.id}: Structured nodes uploaded to database successfully!")
        except Exception as e:
            failures.append(document.id)
            print(f"Document {document.id}: Structured nodes failed to upload to database: {e}")

    # 11. Enrich the database records by generating relations from anchor tags
    # TODO: Implement this

    # 12. Generate embeddings for each ContentData record
    # TODO: For tables, explore embedding the table node's entire html content (add embeddingType for this?)
    print("Generating embeddings...")
    generate_embeddings(limit=LIMIT, api_key=openai_api_key)
    print("Embeddings generated successfully!")

    print("Content blocks processing completed!")


if __name__ == "__main__":
    asyncio.run(main())
