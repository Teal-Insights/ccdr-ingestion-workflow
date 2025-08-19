"""CCDR Content Blocks Processing

Processes content blocks into structured nodes and uploads them to the database.
This is the second phase of the pipeline that takes JSON content blocks and converts
them into the final database schema with section classification and embeddings.

Usage:
    uv run process_content_blocks.py

The script loads content blocks from JSON files and processes them through restructuring,
classification, database upload, and embedding generation.
"""

import dotenv
import json
import asyncio
from pathlib import Path
from typing import Sequence
from sqlmodel import Session, select
from utils.models import ContentBlock
from transform.restructure_with_CC_service import restructure_with_claude_code
from utils.db import engine, check_schema_sync
from utils.schema import Document, Node
from utils.aws import sync_s3_to_folder, sync_folder_to_s3


async def main() -> None:
    dotenv.load_dotenv(override=True)

    # Configure
    LIMIT: int = 1
    USE_S3: bool = True
    content_blocks_dir: str = "./data/content_blocks"
    html_dir: str = "./data/html"
    
    # Fail fast if DB schema is out of sync
    check_schema_sync()
    
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
        sync_s3_to_folder("content_blocks", content_blocks_dir)
        sync_s3_to_folder("html", html_dir)

    # Get documents that have content blocks files but not HTML files yet
    processable_documents: list[Document] = [
        document for document in missing_documents
        if (Path(content_blocks_dir) / f"doc_{document.id}_content_blocks.json").exists()
    ]

    if not processable_documents:
        print("No documents with content blocks found to process. Run generate_content_blocks.py first.")
        return

    # Convert the content blocks into HTML
    counter: int = 0
    tasks: list[asyncio.Task] = []
    doc_ids: list[int] = []
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

        # Generate input HTML
        input_html = "\n".join([block.to_html(block_id=i) for i, block in enumerate(content_blocks)])
        doc_ids.append(int(document.id))
        tasks.append(
            asyncio.create_task(
                restructure_with_claude_code(
                    input_html=input_html,
                    output_file="output.html",
                    timeout_seconds=3600,
                )
            )
        )

    # Wait for all tasks to complete
    html_files = await asyncio.gather(*tasks)

    # Save the HTML to files mapped to their corresponding document IDs
    for doc_id, html_file in zip(doc_ids, html_files):
        output_file = Path(html_dir) / f"doc_{doc_id}_html.html"
        with open(output_file, "w") as f:
            f.write(html_file)

    # Upload the HTML to S3
    if USE_S3:
        sync_folder_to_s3(html_dir, "html")

    print("Content blocks processing completed!")


if __name__ == "__main__":
    asyncio.run(main())
