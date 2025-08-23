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
import os
import json
import asyncio
from pathlib import Path
from typing import Sequence
from sqlmodel import Session, select
from utils.models import ContentBlock
from html_maker.restructure_with_recursion import detect_nested_structure, Context
# from html_maker.first_pass import run_first_pass
from html_maker.fixup_pass import run_fixup_pass
from html_maker.provide_feedback import provide_feedback, Feedback
from utils.db import engine, check_schema_sync
from utils.schema import Document, Node
from utils.aws import sync_s3_to_folder, sync_folder_to_s3
from utils.html import validate_data_sources, validate_html_tags
from litellm import Router
from html_maker.restructure_with_recursion import create_router
from utils.file_editor import file_starts_with

logger = logging.getLogger(__name__)


async def main() -> None:
    dotenv.load_dotenv(override=True)

    # Configure
    LIMIT: int = 3
    USE_S3: bool = True
    content_blocks_dir: str = "./data/content_blocks"
    html_dir: str = "./data/html"
    
    # Fail fast if DB schema is out of sync
    check_schema_sync()

    # Setup LLM router
    api_key = os.getenv("API_KEY", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    router: Router = create_router(
        api_key=api_key,
        openai_api_key=openai_api_key,
        deepseek_api_key=deepseek_api_key,
        openrouter_api_key=openrouter_api_key,
    )
    
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

    # Get non-approved documents (HTML files that don't start with "<!-- Approved -->")
    non_approved_documents: list[Document] = [
        document for document in missing_documents
        if not (Path(html_dir) / f"doc_{document.id}" / "output.html").exists()
        or not file_starts_with((Path(html_dir) / f"doc_{document.id}" / "output.html"), "<!-- Approved -->")
    ]

    # Get documents that have content blocks files but not HTML files yet
    processable_documents: list[Document] = [
        document for document in non_approved_documents
        if (Path(content_blocks_dir) / f"doc_{document.id}_content_blocks.json").exists()
    ]

    if not processable_documents:
        print("No non-approved documents with content blocks found to process. Run generate_content_blocks.py first.")
        return

    async def process_document(document: Document) -> None:
        assert document.id, "Document ID is required"
        doc_id: int = int(document.id)

        print(f"Processing content blocks for document {doc_id}...")

        # Per-document output location
        doc_output_dir = Path(html_dir) / f"doc_{doc_id}"
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        output_file_name: str = "output.html"
        output_html_path = doc_output_dir / output_file_name

        # Load the content blocks from the file
        content_blocks_file = Path(content_blocks_dir) / f"doc_{doc_id}_content_blocks.json"
        with open(content_blocks_file, "r") as f:
            content_blocks_data = json.load(f)
            content_blocks: list[ContentBlock] = [
                ContentBlock.model_validate(block) for block in content_blocks_data
            ]

        # Generate input HTML
        input_html = "\n".join([block.to_html(block_id=i) for i, block in enumerate(content_blocks)])

        # If output.html exists, skip first pass; otherwise run first pass
        if not output_html_path.exists():
            current_html = await detect_nested_structure(
                flat_html=input_html,
                context=Context(),
                router=router,
                max_depth=7,
            )

            # Persist to file
            with open(output_html_path, "w") as wf:
                wf.write(current_html)

        else:
            with open(output_html_path, "r") as rf:
                current_html = rf.read()

        # Validate and iterate fixups up to 3 times, incorporating feedback when mechanical checks pass
        fixup_attempt: int = 0
        while fixup_attempt < 3:
            missing_ids, extra_ids = validate_data_sources(input_html, current_html)
            is_valid_html, invalid_tags = validate_html_tags(current_html)

            # If mechanical checks pass, request feedback
            feedback_text: str | None = None
            if (
                current_html
                and len(missing_ids) + len(extra_ids) == 0
                and is_valid_html
                and not invalid_tags
            ):
                # On the 3rd loop, if mechanical validation passes, auto-approve regardless of feedback
                if fixup_attempt >= 2:
                    if not current_html.startswith("<!-- Approved -->"):
                        current_html = "<!-- Approved -->\n" + current_html
                    with open(output_html_path, "w") as wf:
                        wf.write(current_html)
                    return
                try:
                    feedback: list[Feedback] = await provide_feedback(
                        input_html=input_html, output_html=current_html, router=router
                    )
                except Exception:
                    feedback = []

                # Parse feedback and filter critical items
                criticals: list[Feedback] = [item for item in feedback if item.severity == "critical"]
                logger.info("Critical feedback count: %d", len(criticals))
                for idx, item in enumerate(criticals, start=1):
                    msg = str(item.message)
                    ids = item.affected_ids
                    if ids:
                        logger.info("Critical %d: %s affected_ids=%s", idx, msg, ids)
                    else:
                        logger.info("Critical %d: %s", idx, msg)
                if not criticals:
                    if not current_html.startswith("<!-- Approved -->"):
                        current_html = "<!-- Approved -->\n" + current_html
                    with open(output_html_path, "w") as wf:
                        wf.write(current_html)
                    return

                # Build feedback text for fixup prompt
                feedback_lines: list[str] = []
                for item in criticals:
                    msg = str(item.message)
                    ids = item.affected_ids
                    ids_repr = f" affected_ids={ids}" if ids else ""
                    feedback_lines.append(f"- {msg}{ids_repr}")
                feedback_text = "\n".join(feedback_lines)

                fixup_attempt += 1
                current_html = await asyncio.to_thread(
                    run_fixup_pass,
                    input_html,
                    current_html,
                    output_file_name,
                    missing_ids,
                    extra_ids,
                    not is_valid_html,
                    invalid_tags,
                    feedback_text,
                    3600,
                    doc_id,
                    use_deepseek=False,
                )

                # Only persist if we got back text (don't overwrite if we errored!)
                if current_html:
                    with open(output_html_path, "w") as wf:
                        wf.write(current_html)
                continue

    # Convert the content blocks into HTML for up to LIMIT docs concurrently
    tasks: list[asyncio.Task] = []
    counter: int = 0
    for document in processable_documents:
        if counter >= LIMIT:
            break
        counter += 1
        tasks.append(asyncio.create_task(process_document(document)))

    await asyncio.gather(*tasks)

    # Upload the HTML to S3
    if USE_S3:
        sync_folder_to_s3(html_dir, "html")

    print("Content blocks processing completed!")


if __name__ == "__main__":
    asyncio.run(main())
