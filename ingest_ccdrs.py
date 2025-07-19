"""
1. Get document records from the database and identify which ones we need to process
2. For each document, download the PDF from S3 to a local temp file
3. Send the PDF to the Layout Extractor API to get bounding boxes and labels for each element on each page
4. Ignore headers and footers based on their labels
5. Many boxes are mislabeled as text; use heuristics to re-label them as images or figures
6. Mechanically extract text from text boxes using PyMuPDF
7. Get pixmap and extract images from image boxes using PyMuPDF
8. Describe the images with a VLM (e.g., Gemini) and add the descriptions to the JSON
9. Convert text to my HTML spec
10, Convert HTML to graph and ingest into the database
"""

import dotenv
import asyncio
import os
import requests
from typing import Literal, Sequence
from sqlmodel import Session, select
from transform.extract_text_blocks import extract_text_blocks_with_styling
from transform.extract_images import extract_images_from_pdf
from transform.convert_to_html import convert_blocks_to_html
from transform.detect_structure import detect_structure
from transform.clean_html import process_html_inputs_concurrently
from transform.describe_images import describe_images_in_json
from transform.extract_layout import extract_layout
from utils.db import engine, check_schema_sync
from utils.schema import Document, Node
from utils.aws import download_from_s3, verify_environment_variables

dotenv.load_dotenv(override=True)

# Configure
LIMIT: int = 1
USE_S3: bool = True

# Fail fast if required env vars are not set or DB schema is out of sync
gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
assert gemini_api_key, "GEMINI_API_KEY is not set"
layout_extractor_api_key: str = os.getenv("LAYOUT_EXTRACTOR_API_KEY", "")
assert layout_extractor_api_key, "LAYOUT_EXTRACTOR_API_KEY is not set"
layout_extractor_api_url: str = os.getenv("LAYOUT_EXTRACTOR_API_URL", "")
assert layout_extractor_api_url, "LAYOUT_EXTRACTOR_API_URL is not set"
verify_environment_variables()
check_schema_sync()

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
        pdf_path, layout_path = download_from_s3(temp_dir, (publication_id, document_id), storage_url)
    else:
        # Fall back to downloading directly from the World Bank
        pdf_path = os.path.join(temp_dir, "pdfs", f"{document_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(requests.get(download_url).content)
        print(f"Downloaded PDF to {pdf_path}")
    assert pdf_path, "PDF path is None"

    # 3. Extract layout JSON from the PDF using the Layout Extractor API
    if not layout_path:
        layout_path = extract_layout(pdf_path, os.path.join(temp_dir, "layout.json"))

    # 4. Use page numbers to label all blocks with logical page numbers, then discard header and footers

    # 5. Re-label text blocks that are actually images or figures by detecting if there's an image or geometry in the bbox

    # Extract ImageBlocks and write BlocksDocument to <temp_dir>/images.json
    # extracted_image_blocks_path, images_dir = extract_images_from_pdf(
    #     pdf_path, os.path.join(temp_dir, "images.json"), images_dir=os.path.join(temp_dir, "images")
    # )
    # print("Image blocks extracted successfully!")

    # 6. Extract TextBlocks from the PDF and write BlocksDocument to <temp_dir>/text_blocks.json
    # extracted_text_blocks_path: str = extract_text_blocks_with_styling(
    #     pdf_path, os.path.join(temp_dir, "text_blocks.json"), temp_dir
    # )
    # print(f"Text blocks extracted successfully to {extracted_text_blocks_path}!")

    # 7. Get pixmap and extract images from image boxes using PyMuPDF

    # 8. Describe the images and SVGs with a VLM
    # TODO: Pass text context with image to Gemini to improve description
    # TODO: Validate that all ImageBlocks and SvgBlocks have a description after this step
    # descriptions_output_file_path: str = os.path.join(temp_dir, "described_blocks.json")
    # described_blocks_path: str = asyncio.run(describe_images_in_json(
    #     json_file_path=combined_blocks_path,
    #     images_dir=images_dir,
    #     api_key=gemini_api_key,
    #     output_file_path=descriptions_output_file_path,
    #     svg_as_text=True,
    # ))
    # print("Images and SVGs described successfully!")

    # TODO: Blocks' text field contains HTML with in-line styles that still need cleaning
    # Extract unique styles and have an LLM write a transformation rule for each one

    # 6. Convert the blocks to HTML with ids, plaintext, and no bboxes
    # (This is purely about context length management and id tagging for the next step)
    # html_output_file_path: str = os.path.join(temp_dir, "html.html")
    # html_path: str = convert_blocks_to_html(
    #     combined_blocks_path,
    #     html_output_file_path,
    #     rich_text=False,
    #     bboxes=False,
    #     include_ids=True,
    # )
    # print("HTML created successfully!")

    # 7. Detect the structure of the document and return paths to JSON blocksdocs for each section
    # (This is purely about content length management for the next step, since outputs can be max 8k tokens)
    # structure_output_dir: str = os.path.join(temp_dir, "structure")
    # structure_paths: list[tuple[Literal["header", "main", "footer"], str]] = asyncio.run(
    #     detect_structure(
    #         html_path, combined_blocks_path, structure_output_dir, api_key=gemini_api_key
    #     )
    # )
    # print("Structure detected successfully!")

    # 8. Convert the blocks to HTML with rich text and bboxes
    # (reuse the function from step 5 with different parameters)
    # rich_html_output_paths: list[tuple[Literal["header", "main", "footer"], str]] = []
    # for section_name, section_path in structure_paths:
    #     rich_html_output_file_path: str = os.path.join(temp_dir, f"{section_name}.html")
    #     rich_html_output_paths.append(
    #         (
    #             section_name,
    #             convert_blocks_to_html(
    #                 section_path,
    #                 rich_html_output_file_path,
    #                 rich_text=True,
    #                 bboxes=True,
    #                 include_ids=True,
    #             ),
    #         )
    #     )
    #     print(f"Rich HTML created successfully for {section_name}!")

    # 9. Have an LLM clean and conform the HTML to our spec
    # cleaned_html_path: str = asyncio.run(
    #     process_html_inputs_concurrently(
    #         rich_html_output_paths,
    #         os.path.join(temp_dir, "cleaned_document.html"),
    #         api_key=deepseek_api_key,
    #         max_concurrent_calls=3,
    #     )
    # )
    # print("HTML cleaned and assembled successfully!")

    # 10. Transform the cleaned HTML document into a graph matching our schema and ingest it into our database


    # 11. Enrich the database records by generating relations from anchor tags

    print(f"Pipeline completed! All outputs in: {temp_dir}")
