# TODO: Ignore text blocks that aren't visible; use playwright's element.is_visible()
# (Note: this actually doesn't work unless you're working with the SVG version of the page)

import re
from typing import cast
from pathlib import Path
from collections import defaultdict
import pymupdf
from bs4 import BeautifulSoup, Tag
import copy

from transform.models import ContentBlock, BlockType, PositionalData


def preprocess_json_structure(json_data: dict) -> list[dict]:
    elements = []
    for block in json_data["blocks"]:
        # Text blocks (type 0)
        if block["type"] == 0:
            for line in block["lines"]:
                text = "".join(span["text"] for span in line["spans"])
                # Convert PyMuPDF bbox tuple (x0, y0, x1, y1) to dictionary format
                bbox_tuple = line["bbox"]
                bbox_dict = {
                    "x1": int(bbox_tuple[0]),  # x0 -> x1
                    "y1": int(bbox_tuple[1]),  # y0 -> y1
                    "x2": int(bbox_tuple[2]),  # x1 -> x2
                    "y2": int(bbox_tuple[3])   # y1 -> y2
                }
                elements.append({
                    "text": text.strip(),
                    "bbox": bbox_dict,
                    "type": "text"
                })
    return elements


def match_html_to_json(html_content: str, page_json: dict) -> list[tuple[Tag, dict[str, int]]]:
    """
    Match HTML elements to JSON elements based on position and text content.
    """
    json_text_elements = preprocess_json_structure(page_json)
    
    soup = BeautifulSoup(html_content, "html.parser")
    page_div = soup.find("div")
    matches = []

    # Ensure we found a div element as expected
    assert page_div is not None, "Expected to find a div element in PDF-generated HTML, but none was found"
    assert isinstance(page_div, Tag), "Expected div to be a Tag element"

    p_elements = page_div.find_all(["p"])

    for element in p_elements:
        # Ensure we're working with a Tag object
        if isinstance(element, Tag):
            text = element.get_text().strip()
            style_attr = element.get("style")
            style = str(style_attr) if style_attr is not None else ""
            y_1_match = re.search(r"top:([\d.]+)pt", style)
            x_1_match = re.search(r"left:([\d.]+)pt", style)

            if not (y_1_match and x_1_match):
                continue

            y_1, x_1 = float(y_1_match.group(1)), float(x_1_match.group(1))
            
            # Find matching JSON element
            for json_elem in json_text_elements:
                if (
                    json_elem["type"] == "text" and
                    json_elem["text"] == text and
                    abs(json_elem["bbox"]["y1"] - y_1) < 5.0 and
                    abs(json_elem["bbox"]["x1"] - x_1) < 5.0
                ):
                    matches.append((element, json_elem["bbox"]))
                    break

    return matches


def is_contained(html_bbox, target_bbox, margin_of_error=10) -> bool:
    """Check if html_bbox is fully inside target_bbox (within some margin of error)."""
    return (
        html_bbox["x1"] >= target_bbox["x1"] - margin_of_error and
        html_bbox["y1"] >= target_bbox["y1"] - margin_of_error and
        html_bbox["x2"] <= target_bbox["x2"] + margin_of_error and
        html_bbox["y2"] <= target_bbox["y2"] + margin_of_error
    )


def extract_text_blocks(
    content_blocks: list[ContentBlock], pdf_path: str, temp_dir: str | None = None
) -> list[ContentBlock]:
    """
    For each ContentBlock of a text-like type, extract the HTML elements positioned
    within its bounding box.
    
    1. We must strip out the `span` tags from the text elements, but preserve the
    `b`, `i`, `u`, `s`, `sup`, `sub` tags.
    2. If there is more than one text element in the bounding box, we should split
    into multiple content blocks.

    Args:
        pdf_path: Path to the PDF file to process
        output_filename: Full path to the output JSON file
        temp_dir: Directory to use for temporary files (optional, creates one if not provided)

    Returns:
        Path to the output JSON file

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If there's an error processing the PDF
    """
    # Check if PDF file exists
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file '{pdf_path}' not found")

    # Open the PDF document
    doc: pymupdf.Document = pymupdf.open(pdf_path)

    # Group blocks by page number for efficient processing
    blocks_by_page: dict[int, list[ContentBlock]] = defaultdict(list)
    for block in content_blocks:
        blocks_by_page[block.positional_data.page_pdf].append(block)

    # Process each page
    modified_content_blocks: list[ContentBlock] = []
    for page_num in range(len(doc)):
        page: pymupdf.Page = cast(pymupdf.Page, doc[page_num])
        page_html = page.get_text("html")
        page_json = page.get_text("dict")
        p_elements_with_bbox = match_html_to_json(page_html, page_json)

        # Process each text block and find matching HTML elements
        for content_block in blocks_by_page[page_num]:
            # Skip empty blocks and non-text blocks
            if content_block.block_type in [
                BlockType.PICTURE, BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER
            ]:
                continue

            block_bbox: dict[str, int] = content_block.positional_data.bbox

            # Find HTML elements contained in the content block
            matching_elements = [
                (element, element_bbox) for element, element_bbox in p_elements_with_bbox
                if is_contained(element_bbox, block_bbox)
            ]

            if not matching_elements:
                print(f"Warning: No matching elements found for content block on page {page_num}; skipping")
                print(f"Original text detected by LayoutLM: {content_block.text_content}")
                continue

            # For each matching element, we must insert a copy of the content block
            # with the text content of the matching element.
            for element, element_bbox in matching_elements:
                # But first we must strip out the `span` tags, but preserve the
                # `b`, `i`, `u`, `s`, `sup`, `sub` tags.
                cleaned_element = copy.deepcopy(element)
                if isinstance(cleaned_element, Tag):
                    # Find all span tags and unwrap them (remove tag but keep content)
                    span_tags = cleaned_element.find_all('span')
                    for span in span_tags:
                        if isinstance(span, Tag):
                            span.unwrap()

                modified_content_blocks.append(
                    ContentBlock(
                        positional_data=PositionalData(
                            page_pdf=content_block.positional_data.page_pdf,
                            page_logical=content_block.positional_data.page_logical,
                            # element bbox becomes source of truth
                            bbox=element_bbox,
                        ),
                        block_type=content_block.block_type,
                        text_content=cleaned_element.get_text().strip(),
                    )
                )

    # Close the document
    doc.close()

    return modified_content_blocks


if __name__ == "__main__":
    import pickle
    import os
    import json

    pdf_path: str = "./artifacts/wkdir/doc_601.pdf"
    temp_dir: str = "./artifacts"

    with open(os.path.join("artifacts", "content_blocks_with_descriptions.pkl"), "rb") as fr:
        content_blocks: list[ContentBlock] = pickle.load(fr)
    print(f"Loaded {len(content_blocks)} content blocks before text extraction")
    content_blocks_with_text: list[ContentBlock] = extract_text_blocks(content_blocks, pdf_path, temp_dir)
    print(f"Extracted text for {len(content_blocks_with_text)} content blocks")
    with open(os.path.join("artifacts", "content_blocks_with_text.json"), "w") as fw:
        json.dump([block.model_dump() for block in content_blocks_with_text], fw, indent=2)