# TODO: Ignore text blocks that aren't visible; use playwright's element.is_visible()
# (Note: this actually doesn't work unless you're working with the SVG version of the page)
# TODO: See how pymupdf handles tables, and what we need to do to handle them
# TODO: Investigate "no matching elements found" warnings on about 50% of blocks;
# should these be picture blocks?

import re
from typing import cast
from pathlib import Path
from collections import defaultdict
import pymupdf
from bs4 import BeautifulSoup, Tag

from transform.models import ContentBlock, BlockType, PositionalData
from utils.schema import EmbeddingSource


def preprocess_json_structure(json_data: dict) -> list[dict]:
    """
    Extract line-level text elements from PyMuPDF JSON structure with rotation awareness.
    
    This processes individual lines within text blocks rather than entire blocks,
    and includes rotation information for proper coordinate matching.
    """
    elements = []
    for block in json_data["blocks"]:
        # Text blocks (type 0)
        if block["type"] == 0:
            for line in block["lines"]:
                text = "".join(span["text"] for span in line["spans"])
                # Convert PyMuPDF bbox tuple (x0, y0, x1, y1) to dictionary format
                bbox_tuple = line["bbox"]  # Use line bbox, not block bbox
                bbox_dict = {
                    "x1": int(bbox_tuple[0]),  # x0 -> x1
                    "y1": int(bbox_tuple[1]),  # y0 -> y1
                    "x2": int(bbox_tuple[2]),  # x1 -> x2
                    "y2": int(bbox_tuple[3])   # y1 -> y2
                }
                # Extract rotation information from dir field
                direction = line.get("dir", [1.0, 0.0])
                is_rotated = direction != [1.0, 0.0]
                rotation_angle = get_rotation_angle(direction)
                
                elements.append({
                    "text": text.strip(),
                    "bbox": bbox_dict,
                    "type": "text",
                    "spans": line["spans"],  # Include spans for style information
                    "direction": direction,
                    "is_rotated": is_rotated,
                    "rotation_angle": rotation_angle
                })
    return elements


def get_rotation_angle(direction: list[float]) -> float:
    """Convert direction vector to rotation angle in degrees"""
    if direction == [1.0, 0.0]:
        return 0.0      # Normal horizontal text
    elif direction == [0.0, -1.0]:
        return 90.0     # Rotated 90° clockwise (vertical, top-to-bottom)
    elif direction == [-1.0, 0.0]:
        return 180.0    # Rotated 180°
    elif direction == [0.0, 1.0]:
        return 270.0    # Rotated 270° clockwise (bottom-to-top)
    else:
        # Calculate angle from direction vector
        import math
        return math.degrees(math.atan2(-direction[1], direction[0]))


def calculate_rotation_aware_distance(html_x: float, html_y: float, json_elem: dict) -> float:
    """Calculate distance taking text rotation into account"""
    bbox = json_elem["bbox"]
    
    if not json_elem["is_rotated"]:
        # Normal text: use top-left corner
        json_x, json_y = bbox["x1"], bbox["y1"]
    else:
        # Rotated text: adjust based on rotation
        rotation = json_elem["rotation_angle"]
        
        if rotation == 90.0:  # [0.0, -1.0] - vertical, top-to-bottom
            # For 90° rotation, HTML positioning seems to use a different reference point
            # Use the bottom-left of the bbox (x1, y2) as the reference
            json_x, json_y = bbox["x1"], bbox["y2"]
        elif rotation == 180.0:  # [-1.0, 0.0]
            # For 180° rotation, use bottom-right
            json_x, json_y = bbox["x2"], bbox["y2"]
        elif rotation == 270.0:  # [0.0, 1.0]
            # For 270° rotation, use top-right
            json_x, json_y = bbox["x2"], bbox["y1"]
        else:
            # Default to top-left for unknown rotations
            json_x, json_y = bbox["x1"], bbox["y1"]
    
    # Calculate Euclidean distance
    return ((json_x - html_x) ** 2 + (json_y - html_y) ** 2) ** 0.5


def match_html_to_json(html_content: str, page_json: dict) -> list[tuple[Tag, dict[str, int]]]:
    """
    Match HTML elements to JSON elements based on rotation-aware coordinate distance.
    
    This implementation uses coordinate proximity to match HTML p elements to 
    individual text lines from the PyMuPDF JSON structure, handling text rotation properly.
    """
    json_text_elements = preprocess_json_structure(page_json)
    
    soup = BeautifulSoup(html_content, "html.parser")
    page_div = soup.find("div")
    matches = []

    # Ensure we found a div element as expected
    assert page_div is not None, "Expected to find a div element in PDF-generated HTML, but none was found"
    assert isinstance(page_div, Tag), "Expected div to be a Tag element"

    p_elements = page_div.find_all(["p"])

    # Track which JSON elements have been matched to avoid double-matching
    used_json_indices = set()

    for element in p_elements:
        # Ensure we're working with a Tag object
        if isinstance(element, Tag):
            style_attr = element.get("style")
            style = str(style_attr) if style_attr is not None else ""
            y_1_match = re.search(r"top:([\d.]+)pt", style)
            x_1_match = re.search(r"left:([\d.]+)pt", style)

            if not (y_1_match and x_1_match):
                continue

            y_1, x_1 = float(y_1_match.group(1)), float(x_1_match.group(1))
            
            # Find the closest JSON element by rotation-aware distance
            best_match = None
            best_distance = float('inf')
            best_index = -1
            
            for j, json_elem in enumerate(json_text_elements):
                if j in used_json_indices:
                    continue
                    
                if json_elem["type"] == "text":
                    # Calculate rotation-aware distance
                    distance = calculate_rotation_aware_distance(x_1, y_1, json_elem)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = json_elem
                        best_index = j

            # Accept the closest match if it's within reasonable tolerance
            if best_match and best_distance < 20.0:  # 20pt tolerance
                matches.append((element, best_match["bbox"]))
                used_json_indices.add(best_index)

    return matches


def is_contained(html_bbox, target_bbox, margin_of_error=20) -> bool:
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
            # Append image blocks without changes
            if content_block.block_type in [
                BlockType.PICTURE
            ]:
                modified_content_blocks.append(content_block)
                continue

            # Discard header/footer block
            if content_block.block_type in [BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER]:
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

            # For each matching element, we will concatenate the text content with <br/> and add it to the block
            text_contents: list[str] = []
            for element, _ in matching_elements:
                # Strip out the `span` tags, but preserve the `b`, `i`, `u`, `s`, `sup`, `sub` tags.
                if isinstance(element, Tag):
                    # Find all span tags and unwrap them (remove tag but keep content)
                    span_tags = element.find_all('span')
                    for span in span_tags:
                        if isinstance(span, Tag):
                            span.unwrap()
                text_contents.append(element.get_text().strip())

            text_content = "<br/>".join(text_contents)

            modified_content_blocks.append(
                ContentBlock(
                    positional_data=PositionalData(
                        page_pdf=content_block.positional_data.page_pdf,
                        page_logical=content_block.positional_data.page_logical,
                        bbox=content_block.positional_data.bbox,
                    ),
                    block_type=content_block.block_type,
                    text_content=text_content,
                    embedding_source=EmbeddingSource.TEXT_CONTENT,
                )
            )

    # Close the document
    doc.close()

    return modified_content_blocks


if __name__ == "__main__":
    import os
    import json

    pdf_path: str = "./artifacts/wkdir/doc_601.pdf"
    temp_dir: str = "./artifacts"

    with open(os.path.join("artifacts", "doc_601_content_blocks_with_descriptions.json"), "r") as fr:
        content_blocks: list[ContentBlock] = json.load(fr)
        content_blocks = [ContentBlock.model_validate(block) for block in content_blocks]
    print(f"Loaded {len(content_blocks)} content blocks before text extraction")
    print(f"Headers and footers: {len([block for block in content_blocks if block.block_type in [BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER]])}")
    content_blocks_with_text: list[ContentBlock] = extract_text_blocks(content_blocks, pdf_path, temp_dir)
    print(f"Extracted text for {len(content_blocks_with_text)} content blocks")
    with open(os.path.join("artifacts", "doc_601_content_blocks_with_text.json"), "w") as fw:
        json.dump([block.model_dump() for block in content_blocks_with_text], fw, indent=2)