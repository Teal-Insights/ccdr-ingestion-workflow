# TODO: Move text_content into ContentBlockBase and save the original text_content
# TODO: We should discard any detected LayoutBlocks whose bbox is fully outside the page area

import pymupdf
from transform.models import LayoutBlock, BlockType, ContentBlockBase
from utils.schema import EmbeddingSource, PositionalData
from transform.extract_drawings_by_bbox import is_completely_contained, is_bbox_contained
from utils.svg import has_geometry, is_font_element
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import base64
import os
from math import ceil
import dotenv
from litellm import acompletion
from collections import defaultdict
from typing import Dict, List
from svgelements import SVG, SVGElement
from io import StringIO
import pydantic
from typing import Literal

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class ImageClassification(pydantic.BaseModel):
    reason: str
    classification: Literal["text", "picture", "table"]


def get_classification(response: str) -> BlockType:
    classification = ImageClassification.model_validate_json(response)
    if classification.classification == "picture":
        return BlockType.PICTURE
    elif classification.classification == "table":
        return BlockType.TABLE
    elif classification.classification == "text":
        return BlockType.TEXT
    else:
        print(f"Warning: LLM returned invalid classification ({classification.classification}); falling back to 'text'")
        return BlockType.TEXT


def get_embedding_source(block_type: BlockType) -> EmbeddingSource:
    if block_type == BlockType.TEXT:
        return EmbeddingSource.TEXT_CONTENT
    elif block_type == BlockType.PICTURE:
        return EmbeddingSource.DESCRIPTION


def extract_page_geometries(svg_content: str) -> List[SVGElement]:
    """
    Extract all non-font SVG geometries from a page's SVG content.
    This replaces the per-block extraction to avoid redundant parsing.
    """
    # Parse the SVG
    svg = SVG.parse(StringIO(svg_content), reify=True)
    
    geometries = []
    
    # Iterate through all elements in the SVG
    for element in svg.elements():
        # Skip containers and non-geometric elements
        if isinstance(element, SVG) or not hasattr(element, 'bbox'):
            continue

        try:
            # Skip if element is a font element
            if is_font_element(element):
                continue
            
            # Skip elements without actual geometry
            if not has_geometry(element):
                continue
                
            geometries.append(element)
            
        except Exception as e:
            # Some elements might not have proper bbox calculation
            print(f"Warning: Could not process element {type(element)}: {e}")
            continue

    return geometries


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def _classify_single_image(
    message: list, semaphore: asyncio.Semaphore
) -> str:
    """Helper function to classify a single image with semaphore control."""
    async with semaphore:
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=message,
            temperature=0.0,
            response_format={
                "type": "json_object",
                "response_schema": ImageClassification.model_json_schema(),
            }
        )
        return response.choices[0].message.content


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def reclassify_images_with_gemini(
    content_blocks: list[ContentBlockBase], pdf_path: str, api_key: str, semaphore: asyncio.Semaphore
) -> list[BlockType]:
    """Use Gemini to reclassify the image with async retry logic."""
    # Set environment variable for LiteLLM
    os.environ["GEMINI_API_KEY"] = api_key

    prompt = """Here is a layout block, extracted from a PDF. Classify it as a text
block, a picture/figure block, or a table block.

Charts should be classified as "picture", even if the legends or axes contain text.
If the block contains more than one chart, return "picture".
If it contains a combination of text and decorative elements (such as logos or icons),
return "text".

You should return a JSON object with the following fields:
- reason: a short explanation of why you made this classification
- classification: "text", "picture", or "table"
"""

    messages = []
    for content_block in content_blocks:
        # Use positional data to get pixmap for the page and crop to the bbox
        page: pymupdf.Page = pymupdf.open(pdf_path)[content_block.positional_data.page_pdf - 1]
        image = page.get_pixmap(clip=content_block.positional_data.bbox)

        # Convert PIL Image to base64
        img_base64 = base64.b64encode(image.tobytes()).decode("utf-8")

        # Create message content with text and image - this should be a list of message objects
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ],
            }
        ]
        messages.append(message)

    # Create tasks for concurrent API calls
    tasks = []
    for message in messages:
        tasks.append(_classify_single_image(message, semaphore))

    responses = await asyncio.gather(*tasks)
    return [get_classification(response) for response in responses]


async def reclassify_block_types(blocks: list[LayoutBlock], pdf_path: str) -> list[ContentBlockBase]:
    """Reclassify block types based on the content of the bbox, optimized to process by page"""
    pdf = pymupdf.open(pdf_path)

    # Group blocks by page number for efficient processing
    blocks_by_page: Dict[int, List[LayoutBlock]] = defaultdict(list)
    for block in blocks:
        blocks_by_page[block.page_number].append(block)

    content_blocks: list[ContentBlockBase] = []
    indices_to_reclassify: list[int] = []

    # Process each page
    for page_num in sorted(blocks_by_page.keys()):
        page_blocks = blocks_by_page[page_num]

        print(f"Processing page {page_num} with {len(page_blocks)} blocks")

        # Extract page SVG content and geometries once per page
        page = pdf[page_num - 1]
        page_svg_content = page.get_svg_image()
        page_geometries = extract_page_geometries(page_svg_content)

        # Extract image list once per page
        image_list = page.get_images(full=True)

        # Process each block on this page
        for block in page_blocks:
            # Coerce positional data to our own schema
            positional_data = PositionalData(
                bbox={
                    "x1": int(block.left),
                    "y1": int(block.top),
                    "x2": int(ceil(float(block.left) + float(block.width))),
                    "y2": int(ceil(float(block.top) + float(block.height))),
                },
                page_pdf=block.page_number,
                page_logical=block.logical_page_number,
            )

            # If a text block, check for images or drawings in the bbox
            new_block_type = block.type
            if block.type == BlockType.TEXT:
                # Check for images in bbox using pre-extracted image list
                for xref, *_rest in image_list:
                    rect_list = page.get_image_rects(xref)
                    for rect in rect_list:
                        if is_completely_contained(rect, positional_data.bbox):
                            new_block_type = BlockType.PICTURE
                            break
                    if new_block_type == BlockType.PICTURE:
                        break

                # Check for drawings in bbox using pre-extracted geometries
                for geometry in page_geometries:
                    try:
                        element_bbox = geometry.bbox()
                        if element_bbox is None:
                            continue

                        # Check if geometry is contained within block bbox using optimized function
                        if is_bbox_contained(element_bbox, positional_data.bbox):
                            indices_to_reclassify.append(len(content_blocks))
                            break
                    except Exception as e:
                        print(f"Warning: Could not check geometry bbox: {e}")
                        continue

            content_blocks.append(ContentBlockBase(
                positional_data=positional_data,
                block_type=new_block_type,
                embedding_source=get_embedding_source(new_block_type),
                text_content=block.text,
            ))

    pdf.close()

    if indices_to_reclassify:
        print(f"Reclassifying {len(indices_to_reclassify)} blocks with Gemini")
        semaphore = asyncio.Semaphore(2)
        blocks_to_reclassify = [content_blocks[i] for i in indices_to_reclassify]
        classifications = await reclassify_images_with_gemini(
            blocks_to_reclassify, pdf_path, GEMINI_API_KEY, semaphore
        )
        for i, classification in enumerate(classifications):
            content_blocks[indices_to_reclassify[i]].block_type = classification
            content_blocks[indices_to_reclassify[i]].embedding_source = get_embedding_source(classification)

    # Exclude content blocks of non-picture type that have no text content
    content_blocks = [block for block in content_blocks if block.block_type == BlockType.PICTURE or block.text_content.strip()]
    return content_blocks


if __name__ == "__main__":
    import json
    import asyncio
    with open(os.path.join("artifacts", "doc_601_with_logical_page_numbers.json"), "r") as f:
        layout_blocks = json.load(f)
        layout_blocks = [LayoutBlock.model_validate(block) for block in layout_blocks]
    print(f"Loaded {len(layout_blocks)} layout blocks")
    content_blocks = asyncio.run(reclassify_block_types(layout_blocks, "artifacts/wkdir/doc_601.pdf"))
    print(f"Re-classified {len(content_blocks)} content blocks")
    with open(os.path.join("artifacts", "doc_601_content_blocks.json"), "w") as f:
        json.dump([block.model_dump() for block in content_blocks], f, indent=2)