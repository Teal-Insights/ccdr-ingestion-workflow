import os
import re
import time
import asyncio
import base64
from math import ceil
from collections import defaultdict
from typing import List, Tuple, Literal
from litellm import Router, ModelResponse, Choices
import pydantic
import pymupdf

from transform.models import LayoutBlock, BlockType, ContentBlockBase
from utils.schema import EmbeddingSource, PositionalData, BoundingBox


def create_router(openrouter_api_key: str) -> Router:
    """Create a LiteLLM Router with both image and text classifiers."""
    model_list = [
        {
            "model_name": "image-classifier",
            "litellm_params": {
                "model": "openrouter/google/gemini-2.5-flash",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        {
            "model_name": "text-classifier",
            "litellm_params": {
                "model": "openrouter/openai/gpt-4o-mini",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 20,  # Higher for text-only tasks
                "weight": 1,
            }
        }
    ]

    return Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",
        fallbacks=[
            {"image-classifier": ["image-classifier"]},
            {"text-classifier": ["text-classifier"]}
        ],
        num_retries=2,
        allowed_fails=5,
        cooldown_time=30,
        enable_pre_call_checks=True,
        default_max_parallel_requests=50,
        set_verbose=False,
    )


def analyze_text_patterns(text: str) -> dict:
    """Fast analysis of text patterns that indicate visual content"""
    if not text or not text.strip():
        return {'is_empty': True, 'suggests_visual': False, 'char_count': 0}

    # Quick checks for visual content indicators
    newline_count = text.count('\n')
    word_count = len(text.split())
    char_count = len(text.strip())

    # High newline ratio suggests chart labels/axis text
    high_newline_ratio = newline_count > 0 and (newline_count / max(word_count, 1)) > 0.3

    # Pattern matching for axis labels, statistics, dates
    has_visual_patterns = bool(re.search(
        r'\b\d{4}\b|'  # Years
        r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b|'  # Months
        r'\d+%|'  # Percentages
        r'\$\d+|'  # Currency
        r'\b\d+\.\d+\b|'  # Decimals
        r'\b\d+[KMB]\b',  # K/M/B suffixes
        text.upper()
    ))

    return {
        'is_empty': False,
        'suggests_visual': (high_newline_ratio or (has_visual_patterns and char_count < 50)),
        'char_count': char_count,
        'word_count': word_count
    }


def should_use_llm_text_filter(text: str, area: float | int) -> bool:
    """
    Decide if we should use LLM text filtering for this block.
    Only for blocks with substantial text that might be prose.
    """
    if not text or not text.strip():
        return False  # Empty blocks don't need text filtering

    char_count = len(text.strip())
    word_count = len(text.split())

    # Use LLM filtering for blocks with enough text to be potentially problematic
    return char_count > 100 or word_count > 15


class TextContentType(pydantic.BaseModel):
    reasoning: str
    content_type: Literal["prose", "title", "caption", "citation", "data_labels", "other"]


async def classify_text_content_type(router: Router, text: str) -> str:
    """Use LLM to classify the type of text content"""

    prompt = f"""Analyze this text content and classify it as one of the following types:

- "prose": Substantial paragraph text, explanatory content, or academic writing
- "title": Headings, section titles, or chapter names  
- "caption": Figure captions, table captions, or source attributions for figures/tables
- "citation": Academic references, bibliographic entries, source citations, URLs
- "data_labels": Chart labels, axis labels, data points, or statistical values
- "other": Anything else (short phrases, single words, etc.)

Text to classify: "{text}"

Return JSON with:
- reasoning: brief explanation of your classification
- content_type: one of the six types above"""

    message = [{
        "role": "user", 
        "content": prompt
    }]

    response = await router.acompletion(
        model="text-classifier",
        messages=message,  # type: ignore
        temperature=0.0,
        response_format=TextContentType
    )

    try:
        if (response and isinstance(response, ModelResponse) and 
            isinstance(response.choices[0], Choices) and 
            response.choices[0].message.content):
            return response.choices[0].message.content
        else:
            raise ValueError("Invalid response structure from router")
    except Exception as e:
        print(f"Error processing text classification response: {e}")
        # Fallback to allowing classification if LLM fails
        return '{"reasoning": "LLM error", "content_type": "other"}'


def is_image_contained(rect, bbox: BoundingBox) -> bool:
    """Fast check if image rect is contained in bbox"""
    return (rect.x0 >= bbox.x1 and rect.y0 >= bbox.y1 and 
            rect.x1 <= bbox.x2 and rect.y1 <= bbox.y2)


def is_drawing_contained(drawing_rect, bbox: BoundingBox) -> bool:
    """Fast check if drawing rect is contained in bbox"""
    return (drawing_rect.x0 >= bbox.x1 and drawing_rect.y0 >= bbox.y1 and 
            drawing_rect.x1 <= bbox.x2 and drawing_rect.y1 <= bbox.y2)


def is_horizontal_line(drawing: dict, tolerance: float = 2.0) -> bool:
    """
    Check if a drawing is likely a straight horizontal line (e.g., text underline).
    
    Args:
        drawing: PyMuPDF drawing dictionary with 'rect' and potentially other properties
        tolerance: Maximum allowed height for considering something a horizontal line
        
    Returns:
        True if this appears to be a horizontal line that should be filtered out
    """
    if 'rect' not in drawing:
        return False
    
    rect = drawing['rect']
    height = abs(rect.y1 - rect.y0)
    width = abs(rect.x1 - rect.x0)
    
    # Consider it a horizontal line if:
    # 1. Height is very small (within tolerance) 
    # 2. Width is significantly larger than height (at least 10:1 ratio)
    # 3. Width is reasonable (not just a tiny speck)
    if height <= tolerance and width > height * 10 and width > 20:
        return True
    
    return False


async def find_visual_candidates_llm_filtered(
    blocks: list[LayoutBlock], 
    pdf: pymupdf.Document,
    router: Router
) -> Tuple[List[ContentBlockBase], List[int]]:
    """
    Visual content detection with LLM-based text filtering
    """

    blocks_by_page = defaultdict(list)
    for block in blocks:
        blocks_by_page[block.page_number].append(block)

    content_blocks: list[ContentBlockBase] = []
    preliminary_candidates: list[dict[str, str | int | float]] = []  # Candidates before LLM filtering

    detection_stats = {
        'initial_candidates': 0,
        'llm_text_filtered': 0,
        'final_candidates': 0,
        'large_empty': 0,
        'large_text': 0,
        'visual_patterns': 0,
        'images': 0,
        'drawings': 0
    }

    # PHASE 1: Fast heuristic detection (same as before)
    for page_num in sorted(blocks_by_page.keys()):
        page_blocks = blocks_by_page[page_num]
        page = pdf[page_num - 1]

        # Fast resources
        image_list = page.get_images(full=True)
        drawings = page.get_drawings()

        for block in page_blocks:
            positional_data = PositionalData(
                bbox=BoundingBox(
                    x1=int(block.left),
                    y1=int(block.top),
                    x2=int(ceil(block.left + block.width)),
                    y2=int(ceil(block.top + block.height)),
                ),
                page_pdf=block.page_number,
                page_logical=block.logical_page_number,
            )

            new_block_type = block.type
            needs_reclassification = False

            if block.type == BlockType.TEXT:
                bbox = positional_data.bbox
                area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
                text_analysis = analyze_text_patterns(block.text)

                # Fast heuristic checks (same as defensive approach)

                # 1. Large empty blocks (likely visual content)
                if text_analysis['is_empty']:
                    if area > 5000:
                        needs_reclassification = True
                        detection_stats['large_empty'] += 1

                # 1b. Large blocks with text (potential infographics)
                elif area > 200000:
                    needs_reclassification = True
                    detection_stats['large_text'] += 1

                # 2. Text patterns suggesting visual content
                elif text_analysis['suggests_visual']:
                    needs_reclassification = True
                    detection_stats['visual_patterns'] += 1

                # 3. Image containment
                elif image_list:
                    for xref, *_rest in image_list:
                        rect_list = page.get_image_rects(xref)
                        if any(is_image_contained(rect, bbox) for rect in rect_list):
                            new_block_type = BlockType.PICTURE
                            detection_stats['images'] += 1
                            break

                # 4. PyMuPDF drawings
                if not needs_reclassification and not (new_block_type == BlockType.PICTURE) and drawings:
                    drawings_in_bbox = []
                    for drawing in drawings:
                        if 'rect' in drawing and is_drawing_contained(drawing['rect'], bbox):
                            drawings_in_bbox.append(drawing)
                    
                    if drawings_in_bbox:
                        # Apply horizontal line filtering only when there are few drawings in this bbox
                        # (likely underlines), but not when there are many (likely legitimate graphics)
                        should_filter_horizontal_lines = len(drawings_in_bbox) < 3
                        
                        # Check each drawing in the bbox
                        has_non_horizontal_drawing = False
                        for drawing in drawings_in_bbox:
                            # Skip horizontal lines if filtering is enabled and this looks like an underline
                            if should_filter_horizontal_lines and is_horizontal_line(drawing):
                                continue
                            
                            # Found a non-horizontal drawing, trigger reclassification
                            has_non_horizontal_drawing = True
                            break
                        
                        if has_non_horizontal_drawing:
                            needs_reclassification = True
                            detection_stats['drawings'] += 1

                if needs_reclassification:
                    detection_stats['initial_candidates'] += 1
                    preliminary_candidates.append({
                        'content_block_index': len(content_blocks),
                        'text': block.text,
                        'area': area,
                        'page': block.page_number
                    })

            content_blocks.append(ContentBlockBase(
                positional_data=positional_data,
                block_type=new_block_type,
                embedding_source=get_embedding_source(new_block_type),
                text_content=block.text,
            ))

    print(f"Found {len(preliminary_candidates)} preliminary candidates")

    # PHASE 2: LLM-based text filtering
    final_indices_to_reclassify: list[int] = []

    if preliminary_candidates:
        # Filter candidates that need LLM text classification
        llm_filter_candidates = []
        auto_approve_candidates = []

        for candidate in preliminary_candidates:
            if isinstance(candidate['text'], str) and isinstance(candidate['area'], (int, float)):
                if should_use_llm_text_filter(candidate['text'], candidate['area']):
                    llm_filter_candidates.append(candidate)
                else:
                    # Auto-approve short text, empty blocks, or obvious chart labels
                    auto_approve_candidates.append(candidate)
            else:
                raise ValueError(f"Invalid candidate: {candidate}")

        print(f"Auto-approving {len(auto_approve_candidates)} candidates")
        print(f"LLM filtering {len(llm_filter_candidates)} candidates")

        # Add auto-approved candidates
        for candidate in auto_approve_candidates:
            if isinstance(candidate['content_block_index'], int):
                final_indices_to_reclassify.append(candidate['content_block_index'])
            else:
                raise ValueError(f"Invalid candidate: {candidate}")

        # Process LLM filtering candidates
        if llm_filter_candidates:
            llm_tasks = []
            for candidate in llm_filter_candidates:
                if isinstance(candidate['text'], str):
                    llm_tasks.append(classify_text_content_type(router, candidate['text']))
                else:
                    raise ValueError(f"Invalid candidate: {candidate}")

            llm_responses = await asyncio.gather(*llm_tasks)

            for i, response in enumerate(llm_responses):
                candidate = llm_filter_candidates[i]
                try:
                    classification = TextContentType.model_validate_json(response)

                    # Only allow reclassification if it's NOT prose, title, caption, or citation
                    if classification.content_type in ['data_labels', 'other']:
                        if isinstance(candidate['content_block_index'], int):
                            final_indices_to_reclassify.append(candidate['content_block_index'])
                            detection_stats['final_candidates'] += 1
                        else:
                            raise ValueError(f"Invalid candidate: {candidate}")
                    else:
                        detection_stats['llm_text_filtered'] += 1
                        print(f"LLM FILTERED Page {candidate['page']}: {classification.content_type} - {classification.reasoning}")

                except Exception as e:
                    print(f"Error parsing LLM response: {e}")
                    # If LLM fails, err on the side of allowing classification
                    if isinstance(candidate['content_block_index'], int):
                        final_indices_to_reclassify.append(candidate['content_block_index'])
                    else:
                        raise ValueError(f"Invalid candidate: {candidate}")

    print(f"Detection stats: {detection_stats}")
    return content_blocks, final_indices_to_reclassify


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
    if block_type == BlockType.PICTURE:
        return EmbeddingSource.DESCRIPTION
    else:
        return EmbeddingSource.TEXT_CONTENT


async def classify_single_image_with_router(router: Router, message: list) -> str:
    """Classify single image using router"""
    response = await router.acompletion(
        model="image-classifier",
        messages=message, 
        temperature=0.0,
        response_format=ImageClassification
    )

    try:
        if (response and isinstance(response, ModelResponse) and 
            isinstance(response.choices[0], Choices) and 
            response.choices[0].message.content):
            return response.choices[0].message.content
        else:
            raise ValueError("Invalid response structure from router")
    except Exception as e:
        print(f"Error processing router response: {e}")
        raise


async def reclassify_images_llm_filtered(
    content_blocks: list[ContentBlockBase], 
    pdf_path: str, 
    router: Router
) -> list[BlockType]:
    """Image reclassification with improved prompt"""
    
    prompt = """Here is a layout block extracted from a PDF. Classify it as text, picture, or table.

IMPORTANT GUIDELINES:
- Only classify as "picture" if it's clearly a visual element like charts, diagrams, images, or figures
- Only classify as "table" if it shows structured data in rows/columns
- If in doubt between text and picture, choose "text"

You should return a JSON object with:
- reason: brief explanation of your classification
- classification: "text", "picture", or "table"
"""

    pdf = pymupdf.open(pdf_path)
    messages = []

    for content_block in content_blocks:
        page = pdf[content_block.positional_data.page_pdf - 1]
        image = page.get_pixmap(clip=content_block.positional_data.bbox)
        img_base64 = base64.b64encode(image.tobytes()).decode("utf-8")

        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
            ],
        }]
        messages.append(message)

    pdf.close()

    tasks = [classify_single_image_with_router(router, msg) for msg in messages]
    responses = await asyncio.gather(*tasks)
    return [get_classification(response) for response in responses]


async def reclassify_block_types(
    blocks: list[LayoutBlock], 
    pdf_path: str, 
    openrouter_api_key: str
) -> list[ContentBlockBase]:
    """LLM-filtered reclassification workflow"""

    start_time = time.time()

    router = create_router(openrouter_api_key)
    pdf = pymupdf.open(pdf_path)

    print("Finding visual candidates with LLM text filtering...")
    candidates_start = time.time()

    content_blocks, indices_to_reclassify = await find_visual_candidates_llm_filtered(blocks, pdf, router)

    candidates_time = time.time() - candidates_start
    print(f"LLM-filtered candidate detection: {candidates_time:.2f}s")

    if indices_to_reclassify:
        print(f"Reclassifying {len(indices_to_reclassify)} blocks with image classifier")

        reclassify_start = time.time()
        blocks_to_reclassify = [content_blocks[i] for i in indices_to_reclassify]

        classifications = await reclassify_images_llm_filtered(
            blocks_to_reclassify, pdf_path, router
        )

        for i, classification in enumerate(classifications):
            content_blocks[indices_to_reclassify[i]].block_type = classification
            content_blocks[indices_to_reclassify[i]].embedding_source = get_embedding_source(classification)

        reclassify_time = time.time() - reclassify_start
        print(f"LLM-filtered reclassification: {reclassify_time:.2f}s")

    pdf.close()

    total_time = time.time() - start_time
    print(f"Total LLM-filtered time: {total_time:.2f}s")

    return content_blocks


if __name__ == "__main__":
    import json
    import asyncio
    import dotenv

    dotenv.load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    assert OPENROUTER_API_KEY, "OPENROUTER_API_KEY is not set"

    with open(os.path.join("artifacts", "doc_601_with_logical_page_numbers.json"), "r") as f:
        layout_blocks = json.load(f)
        layout_blocks = [LayoutBlock.model_validate(block) for block in layout_blocks]

    print(f"Loaded {len(layout_blocks)} layout blocks")

    start_time = time.time()
    content_blocks = asyncio.run(reclassify_block_types(
        layout_blocks, 
        "artifacts/wkdir/doc_601.pdf", 
        OPENROUTER_API_KEY
    ))
    total_time = time.time() - start_time

    print(f"LLM-filtered re-classified {len(content_blocks)} content blocks in {total_time:.2f}s")

    with open(os.path.join("artifacts", "doc_601_content_blocks_llm_filtered.json"), "w") as f:
        json.dump([block.model_dump() for block in content_blocks], f, indent=2)