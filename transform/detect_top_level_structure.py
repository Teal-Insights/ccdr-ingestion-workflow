import logging
from typing import List, Optional
import pydantic
from pydantic import ValidationError
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices
from litellm import Router

from transform.models import ContentBlock
from utils.schema import TagName
from utils.litellm_router import create_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopLevelStructure(pydantic.BaseModel):
    """Pydantic model for document structure detection response"""

    header: str = pydantic.Field(
        description="Comma-separated ranges for header blocks (e.g., '1-3,5,7-9')"
    )
    main: str = pydantic.Field(
        description="Comma-separated ranges for main content blocks (e.g., '4,6,10-12')"
    )
    footer: str = pydantic.Field(
        description="Comma-separated ranges for footer blocks (e.g., '13-15')"
    )

assert all(
    field in TopLevelStructure.model_fields
    for field in [TagName.HEADER, TagName.MAIN, TagName.FOOTER]
)

TOP_LEVEL_PROMPT = """Analyze the following HTML content (extracted from a PDF) and identify the document structure. The HTML elements are annotated with sequential id numbers, e.g., <p id="1">, <p id="2">, etc.).

Your task is to categorize these elements into three sections:

- {{header}}: Front matter (title pages, table of contents, preface, etc.)
- {{main}}: Body content (main chapters, sections, core content)
- {{footer}}: Back matter (appendices, bibliography, index, etc.)

Return the id numbers for each section as comma-separated ranges. Use hyphenated ranges for consecutive ids (e.g., "1-3" to include the elements with id="1", "2", and "3") and comma-separated individual ids or ranges (e.g., "1-3,5,7-9" to skip the elements with id="4" and "6").

Rules:
1. Every element must be assigned to exactly one section
2. Any section can be empty (use empty string "")
3. Use the actual content and layout to determine the structure
4. Consider typical document patterns (title pages at start, references at end, etc.)

HTML Content:
{html_content}

Respond with a JSON object containing the three sections."""


def parse_range_string(range_str: str) -> List[int]:
    """
    Parse a comma-separated range string into a list of integers.

    Args:
        range_str: String like "1-3,5,7-9" or empty string

    Returns:
        List of integers representing all tag ids in the ranges
    """
    if not range_str.strip():
        return []

    ids: list[int] = []
    parts = range_str.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            # Handle range like "1-3"
            start, end = part.split("-", 1)
            start_num = int(start.strip())
            end_num = int(end.strip())
            ids.extend(range(start_num, end_num + 1))
        else:
            # Handle single number like "5"
            ids.append(int(part))

    return sorted(ids)


def validate_tag_coverage(
    header_ids: List[int],
    main_ids: List[int],
    footer_ids: List[int],
    total_ids: int,
) -> None:
    """
    Validate that all ids from 1 to total_ids are covered exactly once.

    Args:
        header_ids: List of id numbers in header section
        main_ids: List of id numbers in main section
        footer_ids: List of id numbers in footer section
        total_ids: Total number of ids that should be covered

    Raises:
        ValueError: If ids are missing, duplicated, or out of range
    """
    all_assigned_ids = set(header_ids + main_ids + footer_ids)
    expected_ids = set(range(total_ids))

    assert len(all_assigned_ids) == len(header_ids) + len(main_ids) + len(footer_ids), f"Assigned ids do not match expected ids. Duplicates: {all_assigned_ids - expected_ids}"
    assert all_assigned_ids == expected_ids, f"Assigned ids do not match expected ids. Differences: {expected_ids - all_assigned_ids}"


def filter_blocks_by_numbers(
    all_blocks: List[ContentBlock], target_id_numbers: List[int]
) -> List[ContentBlock]:
    """
    Filter blocks to only include those whose index is in the specified id list.

    Args:
        all_blocks: List of content blocks
        target_id_numbers: List of tag id numbers to include

    Returns:
        Filtered list of HTML elements
    """
    # Tags should be listed in numerical order, so we can retrieve by index
    blocks = [all_blocks[i] for i in target_id_numbers]
    for id_number, block in zip(target_id_numbers, blocks):
        assert f"{id_number}" in block.to_html(bboxes=False, block_id=id_number), f"id '{id_number}' not found in html"

    return blocks


async def detect_top_level_structure(
    content_blocks: list[ContentBlock],
    router: Router,
    messages: Optional[list[dict[str, str]]] = None,
    max_validation_attempts: int = 3,
    attempt: int = 0,
) -> list[tuple[TagName, list[ContentBlock]]]:
    """
    Use LLM to analyze HTML content and detect document structure.

    Args:
        content_blocks: List of content blocks
        router: LiteLLM Router for LLM calls
        messages: Optional list of messages for retry attempts
        max_validation_attempts: Maximum number of validation attempts
        attempt: Current attempt number

    Returns:
        List of tuples with (TagName, blocks) for each non-empty section
    """
    # Prepare the prompt
    messages = messages or [{
        "role": "user",
        "content": TOP_LEVEL_PROMPT.format(
            html_content="\n".join([block.to_html(bboxes=True, block_id=i) for i, block in enumerate(content_blocks)]),
            header=TagName.HEADER.value,
            main=TagName.MAIN.value,
            footer=TagName.FOOTER.value,
        )
    }]

    # Make the LLM call
    try:
        response: ModelResponse = await router.acompletion(
            model="structure-detector",
            messages=messages, # type: ignore
            temperature=0.0,
            response_format={
                "type": "json_object",
                "response_schema": TopLevelStructure.model_json_schema(),
            }
        )

        # Parse JSON response into Pydantic model
        if (
            response
            and isinstance(response, ModelResponse)
            and isinstance(response.choices[0], Choices)
            and response.choices[0].message.content
        ):
            try:
                top_level_structure = TopLevelStructure.model_validate_json(
                    response.choices[0].message.content
                )

                # Parse range strings
                header_ids = parse_range_string(top_level_structure.header)
                main_ids = parse_range_string(top_level_structure.main)
                footer_ids = parse_range_string(top_level_structure.footer)

                # Validate tag coverage
                validate_tag_coverage(
                    header_ids=header_ids,
                    main_ids=main_ids,
                    footer_ids=footer_ids,
                    total_ids=len(content_blocks),
                )

                # Build result list, only including sections that have content
                result = []
                
                if header_ids:
                    header_blocks = filter_blocks_by_numbers(content_blocks, header_ids)
                    result.append((TagName.HEADER, header_blocks))
                
                if main_ids:
                    main_blocks = filter_blocks_by_numbers(content_blocks, main_ids)
                    result.append((TagName.MAIN, main_blocks))
                
                if footer_ids:
                    footer_blocks = filter_blocks_by_numbers(content_blocks, footer_ids)
                    result.append((TagName.FOOTER, footer_blocks))

                return result
            except (ValidationError, ValueError) as e:
                if attempt < max_validation_attempts - 1:
                    logger.warning(f"Validation error (attempt {attempt+1}/{max_validation_attempts}): {e}")
                    # Append error message and retry
                    messages.append(response.choices[0].message.model_dump())
                    messages.append({
                        "role": "user",
                        "content": f"Your previous response had a validation error: {str(e)}. "
                                    "Please correct your response to match the required schema and constraints."
                    })
                    return await detect_top_level_structure(content_blocks, router, messages, max_validation_attempts, attempt + 1)
                else:
                    raise ValueError(f"Validation error on final attempt: {e}")
        else:
            raise ValueError("No valid response from LLM")
            
    except Exception as e:
        if attempt < max_validation_attempts - 1:
            logger.warning(f"Error during structure detection (attempt {attempt+1}/{max_validation_attempts}): {e}")
            return await detect_top_level_structure(content_blocks, router, messages, max_validation_attempts, attempt + 1)
        else:
            logger.error(f"Error during structure detection on final attempt: {e}")
            raise


if __name__ == "__main__":
    import os
    import json
    import dotenv
    import asyncio

    dotenv.load_dotenv()

    gemini_api_key, openai_api_key, deepseek_api_key, openrouter_api_key = "", "", "", os.getenv("OPENROUTER_API_KEY", "")
    router = create_router(gemini_api_key, openai_api_key, deepseek_api_key, openrouter_api_key)
    
    with open(os.path.join("artifacts", "doc_601_content_blocks_with_styles.json"), "r") as fr:
        content_blocks: list[ContentBlock] = json.load(fr)
        content_blocks = [ContentBlock.model_validate(block) for block in content_blocks]

    top_level_structure = asyncio.run(detect_top_level_structure(
        content_blocks, router
    ))
    with open(os.path.join("artifacts", "doc_601_top_level_structure.json"), "w") as fw:
        json.dump(
            [
                (tag_name, [block.model_dump() for block in blocks])
                for tag_name, blocks in top_level_structure
            ],
            fw,
            indent=2
        )