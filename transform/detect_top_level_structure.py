from typing import List
import pydantic
from tenacity import retry, stop_after_attempt, wait_exponential
from litellm import completion
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices

from transform.models import ContentBlock
from utils.schema import TagName


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

    ids = []
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
    expected_ids = set(range(0, total_ids))

    assert len(all_assigned_ids) == len(header_ids) + len(main_ids) + len(footer_ids)
    assert all_assigned_ids == expected_ids    


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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def detect_top_level_structure(
    content_blocks: list[ContentBlock], api_key: str
) -> list[tuple[TagName, list[ContentBlock]]]:
    """
    Use Gemini to analyze HTML content and detect document structure.

    Args:
        content_blocks: List of content blocks
        api_key: Gemini API key

    Returns:
        DocumentStructure with header, main, footer block ranges
    """
    prompt = TOP_LEVEL_PROMPT.format(
        html_content="\n".join([block.to_html(bboxes=True, block_id=i) for i, block in enumerate(content_blocks)]),
        header=TagName.HEADER.value,
        main=TagName.MAIN.value,
        footer=TagName.FOOTER.value,
    )
    messages = [{"role": "user", "content": prompt}]

    response: ModelResponse = completion(
        model="gemini/gemini-2.5-flash",
        messages=messages,
        temperature=0.0,
        response_format={
            "type": "json_object",
            "response_schema": TopLevelStructure.model_json_schema(),
        },
        api_key=api_key
    )

    # Parse JSON response into Pydantic model
    if (
        response
        and isinstance(response, ModelResponse)
        and isinstance(response.choices[0], Choices)
        and response.choices[0].message.content
    ):
        top_level_structure = TopLevelStructure.model_validate_json(
            response.choices[0].message.content
        )

    if not top_level_structure:
        raise Exception("No valid response from Gemini")

    header_ids = parse_range_string(top_level_structure.HEADER)
    main_ids = parse_range_string(top_level_structure.MAIN)
    footer_ids = parse_range_string(top_level_structure.FOOTER)

    validate_tag_coverage(
        header_ids=header_ids,
        main_ids=main_ids,
        footer_ids=footer_ids,
        total_ids=len(content_blocks),
    )

    header_blocks = filter_blocks_by_numbers(content_blocks, header_ids)
    main_blocks = filter_blocks_by_numbers(content_blocks, main_ids)
    footer_blocks = filter_blocks_by_numbers(content_blocks, footer_ids)

    return [(TagName.HEADER.value, header_blocks), (TagName.MAIN.value, main_blocks), (TagName.FOOTER.value, footer_blocks)]


if __name__ == "__main__":
    import os
    import json
    import dotenv

    dotenv.load_dotenv()
    
    with open(os.path.join("artifacts", "doc_601_content_blocks_with_styles.json"), "r") as fr:
        content_blocks: list[ContentBlock] = json.load(fr)
        content_blocks = [ContentBlock.model_validate(block) for block in content_blocks]

    top_level_structure = detect_top_level_structure(content_blocks, api_key=os.getenv("GEMINI_API_KEY"))
    with open(os.path.join("artifacts", "doc_601_top_level_structure.json"), "w") as fw:
        json.dump(
            [
                (tag_name, [block.model_dump() for block in blocks])
                for tag_name, blocks in top_level_structure
            ],
            fw,
            indent=2)