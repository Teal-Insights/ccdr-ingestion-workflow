import logging
from typing import List, Optional
import pydantic
from pydantic import ValidationError, field_validator, model_validator
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices
from litellm import Router

from transform.models import ContentBlock
from utils.schema import TagName
from utils.litellm_router import create_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_range_string(range_str: str) -> List[int]:
    """
    Parse a comma-separated range string into a list of integers.
    
    Args:
        range_str: String like "0-3,5,7-9" or empty string
        
    Returns:
        List of integers representing all tag ids in the ranges
        
    Raises:
        ValueError: If the range string format is invalid
    """
    if not range_str.strip():
        return []
    
    ids: list[int] = []
    parts = range_str.split(",")
    
    for part in parts:
        part = part.strip()
        if "-" in part:
            # Handle range like "1-3"
            range_parts = part.split("-", 1)
            if len(range_parts) != 2:
                raise ValueError(f"Invalid range format: {part}")
            try:
                start_num = int(range_parts[0].strip())
                end_num = int(range_parts[1].strip())
                if start_num > end_num:
                    raise ValueError(f"Invalid range: start ({start_num}) > end ({end_num})")
                ids.extend(range(start_num, end_num + 1))
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Non-numeric values in range: {part}")
                raise
        else:
            # Handle single number like "5"
            try:
                ids.append(int(part))
            except ValueError:
                raise ValueError(f"Non-numeric value: {part}")
    
    return sorted(ids)


class TopLevelStructure(pydantic.BaseModel):
    """Pydantic model for document structure detection response"""

    header: List[int] = pydantic.Field(
        description="List of header block IDs (parsed from comma-separated ranges like '0-3,5,7-9')"
    )
    main: List[int] = pydantic.Field(
        description="List of main content block IDs (parsed from comma-separated ranges like '4,6,10-12')"
    )
    footer: List[int] = pydantic.Field(
        description="List of footer block IDs (parsed from comma-separated ranges like '13-15')"
    )
    
    @field_validator('header', 'main', 'footer', mode='before')
    @classmethod
    def parse_range_strings(cls, v) -> List[int]:
        """Parse range strings into lists of integers."""
        if isinstance(v, str):
            return _parse_range_string(v)
        elif isinstance(v, list):
            # Already parsed (e.g., in tests or direct construction)
            return v
        else:
            raise ValueError(f"Expected string or list, got {type(v)}")
    
    @model_validator(mode='after')
    def validate_comprehensive_coverage(self) -> 'TopLevelStructure':
        """Validate that all sections together provide comprehensive coverage without overlaps."""
        # This validation will be completed when we know the total number of content blocks
        # For now, just validate no overlaps within the model itself
        all_ids = self.header + self.main + self.footer
        unique_ids = set(all_ids)
        
        if len(all_ids) != len(unique_ids):
            duplicates = [id for id in unique_ids if all_ids.count(id) > 1]
            raise ValueError(f"Duplicate IDs found across sections: {sorted(duplicates)}")
        
        # Validate that all IDs are non-negative
        negative_ids = [id for id in all_ids if id < 0]
        if negative_ids:
            raise ValueError(f"Negative IDs are not allowed: {sorted(negative_ids)}")
            
        return self
    
    def validate_total_coverage(self, total_ids: int) -> None:
        """Validate that the sections cover all IDs from 0 to total_ids-1 exactly once."""
        all_ids = set(self.header + self.main + self.footer)
        expected_ids = set(range(total_ids))
        
        if all_ids != expected_ids:
            missing_ids = expected_ids - all_ids
            extra_ids = all_ids - expected_ids
            raise ValueError(
                f"Sections must cover all content blocks exactly once. "
                f"Missing IDs: {sorted(missing_ids)}, Extra IDs: {sorted(extra_ids)}"
            )

# Ensure the model fields align with TagName enum values
assert all(
    field in TopLevelStructure.model_fields
    for field in [TagName.HEADER.value, TagName.MAIN.value, TagName.FOOTER.value]
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

                # Validate total coverage (already parsed by field validators)
                top_level_structure.validate_total_coverage(len(content_blocks))

                # Build result list, only including sections that have content
                result = []
                
                if top_level_structure.header:
                    header_blocks = filter_blocks_by_numbers(content_blocks, top_level_structure.header)
                    result.append((TagName.HEADER, header_blocks))
                
                if top_level_structure.main:
                    main_blocks = filter_blocks_by_numbers(content_blocks, top_level_structure.main)
                    result.append((TagName.MAIN, main_blocks))
                
                if top_level_structure.footer:
                    footer_blocks = filter_blocks_by_numbers(content_blocks, top_level_structure.footer)
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