# TODO: The models suck at this task, perhaps because so few LayoutBlocks have
# correct page header and page footer classifications. I bet accuracy would improve
# with heuristic-based reclassification for the top and bottom inch of the page.
# Also, we could try mechanically interpolating the series to fill in missing numbers.
# TODO: We should perhaps zero-index the page numbers throughout the codebase

import json
import logging
import asyncio
from collections import defaultdict
from typing import Optional
from litellm import Router
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices
from pydantic import BaseModel, ValidationError
from transform.models import ExtractedLayoutBlock, LayoutBlock, BlockType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def de_fence_json(text: str) -> str:
    """
    Remove markdown code fencing from JSON response.
    Based on detect_structure.py's de_fence function.
    """
    stripped_text = text.strip()
    if stripped_text.startswith(("```json\n", "```\n", "``` json\n")):
        # Find the first newline after the opening fence
        first_newline = stripped_text.find('\n')
        if first_newline != -1:
            # Remove the opening fence
            stripped_text = stripped_text[first_newline + 1:]

        # Remove the closing fence if it exists
        if stripped_text.endswith("\n```"):
            stripped_text = stripped_text[:-4].rstrip()
    
    return stripped_text


class PageMappingResult(BaseModel):
    """Pydantic model for the LLM response containing page mappings."""
    mapping: dict[str, str | None]
    
    @classmethod
    def model_validate_json(cls, json_data: str | bytes, **kwargs) -> "PageMappingResult":
        """
        Custom JSON validation with code fence removal.
        Handles LLM responses that may be wrapped in markdown code blocks.
        """
        if isinstance(json_data, bytes):
            json_data = json_data.decode('utf-8')
        
        # Try direct parsing first (most common case)
        try:
            return super().model_validate_json(json_data, **kwargs)
        except json.JSONDecodeError:
            # Fall back to fence removal if direct parsing fails
            de_fenced_json = de_fence_json(json_data)
            return super().model_validate_json(de_fenced_json, **kwargs)


def create_router(
    gemini_api_key: str, 
    openai_api_key: str, 
    deepseek_api_key: str,
    openrouter_api_key: str,
) -> Router:
    """Create a LiteLLM Router with advanced load balancing and fallback configuration."""
    model_list = [
        {
            "model_name": "page-mapper",
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-chat",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        }
    ]

    # Router configuration
    return Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",  # Weighted random selection
        fallbacks=[{"page-mapper": ["page-mapper"]}],  # Falls back within the same group
        num_retries=3,
        allowed_fails=5,
        cooldown_time=30,
        enable_pre_call_checks=True,  # Enable context window and rate limit checks
        default_max_parallel_requests=50,  # Global default
        set_verbose=False,  # Set to True for debugging
    )


async def _get_logical_page_mapping_from_llm(
    page_contents: dict[int, list[str]],
    router: Router,
    messages: Optional[list[dict[str, str]]] = None,
    max_validation_attempts: int = 3,
    attempt: int = 0,
) -> dict[str, str | None]:
    """
    Calls the LLM with prepared page content to get the physical-to-logical mapping.
    Uses LiteLLM Router with built-in concurrency control and fallbacks.
    """
    SYSTEM_PROMPT = """
You are an expert assistant for analyzing document structures. Your task is to identify the logical page number for each physical page of a document based on its header and footer text.

Rules for your analysis:
1.  **Identify Sequences:** Logical page numbers can be integers (1, 2, 3), lowercase Roman numerals (i, ii, iii), uppercase Roman numerals (I, II, III), or letters (A, B, C).
2.  **Infer Missing Numbers in Sequences:** When pages are part of a clear numbering sequence, infer the logical number for pages where it is not explicitly written. For example:
    - If page 10 shows '10' and page 12 shows '12', infer that page 11 is '11'
    - If page 5 shows 'v' and page 7 shows 'vii', infer that page 6 is 'vi'
    - If the sequence starts at page 3 with '2', infer page 2 is '1'
3.  **Handle Transitions:** A document might switch numbering schemes (e.g., from Roman numerals for a preface to integers for the main body). Correctly identify these transitions and continue inferring within each scheme.
4.  **Extract Cleanly:** The text may contain other information (e.g., "Page 5 of 20", "Section 1 - 5"). Focus on extracting only the page number itself ("5").
5.  **Return null for Truly Unnumbered Pages:** If a page cannot be determined to be part of any numbering sequence (e.g., cover pages, separator pages, or pages that genuinely don't belong to the document's logical flow), return null for that page. Do NOT use the physical page number as a fallback.
6.  **Distinguish Implicit vs. Missing:** 
    - Implicit: A blank page between page 5 and page 7 should be inferred as page 6
    - Missing: A cover page before the numbering starts, or a separator page that breaks the sequence, should be null

You will receive a JSON object mapping physical page numbers to a list of their header/footer text.
You MUST return a JSON object with a single "mapping" field that contains a dictionary mapping every physical page number (as a string key) to either:
- Its corresponding logical page number (as a string value), OR  
- null (for pages that are not part of any logical numbering sequence)

Example Output: {"mapping": {"1": null, "2": "i", "3": "ii", "4": "1", "5": "2", "6": "3", "7": null}}
"""

    # Convert the page_contents dict to a JSON string for the user prompt.
    # Using string keys for JSON compatibility.
    user_prompt_data = {str(p): texts for p, texts in sorted(page_contents.items())}
    user_prompt = json.dumps(user_prompt_data, indent=2)

    messages = messages or [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = await router.acompletion(
            model="page-mapper",
            messages=messages, # type: ignore
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        if (
            response
            and isinstance(response, ModelResponse)
            and isinstance(response.choices[0], Choices)
            and response.choices[0].message.content
        ):
            try:
                # Parse and validate the response (includes automatic JSON extraction if needed)
                page_mapping_result = PageMappingResult.model_validate_json(response.choices[0].message.content)
                return page_mapping_result.mapping
            except ValidationError as e:
                if attempt < max_validation_attempts - 1:
                    logger.warning(f"Validation error (attempt {attempt+1}/{max_validation_attempts}): {e}")
                    # Append error message and retry
                    messages.append(response.choices[0].message.model_dump())
                    messages.append({
                        "role": "user",
                        "content": f"Your previous response had a validation error: {str(e)}. "
                                    "Please correct your response to match the required schema and constraints."
                    })
                    return await _get_logical_page_mapping_from_llm(page_contents, router, messages, max_validation_attempts, attempt + 1)
                else:
                    raise ValueError(f"Validation error on final attempt: {e}")
        else:
            raise ValueError("No valid response from LLM")
            
    except Exception as e:
        if attempt < max_validation_attempts - 1:
            logger.warning(f"Error during page mapping (attempt {attempt+1}/{max_validation_attempts}): {e}")
            return await _get_logical_page_mapping_from_llm(page_contents, router, messages, max_validation_attempts, attempt + 1)
        else:
            raise ValueError(f"Error during page mapping on final attempt: {e}")


async def add_logical_page_numbers(
    extracted_layout_blocks: list[ExtractedLayoutBlock],
    gemini_api_key: str,
    openai_api_key: str,
    deepseek_api_key: str,
    openrouter_api_key: str,
) -> list[LayoutBlock]:
    """
    Some pages will have BlockType.PAGE_HEADER or BlockType.PAGE_FOOTER that contain page numbers.
    These will typically be integers, but may be roman numerals or letters. They will typically
    increment by 1 from the previous page, though it's possible in some cases they may skip a
    page. Most of the time, the page number will be the only content in its block. Some pages
    will not have logical page numbers.

    We need to map page numbers to logical page numbers, and then enrich the extracted layout blocks
    with the logical page numbers to create a list of LayoutBlocks.

    Args:
        extracted_layout_blocks: The list of extracted layout blocks to add logical page numbers to.
        gemini_api_key: API key for Gemini models
        openai_api_key: API key for OpenAI models
        deepseek_api_key: API key for DeepSeek models  
        openrouter_api_key: API key for OpenRouter models
    """
    if not extracted_layout_blocks:
        return []

    # 1. Group header/footer text by physical page number
    page_contents = defaultdict(list)
    for block in extracted_layout_blocks:
        if block.type in {BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER}:
            # Clean up text to improve LLM accuracy
            text = block.text.strip()
            if text:
                page_contents[block.page_number].append(text)

    # If no headers or footers with text were found, there's nothing to analyze.
    # Since we can't determine logical page numbers, set them all to None.
    if not page_contents:
        return [
            LayoutBlock(**block.model_dump(), logical_page_number=None)
            for block in extracted_layout_blocks
        ]

    # 2. Create router and call the LLM to get the page mapping.
    router = create_router(gemini_api_key, openai_api_key, deepseek_api_key, openrouter_api_key)
    page_mapping = {}
    try:
        # The internal function handles retries via the router.
        page_mapping = await _get_logical_page_mapping_from_llm(page_contents, router)
    except Exception as e:
        # If the LLM call fails completely after all retries, log the error.
        # The code will gracefully fall back to None logical page numbers in the next step.
        logger.error(f"LLM call to determine logical page numbers failed: {e}. "
                     "Setting all logical page numbers to None.")
        # page_mapping remains empty.

    # 3. Enrich the original blocks with the logical page numbers.
    enriched_blocks = []
    for block in extracted_layout_blocks:
        # Get the mapped logical number for this page.
        # The LLM can return null for pages that are not part of any logical sequence.
        logical_num = page_mapping.get(str(block.page_number))

        # If the LLM returned null or the page wasn't in the mapping,
        # it means this page is not part of the logical numbering sequence.
        # We pass None as the logical_page_number to indicate this.
        if logical_num is None:
            logical_page_number = None
        else:
            logical_page_number = logical_num

        enriched_blocks.append(
            LayoutBlock(
                **block.model_dump(),
                logical_page_number=logical_page_number
            )
        )

    return enriched_blocks


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    async def main():
        load_dotenv(override=True)
        
        # Get all required API keys
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Ensure at least one API key is available
        if not any([gemini_api_key, openai_api_key, deepseek_api_key, openrouter_api_key]):
            raise ValueError("At least one API key must be provided")
        
        with open("artifacts/wkdir/doc_601.json", "r") as f:
            data = json.load(f)

        blocks = [ExtractedLayoutBlock(**block) for block in data]
        blocks = await add_logical_page_numbers(
            blocks, 
            gemini_api_key or "",
            openai_api_key or "",
            deepseek_api_key or "",
            openrouter_api_key or ""
        )
        with open("artifacts/doc_601_with_logical_page_numbers.json", "w") as f:
            json.dump([block.model_dump() for block in blocks], f, indent=2)

    asyncio.run(main())