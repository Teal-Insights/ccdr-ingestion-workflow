import json
import logging
import asyncio
import re
import roman
import pymupdf
from collections import defaultdict
from typing import Optional
from litellm import Router
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices
from pydantic import BaseModel, ValidationError
from transform.models import ExtractedLayoutBlock, LayoutBlock, BlockType
from utils.json import de_fence
from utils.positioning import is_header_or_footer_by_position

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_actual_page_dimensions(pdf_path: str) -> dict[int, tuple[float, float]]:
    """
    Get actual page dimensions from PDF using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary mapping page numbers (1-indexed) to (width, height) tuples
    """
    try:
        doc = pymupdf.open(pdf_path)
        dimensions = {}
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            rect = page.rect
            # Convert to 1-indexed page numbering to match ExtractedLayoutBlock
            dimensions[page_num + 1] = (rect.width, rect.height)
        
        doc.close()
        return dimensions
    except Exception as e:
        logger.warning(f"Could not get page dimensions from PDF: {e}")
        return {}


def validate_and_correct_page_dimensions(
    blocks: list[ExtractedLayoutBlock], 
    pdf_path: Optional[str] = None
) -> list[ExtractedLayoutBlock]:
    """
    Validate page dimensions in blocks against actual PDF dimensions.
    If PDF path is provided and dimensions differ, use actual dimensions.
    
    Args:
        blocks: List of ExtractedLayoutBlocks to validate
        pdf_path: Optional path to PDF for dimension verification
        
    Returns:
        List of blocks with corrected dimensions if needed
    """
    if not pdf_path:
        return blocks
    
    actual_dimensions = get_actual_page_dimensions(pdf_path)
    if not actual_dimensions:
        return blocks
    
    corrected_blocks = []
    corrections_made = 0
    
    for block in blocks:
        corrected_block = block.model_copy()
        
        if block.page_number in actual_dimensions:
            actual_width, actual_height = actual_dimensions[block.page_number]
            
            # Check if stored dimensions differ significantly from actual
            width_diff = abs(block.page_width - actual_width)
            height_diff = abs(block.page_height - actual_height)
            
            if width_diff > 1.0 or height_diff > 1.0:
                corrected_block.page_width = actual_width
                corrected_block.page_height = actual_height
                corrections_made += 1
        
        corrected_blocks.append(corrected_block)
    
    if corrections_made > 0:
        logger.info(f"Corrected page dimensions for {corrections_made} blocks using PyMuPDF")
    
    return corrected_blocks


class ExtractedLayoutBlockBBox:
    """Adapter to make ExtractedLayoutBlock compatible with positioning utilities."""
    def __init__(self, block: ExtractedLayoutBlock):
        self.x1 = block.left
        self.y1 = block.top
        self.x2 = block.left + block.width
        self.y2 = block.top + block.height


def is_header_or_footer_by_position_block(block: ExtractedLayoutBlock) -> bool:
    """
    Determine if a block is a header or footer based on position.
    Wrapper around the shared positioning utility.
    
    Args:
        block: ExtractedLayoutBlock with position and page dimensions
    
    Returns:
        True if the block is entirely contained in header/footer area
    """
    bbox = ExtractedLayoutBlockBBox(block)
    return is_header_or_footer_by_position(bbox, block.page_height)


def looks_like_page_number(text: str) -> bool:
    """
    Check if text looks like a page number using regex patterns.
    
    Patterns found in doc_601.json:
    - Arabic numerals: 01, 1, 2, 3, etc.
    - Roman numerals: i, ii, iii, I, II, III, etc.
    - Letters: A, B, C
    """
    text = text.strip()
    if not text:
        return False
    
    # Arabic numerals (1-2 digits, possibly with leading zero)
    if re.match(r'^\d{1,2}$', text):
        return True
    
    # Roman numerals (case insensitive)
    if re.match(r'^[IVXLCDMivxlcdm]+$', text):
        return True
    
    # Single letters (for appendices)
    if re.match(r'^[A-Z]$', text):
        return True
    
    return False


def reclassify_headers_and_footers(blocks: list[ExtractedLayoutBlock]) -> list[ExtractedLayoutBlock]:
    """
    Reclassify blocks that should be headers or footers based on position and content.
    """
    reclassified_blocks = []
    
    for block in blocks:
        new_block = block.model_copy()
        
        # Only reclassify Text blocks that are in header/footer positions
        if (block.type == BlockType.TEXT and 
            is_header_or_footer_by_position_block(block) and
            looks_like_page_number(block.text)):
            
            # Determine if it's header or footer based on position
            distance_from_top = block.top
            distance_from_bottom = block.page_height - (block.top + block.height)
            
            if distance_from_top < distance_from_bottom:
                new_block.type = BlockType.PAGE_HEADER
            else:
                new_block.type = BlockType.PAGE_FOOTER
                
            logger.debug(f"Reclassified page {block.page_number} text '{block.text}' as {new_block.type}")
        
        reclassified_blocks.append(new_block)
    
    return reclassified_blocks


def detect_page_number_type(page_number: str) -> str:
    """Detect the type of page number: integer, roman_lower, roman_upper, or letter."""
    page_number = page_number.strip()
    
    if re.match(r'^\d+$', page_number):
        return 'integer'
    elif re.match(r'^[ivxlcdm]+$', page_number):
        return 'roman_lower'
    elif re.match(r'^[IVXLCDM]+$', page_number):
        return 'roman_upper'
    elif re.match(r'^[A-Z]$', page_number):
        return 'letter'
    else:
        return 'unknown'


def increment_page_number(page_number: str, increment: int = 1) -> Optional[str]:
    """Increment a page number by the given amount, handling different formats."""
    page_type = detect_page_number_type(page_number)
    
    try:
        if page_type == 'integer':
            return str(int(page_number) + increment)
        elif page_type == 'roman_lower':
            current_val = roman.fromRoman(page_number.upper())
            new_val = current_val + increment
            if new_val <= 0:
                return None
            return roman.toRoman(new_val).lower()
        elif page_type == 'roman_upper':
            current_val = roman.fromRoman(page_number)
            new_val = current_val + increment
            if new_val <= 0:
                return None
            return roman.toRoman(new_val)
        elif page_type == 'letter':
            current_val = ord(page_number) - ord('A')
            new_val = current_val + increment
            if new_val < 0 or new_val > 25:  # A-Z range
                return None
            return chr(ord('A') + new_val)
    except (ValueError, roman.InvalidRomanNumeralError):
        return None
    
    return None


def interpolate_missing_page_numbers(page_mapping: dict[str, str | None]) -> dict[str, str | None]:
    """
    Interpolate missing page numbers based on detected sequences.
    
    This function identifies sequences of page numbers and fills in gaps where
    the LLM might have missed detecting page numbers.
    """
    # Convert to list of tuples for easier processing
    pages = [(int(k), v) for k, v in page_mapping.items()]
    pages.sort()
    
    interpolated = dict(page_mapping)  # Start with original mapping
    
    # Group pages by numbering sequence type
    sequences: dict[str, list[tuple[int, str]]] = {}  # {sequence_type: [(pdf_page, logical_page), ...]}
    
    for pdf_page, logical_page in pages:
        if logical_page is not None:
            page_type = detect_page_number_type(logical_page)
            if page_type != 'unknown':
                if page_type not in sequences:
                    sequences[page_type] = []
                sequences[page_type].append((pdf_page, logical_page))
    
    # For each sequence type, interpolate missing values
    for seq_type, seq_pages in sequences.items():
        if len(seq_pages) < 2:
            continue  # Need at least 2 points to interpolate
        
        seq_pages.sort()
        
        # Try to fill gaps within the sequence
        for i in range(len(seq_pages) - 1):
            current_pdf, current_logical = seq_pages[i]
            next_pdf, next_logical = seq_pages[i + 1]
            
            # Calculate expected increment
            try:
                if seq_type == 'integer':
                    current_val = int(current_logical)
                    next_val = int(next_logical)
                elif seq_type in ['roman_lower', 'roman_upper']:
                    current_val = roman.fromRoman(current_logical.upper())
                    next_val = roman.fromRoman(next_logical.upper())
                elif seq_type == 'letter':
                    current_val = ord(current_logical) - ord('A')
                    next_val = ord(next_logical) - ord('A')
                else:
                    continue
                
                logical_diff = next_val - current_val
                pdf_diff = next_pdf - current_pdf
                
                # Only interpolate if the logical increment is reasonable (1-3 per PDF page)
                if pdf_diff > 1 and 1 <= logical_diff <= pdf_diff * 3:
                    expected_increment = logical_diff / pdf_diff
                    
                    # Fill in missing pages if increment is close to 1
                    if 0.8 <= expected_increment <= 1.2:
                        for pdf_page in range(current_pdf + 1, next_pdf):
                            if interpolated[str(pdf_page)] is None:
                                logical_increment = round((pdf_page - current_pdf) * expected_increment)
                                new_logical = increment_page_number(current_logical, logical_increment)
                                if new_logical:
                                    interpolated[str(pdf_page)] = new_logical
                                    logger.debug(f"Interpolated page {pdf_page} as {new_logical}")
                
            except (ValueError, roman.InvalidRomanNumeralError):
                continue
    
    return interpolated


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
        
        # Always try fence removal first since LLMs often return fenced JSON
        de_fenced_json = de_fence(json_data)
        
        try:
            # Try to parse the de-fenced JSON manually first
            parsed_data = json.loads(de_fenced_json)
            return cls.model_validate(parsed_data, **kwargs)
        except json.JSONDecodeError:
            # If de-fenced parsing fails, try the original
            try:
                return super().model_validate_json(json_data, **kwargs)
            except json.JSONDecodeError:
                # Re-raise the de-fenced error as it's more likely to be informative
                raise ValueError(f"Failed to parse JSON after de-fencing: {de_fenced_json[:200]}...")


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
    pdf_path: Optional[str] = None,
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
        pdf_path: Optional path to PDF file for dimension validation using PyMuPDF
    """
    if not extracted_layout_blocks:
        return []

    # 0. Validate and correct page dimensions using PyMuPDF if PDF path provided
    if pdf_path:
        extracted_layout_blocks = validate_and_correct_page_dimensions(extracted_layout_blocks, pdf_path)

    # 1. Reclassify blocks that should be headers/footers based on position
    extracted_layout_blocks = reclassify_headers_and_footers(extracted_layout_blocks)

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
        
        # 2.5. Interpolate missing page numbers based on detected sequences
        original_count = sum(1 for v in page_mapping.values() if v is not None)
        page_mapping = interpolate_missing_page_numbers(page_mapping)
        interpolated_count = sum(1 for v in page_mapping.values() if v is not None)
        
        if interpolated_count > original_count:
            logger.info(f"Interpolated {interpolated_count - original_count} additional page numbers "
                       f"({original_count} -> {interpolated_count} total)")
        
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
        
        with open("doc_601.json", "r") as f:
            data = json.load(f)

        blocks = [ExtractedLayoutBlock(**block) for block in data]
        blocks = await add_logical_page_numbers(
            blocks, 
            gemini_api_key or "",
            openai_api_key or "",
            deepseek_api_key or "",
            openrouter_api_key or "",
            pdf_path="doc_601.pdf"  # Pass PDF path for dimension validation
        )
        with open("doc_601_with_logical_page_numbers.json", "w") as f:
            json.dump([block.model_dump() for block in blocks], f, indent=2)

    asyncio.run(main())