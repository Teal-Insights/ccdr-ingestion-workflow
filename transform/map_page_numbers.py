import json
import re
from collections import defaultdict
import litellm
from tenacity import retry, stop_after_attempt, wait_fixed
from transform.models import ExtractedLayoutBlock, LayoutBlock, BlockType

def extract_json(text: str) -> dict:
    """Greedily extract exactly one JSON structure."""
    matches = re.findall(r'\{.*?\}|\[.*?\]', text)

    if len(matches) != 1:
        raise ValueError(f"Expected exactly 1 JSON structure, found {len(matches)}")

    match = matches[0].strip()

    return json.loads(match)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def _get_logical_page_mapping_from_llm(
    page_contents: dict[int, list[str]],
    api_key: str,
) -> dict[str, str | None]:
    """
    Calls the LLM with prepared page content to get the physical-to-logical mapping.
    This internal function is decorated with retry logic.
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
You MUST return ONLY a single JSON object that maps every physical page number (as a string key) to either:
- Its corresponding logical page number (as a string value), OR  
- null (for pages that are not part of any logical numbering sequence)

Example Output: {"1": null, "2": "i", "3": "ii", "4": "1", "5": "2", "6": "3", "7": null}
"""

    # Convert the page_contents dict to a JSON string for the user prompt.
    # Using string keys for JSON compatibility.
    user_prompt_data = {str(p): texts for p, texts in sorted(page_contents.items())}
    user_prompt = json.dumps(user_prompt_data, indent=2)

    response = litellm.completion(
        model="deepseek/deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        api_key=api_key,
        response_format={"type": "json_object"}
    )

    return extract_json(response.choices[0].message.content)


def add_logical_page_numbers(
    extracted_layout_blocks: list[ExtractedLayoutBlock],
    api_key: str,
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

    # 2. Call the LLM to get the page mapping.
    page_mapping = {}
    try:
        # The internal function handles retries.
        page_mapping = _get_logical_page_mapping_from_llm(page_contents, api_key)
    except Exception as e:
        # If the LLM call fails completely after all retries, log the error.
        # The code will gracefully fall back to None logical page numbers in the next step.
        print(f"LLM call to determine logical page numbers failed: {e}. "
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

    load_dotenv(override=True)
    
    with open("artifacts/wkdir/doc_601.json", "r") as f:
        data = json.load(f)

    blocks = [ExtractedLayoutBlock(**block) for block in data]
    blocks = add_logical_page_numbers(blocks, api_key=os.getenv("DEEPSEEK_API_KEY"))
    with open("artifacts/doc_601_with_logical_page_numbers.json", "w") as f:
        json.dump([block.model_dump() for block in blocks], f, indent=2)