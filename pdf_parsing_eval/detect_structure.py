# We want a detect_structure function that takes the path to an HTML file with numbered divs (corresponding to PDF blocks)
# Take a look at extract_images.py to see how to call Gemini with LiteLLM to submit the HTML text and generate the structured output (we won't need concurrency this time)
# LLM should return JSON object with inclusive block number ranges for header, main, footer, corresponding to front matter, body matter, and back matter of the document
# Example:
# {
#   "header": "1-3,5,7-9",
#   "main": "4,6,10-12",
#   "footer": "13-15"
# }
# Support comma-separated ranges and hyphenated ranges
# Convert number ranges to lists or some other iterable containing all the block numbers (ints) for each section
# Allow any of the sections to be empty (no pages corresponding to that section)
# Enforce that all blocks are included in one of the sections
# Generate a JSON BlocksDocument file (see models.py) for each section and return a list of section_name, file_path pairs
# Planned for future: check token length of HTML corresponding to each section to see if it exceeds LLM output limit, in which case we need to get next level of section structure

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from functools import partial
import asyncio
import pydantic
from tenacity import retry, stop_after_attempt, wait_exponential
from litellm import completion
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices

from models import BlocksDocument, Block


class DocumentStructure(pydantic.BaseModel):
    """Pydantic model for document structure detection response"""
    header: str = pydantic.Field(description="Comma-separated ranges for header blocks (e.g., '1-3,5,7-9')")
    main: str = pydantic.Field(description="Comma-separated ranges for main content blocks (e.g., '4,6,10-12')")
    footer: str = pydantic.Field(description="Comma-separated ranges for footer blocks (e.g., '13-15')")


def parse_range_string(range_str: str) -> List[int]:
    """
    Parse a comma-separated range string into a list of integers.
    
    Args:
        range_str: String like "1-3,5,7-9" or empty string
        
    Returns:
        List of integers representing all block numbers in the ranges
    """
    if not range_str.strip():
        return []
    
    blocks = []
    parts = range_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range like "1-3"
            start, end = part.split('-', 1)
            start_num = int(start.strip())
            end_num = int(end.strip())
            blocks.extend(range(start_num, end_num + 1))
        else:
            # Handle single number like "5"
            blocks.append(int(part))
    
    return sorted(blocks)


def validate_block_coverage(header_blocks: List[int], main_blocks: List[int], 
                          footer_blocks: List[int], total_blocks: int) -> None:
    """
    Validate that all blocks from 1 to total_blocks are covered exactly once.
    
    Args:
        header_blocks: List of block numbers in header section
        main_blocks: List of block numbers in main section  
        footer_blocks: List of block numbers in footer section
        total_blocks: Total number of blocks that should be covered
        
    Raises:
        ValueError: If blocks are missing, duplicated, or out of range
    """
    all_assigned_blocks = set(header_blocks + main_blocks + footer_blocks)
    expected_blocks = set(range(1, total_blocks + 1))
    
    # Check for missing blocks
    missing_blocks = expected_blocks - all_assigned_blocks
    if missing_blocks:
        raise ValueError(f"Missing blocks: {sorted(missing_blocks)}")
    
    # Check for extra blocks
    extra_blocks = all_assigned_blocks - expected_blocks
    if extra_blocks:
        raise ValueError(f"Extra blocks (out of range): {sorted(extra_blocks)}")
    
    # Check for duplicates
    all_blocks_list = header_blocks + main_blocks + footer_blocks
    if len(all_blocks_list) != len(all_assigned_blocks):
        duplicates = []
        seen = set()
        for block in all_blocks_list:
            if block in seen:
                duplicates.append(block)
            seen.add(block)
        raise ValueError(f"Duplicate blocks: {sorted(set(duplicates))}")


def count_blocks_in_html(html_content: str) -> int:
    """
    Count the number of numbered div blocks in the HTML content.
    
    Args:
        html_content: HTML content with numbered divs
        
    Returns:
        Number of blocks found
    """
    import re
    # Look for div elements with id="block_N" or id="N" pattern
    block_pattern = r'<div[^>]*id="(?:block-)?(\d+)"'
    matches = re.findall(block_pattern, html_content)
    if matches:
        return max(int(match) for match in matches)
    return 0


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def detect_structure_with_llm(html_content: str, api_key: str) -> DocumentStructure:
    """
    Use Gemini to analyze HTML content and detect document structure.
    
    Args:
        html_content: HTML content with numbered div blocks
        api_key: Gemini API key
        
    Returns:
        DocumentStructure with header, main, footer block ranges
    """
    # Set environment variable for LiteLLM
    os.environ["GEMINI_API_KEY"] = api_key
    
    prompt = f"""Analyze the following HTML content and identify the document structure. The HTML contains numbered div blocks (like <div id="block_1">, <div id="block_2">, etc.) that correspond to different parts of a PDF document.

Your task is to categorize these blocks into three sections:
- header: Front matter (title pages, table of contents, preface, etc.)
- main: Body content (main chapters, sections, core content)
- footer: Back matter (appendices, bibliography, index, etc.)

Return the block numbers for each section as comma-separated ranges. Use hyphenated ranges for consecutive blocks (e.g., "1-3") and comma-separated individual blocks or ranges (e.g., "1-3,5,7-9").

Rules:
1. Every block must be assigned to exactly one section
2. Any section can be empty (use empty string "")
3. Use the actual content and layout to determine the structure
4. Consider typical document patterns (title pages at start, references at end, etc.)

HTML Content:
{html_content}

Respond with a JSON object containing the three sections."""

    messages = [
        {
            "role": "user", 
            "content": prompt
        }
    ]
    
    # Convert synchronous completion call to async using run_in_executor
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        partial(
            completion,
            model="gemini/gemini-2.5-flash-preview-05-20",
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object", "response_schema": DocumentStructure.model_json_schema()}
        )
    )
    
    # Parse JSON response into Pydantic model
    if response and isinstance(response, ModelResponse) and isinstance(response.choices[0], Choices) and response.choices[0].message.content:
        structure = DocumentStructure.model_validate_json(response.choices[0].message.content)
        return structure
    else:
        raise Exception("No valid response from Gemini")


def filter_blocks_by_numbers(blocks: List[Block], block_numbers: List[int]) -> List[Block]:
    """
    Filter blocks to only include those with block numbers in the specified list.
    
    Args:
        blocks: List of all blocks
        block_numbers: List of block numbers to include
        
    Returns:
        Filtered list of blocks
    """
    # Create a set for faster lookup
    target_numbers = set(block_numbers)
    
    filtered_blocks = []
    for i, block in enumerate(blocks):
        # Block numbers are 1-indexed, list indices are 0-indexed
        block_number = i + 1
        if block_number in target_numbers:
            filtered_blocks.append(block)
    
    return filtered_blocks


async def detect_structure(
    html_file_path: str,
    blocks_json_path: str,
    output_dir: str,
    api_key: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    Detect document structure from HTML file and generate separate BlocksDocument files for each section.
    
    Args:
        html_file_path: Path to HTML file with numbered div blocks
        blocks_json_path: Path to JSON file containing all blocks
        output_dir: Directory to save section-specific JSON files
        api_key: Gemini API key for structure detection
        
    Returns:
        List of (section_name, file_path) pairs for generated files
    """
    # Read HTML content
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Count total blocks in HTML
    total_blocks = count_blocks_in_html(html_content)
    if total_blocks == 0:
        raise ValueError("No numbered blocks found in HTML file")
    
    # Load original blocks document
    with open(blocks_json_path, 'r', encoding='utf-8') as f:
        blocks_data = json.load(f)
    
    blocks_doc = BlocksDocument.model_validate(blocks_data)
    
    # Validate that block count matches
    if len(blocks_doc.blocks) != total_blocks:
        raise ValueError(f"Block count mismatch: HTML has {total_blocks} blocks, JSON has {len(blocks_doc.blocks)} blocks")
    
    if api_key:
        # Use LLM to detect structure
        structure = await detect_structure_with_llm(html_content, api_key)
    else:
        # Fallback: treat everything as main content
        structure = DocumentStructure(
            header="",
            main=f"1-{total_blocks}",
            footer=""
        )
    
    # Parse range strings into lists
    header_blocks = parse_range_string(structure.header)
    main_blocks = parse_range_string(structure.main)
    footer_blocks = parse_range_string(structure.footer)
    
    # Validate that all blocks are covered
    validate_block_coverage(header_blocks, main_blocks, footer_blocks, total_blocks)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate section files
    section_files = []
    
    sections = [
        ("header", header_blocks),
        ("main", main_blocks), 
        ("footer", footer_blocks)
    ]
    
    for section_name, block_numbers in sections:
        if not block_numbers:
            # Skip empty sections
            continue
            
        # Filter blocks for this section
        section_blocks = filter_blocks_by_numbers(blocks_doc.blocks, block_numbers)
        
        # Create new BlocksDocument for this section
        section_doc = BlocksDocument(
            pdf_path=blocks_doc.pdf_path,
            total_pages=blocks_doc.total_pages,
            total_blocks=len(section_blocks),
            blocks=section_blocks
        )
        
        # Save section file
        section_file_path = output_path / f"{section_name}.json"
        with open(section_file_path, 'w', encoding='utf-8') as f:
            f.write(section_doc.model_dump_json(indent=2, exclude_none=True))
        
        section_files.append((section_name, str(section_file_path)))
    
    return section_files


if __name__ == "__main__":
    import dotenv
    import sys
    
    dotenv.load_dotenv(override=True)
    
    if len(sys.argv) < 4:
        print("Usage: uv run detect_structure.py <html_file> <blocks_json> <output_dir>")
        print("Example: uv run detect_structure.py document.html blocks.json output/")
        sys.exit(1)
    
    html_file = sys.argv[1]
    blocks_json = sys.argv[2]
    output_dir = sys.argv[3]
    
    try:
        section_files = asyncio.run(detect_structure(
            html_file_path=html_file,
            blocks_json_path=blocks_json,
            output_dir=output_dir,
            api_key=os.getenv("GEMINI_API_KEY")
        ))
        
        print("Document structure detected successfully!")
        print("Generated section files:")
        for section_name, file_path in section_files:
            print(f"  {section_name}: {file_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
