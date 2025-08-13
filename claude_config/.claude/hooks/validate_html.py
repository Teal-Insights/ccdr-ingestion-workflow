#!/usr/bin/env python3
"""
Validation script for HTML restructuring output.

This script validates that:
1. The output HTML file contains valid HTML
2. All source IDs from input are covered by data-sources attributes in output
3. No extra IDs are present in output that weren't in input

Usage: uv run validate_html.py <input_html_file> <output_html_file>
"""

import sys
import bs4
from pathlib import Path

def parse_range_string(range_str: str) -> list[int]:
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


def validate_html_structure(input_file: Path, output_file: Path) -> tuple[bool, str]:
    """
    Validate that output HTML properly covers all input IDs.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Read input HTML
        if not input_file.exists():
            return False, f"Input file does not exist: {input_file}"
        
        with open(input_file, "r", encoding="utf-8") as f:
            input_html = f.read()
        
        # Read output HTML
        if not output_file.exists():
            return False, f"Output file does not exist: {output_file}"
        
        with open(output_file, "r", encoding="utf-8") as f:
            output_html = f.read()
        
        # Parse HTML
        try:
            input_soup = bs4.BeautifulSoup(input_html, "html.parser")
            output_soup = bs4.BeautifulSoup(output_html, "html.parser")
        except Exception as e:
            return False, f"Failed to parse HTML: {e}"
        
        # Extract IDs from input
        ids_in_input: set[int] = set()
        for element in input_soup.find_all():
            if "id" in element.attrs:
                try:
                    ids_in_input.add(int(element.attrs["id"]))
                except ValueError:
                    return False, f"Non-numeric ID found in input: {element.attrs['id']}"
        
        # Extract IDs from output data-sources attributes
        ids_in_output: set[int] = set()
        for element in output_soup.find_all():
            if "data-sources" in element.attrs:
                try:
                    ids_in_output.update(parse_range_string(element["data-sources"]))
                except Exception as e:
                    return False, f"Failed to parse data-sources '{element['data-sources']}': {e}"
        
        # Check coverage
        missing_ids = ids_in_input - ids_in_output
        extra_ids = ids_in_output - ids_in_input
        
        if missing_ids or extra_ids:
            error_msg = []
            if missing_ids:
                error_msg.append(f"IDs in input not covered in output: {sorted(missing_ids)}")
            if extra_ids:
                error_msg.append(f"IDs in output not present in input: {sorted(extra_ids)}")
            return False, "; ".join(error_msg)
        
        return True, "All IDs properly covered"
        
    except Exception as e:
        return False, f"Validation error: {e}"


def main():
    if len(sys.argv) != 3:
        print("Usage: uv run validate_html.py <input_html_file> <output_html_file>", file=sys.stderr)
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    is_valid, message = validate_html_structure(input_file, output_file)
    
    if is_valid:
        print("Good news! All ids from the input file are present as data-sources in the output file! This doesn't necessarily mean you're done, but it's a good sign. Once you've checked that the output HTML is well structured with semantic tags and all meaningful content is well-represented by leaf nodes in the output file, you can mark your task complete.")
        sys.exit(0)
    else:
        print(f"We're making progress! 😊 But there's still work to do. {message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

