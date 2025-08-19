#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "beautifulsoup4",
# ]
# ///
"""
Validation script for HTML restructuring output.

This script validates that:
1. The output HTML file contains valid HTML
2. All source IDs from input are covered by data-sources attributes in output
3. No extra IDs are present in output that weren't in input

Usage: uv run --script validate_html.py [--block] <input_html_file> <output_html_file>

Options:
    --block    Use blocking error codes (exit 2) for failures instead of non-blocking (exit 3)
"""

import re
import sys
import argparse
import bs4
from pathlib import Path

ALLOWED_TAGS = [
    "header", "main", "footer", "figure", "figcaption",
    "table", "thead", "tbody", "tfoot", "th", "tr", "td", "caption",
    "section", "nav", "aside", "p", "ul", "ol", "li", "h1",
    "h2", "h3", "h4", "h5", "h6", "img", "math", "code",
    "cite", "blockquote", "b", "i", "u", "s", "sup", "sub", "br"
]

# Inline style tags that don't affect leaf node status
INLINE_STYLE_TAGS = {"b", "i", "u", "s", "sup", "sub", "code", "cite", "br"}

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
        
        # Replace <em> with <i> and <strong> with <b> in output
        replacements_made = []
        if "<em>" in output_html or "</em>" in output_html:
            output_html = output_html.replace("<em>", "<i>").replace("</em>", "</i>")
            replacements_made.append("<em> → <i>")
        if "<strong>" in output_html or "</strong>" in output_html:
            output_html = output_html.replace("<strong>", "<b>").replace("</strong>", "</b>")
            replacements_made.append("<strong> → <b>")
        
        # Write back the modified output if replacements were made
        if replacements_made:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_html)
            print(f"📝 Auto-replaced tags: {', '.join(replacements_made)}", file=sys.stderr)
        
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
        
        # Check for disallowed tags in output (only within body)
        disallowed_tags: set[str] = set()
        body = output_soup.find("body")
        if body:
            for element in body.find_all():
                if element.name and element.name not in ALLOWED_TAGS:
                    disallowed_tags.add(element.name)
        else:
            # If no body, check all elements
            for element in output_soup.find_all():
                if element.name and element.name not in ALLOWED_TAGS:
                    disallowed_tags.add(element.name)
        
        # Extract IDs from output data-sources attributes (leaf nodes only)
        ids_in_output: set[int] = set()
        for element in output_soup.find_all():
            if "data-sources" in element.attrs:
                # Check if this is a leaf node (has no child elements with tags, excluding inline style tags)
                has_child_elements = any(
                    child.name and child.name not in INLINE_STYLE_TAGS 
                    for child in element.children 
                    if hasattr(child, 'name')
                )
                if not has_child_elements:
                    try:
                        ids_in_output.update(parse_range_string(element["data-sources"]))
                    except Exception as e:
                        return False, f"Failed to parse data-sources '{element['data-sources']}': {e}"
        
        # Check coverage
        missing_ids = ids_in_input - ids_in_output
        extra_ids = ids_in_output - ids_in_input
        
        if disallowed_tags or missing_ids or extra_ids:
            error_msg = []
            if disallowed_tags:
                error_msg.append(
                    f"You've used some HTML tags in the {output_file} file that are not allowed: {sorted(disallowed_tags)}.\n"
                    f"The allowed tags are: {', '.join(ALLOWED_TAGS)}.\n"
                    "Fix these tags and keep going! You're doing great!"
                )
            if missing_ids:
                missing_msg = f"IDs in the {input_file} file not yet covered by leaf nodes in the {output_file} file: {sorted(missing_ids)}. You'll need to add more leaf nodes (or make sure all existing leaf nodes have data-sources) that cover these ids."
                if len(missing_ids) < 30:
                    missing_msg += (
                        "\n\nNote: if any input nodes are empty or contain garbage characters "
                        "you think shouldn't be in the final output, you may attach their ids "
                        "as data-sources to neighboring output nodes to pass the id validation "
                        "check. Please do this sparingly."
                    )
                error_msg.append(missing_msg)
            if extra_ids:
                # Provide clarifying guidance when output references non-existent IDs
                error_msg.append(
                    "Output references IDs not present in input (invalid data-sources). "
                    f"Extra IDs: {sorted(extra_ids)}. "
                    "Ensure data-sources only contain IDs that exist in the input and split ranges "
                    "to avoid spanning missing IDs."
                )
            return False, "; ".join(error_msg)
        
        return True, "All IDs properly covered"
        
    except Exception as e:
        return False, f"Validation error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Validate HTML restructuring output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script validates that:
1. The output HTML file contains valid HTML
2. All source IDs from input are covered by data-sources attributes in output
3. No extra IDs are present in output that weren't in input
        """
    )
    parser.add_argument("input_file", type=Path, help="Input HTML file path")
    parser.add_argument("output_file", type=Path, help="Output HTML file path")
    parser.add_argument(
        "--block", 
        action="store_true", 
        help="Use blocking error codes (exit 2) for failures instead of non-blocking (exit 3)"
    )

    args = parser.parse_args()

    is_valid, message = validate_html_structure(args.input_file, args.output_file)

    if is_valid:
        # Read the output file to check for placeholders
        with open(args.output_file, 'r', encoding='utf-8') as f:
            output_html = f.read()

        possible_placeholders = []
        for line_num, line in enumerate(output_html.splitlines(), 1):
            if re.search(r'placeholder|<!--|\.\.\.', line):
                possible_placeholders.append(f"Line {line_num}: {line.strip()}")
        if possible_placeholders:
            print(f"🔍 All ids from the {args.input_file} file are present as data-sources in the {args.output_file} file and all tags are valid.\n")
            print("This *might* mean you're done, **BUT**! Please review the following lines to make sure you haven't hacked the validation check by using placeholders or truncated content.")
            print(f"🔍 Possible placeholders found in the {args.output_file} file:\n" + "\n".join(possible_placeholders), file=sys.stderr)

        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_html = f.read()

        # If output file is <85% the length of the input file, print a warning
        if len(output_html) < 0.85 * len(input_html):
            print("🔍 The output file passed a validation check showing all input ids are present as data-sources, but it's suspiciously short compared to the input file, so you've probably cheated by using placeholders or truncated content.")
            print("Please review the output file to make sure it actually contains all the input content.")

        print(
            f"✅ All ids from the {args.input_file} file are present as data-sources in the {args.output_file} file and all tags are valid.\n"
            "This doesn't necessarily mean you're done, but it's a good sign.\n"
            "Once you've checked that the output HTML is well structured with semantic tags and all meaningful content is well-represented "
            f"by leaf nodes in the {args.output_file} file, you can mark your task complete.", file=sys.stderr)
        # Use non-blocking exit code so the success message is visible to Claude
        sys.exit(3)
    else:
        print(f"We're making progress! 😊 {message}", file=sys.stderr)
        # Use blocking or non-blocking exit code based on --block flag
        exit_code = 2 if args.block else 3
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
