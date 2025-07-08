import json
import os
from typing import List, Dict, Any
from pathlib import Path
from .models import BlocksDocument


def _get_block_sort_key(block: Dict[str, Any]) -> tuple:
    """
    Generate a sort key for a block based on page number and position.
    Returns (page_number, y_top, x_left) for top-left to bottom-right reading order.
    """
    page_number = block.get("page_number", 0)
    bbox = block.get("bbox", [0, 0, 0, 0])

    # Use top-left corner for sorting (x0, y0)
    x_left = bbox[0] if len(bbox) >= 2 else 0
    y_top = bbox[1] if len(bbox) >= 2 else 0

    return (page_number, y_top, x_left)


def combine_blocks(json_file_paths: List[str], output_file_path: str) -> str:
    """
    Combine multiple block JSON files into a single sorted output file.

    Args:
        json_file_paths: List of paths to JSON files containing blocks
        output_file_path: Full path where the combined output file should be written

    Returns:
        Path to the combined JSON file

    Raises:
        FileNotFoundError: If any input file doesn't exist
        ValueError: If JSON files have incompatible schemas
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    combined_blocks = []
    pdf_path = None
    total_pages = 0

    print(f"Combining {len(json_file_paths)} block files...")

    for file_path in json_file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Block file not found: {file_path}")

        print(f"  Loading blocks from: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate and extract blocks
            if isinstance(data, dict):
                # New format with BlocksDocument schema
                if "blocks" in data:
                    blocks = data["blocks"]
                    current_pdf_path = data.get("pdf_path")
                    current_total_pages = data.get("total_pages", 0)
                # Legacy format with text_blocks
                elif "text_blocks" in data:
                    blocks = data["text_blocks"]
                    current_pdf_path = data.get("pdf_path")
                    current_total_pages = data.get("total_pages", 0)
                else:
                    raise ValueError(f"Unrecognized JSON structure in {file_path}")
            elif isinstance(data, list):
                # Direct list of blocks (legacy format)
                blocks = data
                current_pdf_path = None
                current_total_pages = 0
            else:
                raise ValueError(f"Invalid JSON format in {file_path}")

            # Set or validate PDF path consistency
            if pdf_path is None:
                pdf_path = current_pdf_path
            elif current_pdf_path and pdf_path != current_pdf_path:
                print(
                    f"  Warning: PDF path mismatch. Expected {pdf_path}, got {current_pdf_path}"
                )

            # Update total pages (use maximum)
            total_pages = max(total_pages, current_total_pages)

            # Add blocks to combined list
            combined_blocks.extend(blocks)
            print(f"    Added {len(blocks)} blocks")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error processing file {file_path}: {e}")

    # Sort all blocks by page number and position
    print(f"Sorting {len(combined_blocks)} combined blocks...")
    combined_blocks.sort(key=_get_block_sort_key)

    # Assign sequential IDs to blocks after sorting
    print(f"Assigning sequential IDs to {len(combined_blocks)} blocks...")
    for i, block in enumerate(combined_blocks, start=1):
        block["id"] = i

    # Create output document
    output_data = {
        "pdf_path": pdf_path or "unknown",
        "total_pages": total_pages,
        "total_blocks": len(combined_blocks),
        "blocks": combined_blocks,
    }

    # Validate with Pydantic model if possible
    try:
        validated_doc = BlocksDocument(**output_data)
        output_data = validated_doc.model_dump()
        print("✓ Output validated with Pydantic model")
    except Exception as e:
        print(f"Warning: Could not validate with Pydantic model: {e}")
        print("Proceeding with raw data...")

    # Write combined output
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Combined {len(combined_blocks)} blocks from {len(json_file_paths)} files")
    print(f"✓ Output saved to: {output_file_path}")

    return output_file_path


def main():
    """Command-line interface for combining block files"""
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python combine_blocks.py <output_file_path> <file1.json> <file2.json> [file3.json] ..."
        )
        print(
            "Example: python combine_blocks.py /tmp/combined_blocks.json text_blocks.json images.json svgs.json"
        )
        sys.exit(1)

    output_file_path = sys.argv[1]
    input_files = sys.argv[2:]

    try:
        output_path = combine_blocks(input_files, output_file_path)
        print(f"\nSuccess! Combined blocks saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
