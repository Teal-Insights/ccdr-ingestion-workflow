import json
import os
from typing import List, Dict

from .models import BlocksDocument, Block, TextBlock, ImageBlock, SvgBlock


def format_bbox(bbox: List[float]) -> str:
    """Convert bbox coordinates to comma-separated integer string."""
    return ",".join(str(int(coord)) for coord in bbox)


def create_text_block_html(
    block: TextBlock, rich_text: bool, bboxes: bool, block_id: int | None = None
) -> str:
    """Create HTML for a single text block."""
    # Choose content based on rich_text flag
    content = block.text if rich_text else block.plain_text

    # Start building the div tag
    div_attrs = ['class="text-block"']

    # Add id attribute if provided
    if block_id is not None:
        div_attrs.append(f'id="{block_id}"')

    # Add bbox data attribute if requested
    if bboxes:
        bbox_str = format_bbox(block.bbox)
        div_attrs.append(f'data-bbox="{bbox_str}"')

    # Create the complete div tag
    div_tag = f"<div {' '.join(div_attrs)}>"

    return f"{div_tag}{content}</div>"


def create_image_block_html(
    block: ImageBlock, bboxes: bool, block_id: int | None = None
) -> str:
    """Create HTML for a single image block."""
    # Start building the div tag
    div_attrs = ['class="image-block"']

    # Add id attribute if provided
    if block_id is not None:
        div_attrs.append(f'id="{block_id}"')

    # Add bbox data attribute if requested
    if bboxes:
        bbox_str = format_bbox(block.bbox)
        div_attrs.append(f'data-bbox="{bbox_str}"')

    # Create the complete div tag
    div_tag = f"<div {' '.join(div_attrs)}>"

    return f"{div_tag}{block.description}</div>"


def create_svg_block_html(
    block: SvgBlock, bboxes: bool, block_id: int | None = None
) -> str:
    """Create HTML for a single SVG block."""
    # Start building the div tag
    div_attrs = ['class="svg-block"']

    # Add id attribute if provided
    if block_id is not None:
        div_attrs.append(f'id="{block_id}"')

    # Add bbox data attribute if requested
    if bboxes:
        bbox_str = format_bbox(block.bbox)
        div_attrs.append(f'data-bbox="{bbox_str}"')

    # Create the complete div tag
    div_tag = f"<div {' '.join(div_attrs)}>"

    return f"{div_tag}{block.description}</div>"


def create_block_html(
    block: Block, rich_text: bool, bboxes: bool, block_id: int | None = None
) -> str:
    """Create HTML for any type of block."""
    if isinstance(block, TextBlock):
        return create_text_block_html(block, rich_text, bboxes, block_id)
    elif isinstance(block, ImageBlock):
        return create_image_block_html(block, bboxes, block_id)
    elif isinstance(block, SvgBlock):
        return create_svg_block_html(block, bboxes, block_id)
    else:
        # Fallback for unknown block types
        div_attrs = ['class="unknown-block"']
        if block_id is not None:
            div_attrs.append(f'id="{block_id}"')
        if bboxes:
            bbox_str = format_bbox(block.bbox)
            div_attrs.append(f'data-bbox="{bbox_str}"')
        div_tag = f"<div {' '.join(div_attrs)}>"
        return f"{div_tag}Unknown block type: {block.block_type}</div>"


def convert_blocks_to_html(
    input_path: str,
    output_path: str,
    rich_text: bool = False,
    bboxes: bool = False,
    include_ids: bool = False,
) -> str:
    """
    Convert BlocksDocument to HTML format.

    Args:
        input_path: Path to input JSON file containing BlocksDocument
        output_path: Path to output HTML file
        rich_text: If True, use "text" field for TextBlocks; if False, use "plain_text" field
        bboxes: If True, include bbox coordinates as data attributes
        include_ids: If True, include incrementing id attributes on block divs

    Returns:
        The path to the output HTML file
    """
    # Read and parse the JSON data as BlocksDocument
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    document = BlocksDocument(**data)

    # Group blocks by page
    pages: Dict[int, List[Block]] = {}
    for block in document.blocks:
        page_num = block.page_number
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(block)

    # Generate HTML
    html_parts = []
    block_counter = 1  # Initialize block counter for fallback IDs

    # Process only pages that have blocks, in order
    for page_num in sorted(pages.keys()):
        html_parts.append('<div class="page">')

        # Add blocks for this page
        for block in pages[page_num]:
            if include_ids:
                # Use the block's id field if available, otherwise use sequential counter
                block_id = getattr(block, "id", None) or block_counter
                block_counter += 1
            else:
                block_id = None

            block_html = create_block_html(block, rich_text, bboxes, block_id)
            html_parts.append(f"  {block_html}")

        html_parts.append("</div>")

    # Join all parts
    html_content = "\n".join(html_parts)

    # Write to output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Convert JSON text blocks to HTML format"
    )
    parser.add_argument(
        "input_file", help="Path to input JSON file containing text blocks"
    )
    parser.add_argument("output_file", help="Path to output HTML file")
    parser.add_argument(
        "--rich-text",
        action="store_true",
        help="Use 'text' field instead of 'plain_text' field",
    )
    parser.add_argument(
        "--bboxes",
        action="store_true",
        help="Include bbox coordinates as data attributes",
    )
    parser.add_argument(
        "--include-ids",
        action="store_true",
        help="Include incrementing id attributes on block divs",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)

    try:
        output_path = convert_blocks_to_html(
            input_path=args.input_file,
            output_path=args.output_file,
            rich_text=args.rich_text,
            bboxes=args.bboxes,
            include_ids=args.include_ids,
        )
        print("Successfully converted text blocks to HTML!")
        print(f"Output file: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
