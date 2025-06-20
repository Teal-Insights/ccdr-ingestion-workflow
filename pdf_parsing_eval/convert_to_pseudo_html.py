# TODO: For image blocks, use class="image-block" and wrap the description in the block tag
import json
from typing import List, Dict, Any


def format_bbox(bbox: List[float]) -> str:
    """Convert bbox coordinates to comma-separated integer string."""
    return ",".join(str(int(coord)) for coord in bbox)


def create_block_html(block: Dict[str, Any], rich_text: bool, bboxes: bool, block_id: int | None = None) -> str:
    """Create HTML for a single text block."""
    # Choose content based on rich_text flag
    content = block.get("text", "") if rich_text else block.get("plain_text", "")
    
    # Start building the div tag
    div_attrs = ['class="text-block"']
    
    # Add id attribute if provided
    if block_id is not None:
        div_attrs.append(f'id="block-{block_id}"')
    
    # Add bbox data attribute if requested
    if bboxes and "bbox" in block:
        bbox_str = format_bbox(block["bbox"])
        div_attrs.append(f'data-bbox="{bbox_str}"')
    
    # Create the complete div tag
    div_tag = f'<div {" ".join(div_attrs)}>'
    
    return f"{div_tag}{content}</div>"


def convert_blocks_to_pseudo_html(input_path: str, output_path: str, rich_text: bool = False, bboxes: bool = False, include_ids: bool = False) -> str:
    """
    Convert JSON text blocks to HTML format.
    
    Args:
        input_path: Path to input JSON file containing text blocks
        output_path: Path to output HTML file
        rich_text: If True, use "text" field; if False, use "plain_text" field
        bboxes: If True, include bbox coordinates as data attributes
        include_ids: If True, include incrementing id attributes on block divs
    
    Returns:
        The path to the output HTML file
    """
    # Read the JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text_blocks = data.get("text_blocks", [])
    total_pages = data.get("total_pages", 1)
    
    # Group blocks by page
    pages: Dict[int, List[Dict[str, Any]]] = {}
    for block in text_blocks:
        page_num = block.get("page_number", 1)
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(block)
    
    # Generate HTML
    html_parts = []
    block_counter = 1  # Initialize block counter for IDs
    
    # Process pages in order
    for page_num in range(1, total_pages + 1):
        html_parts.append('<div class="page">')
        
        # Add blocks for this page if they exist
        if page_num in pages:
            for block in pages[page_num]:
                block_id = block_counter if include_ids else None
                block_html = create_block_html(block, rich_text, bboxes, block_id)
                html_parts.append(f"  {block_html}")
                if include_ids:
                    block_counter += 1
        
        html_parts.append('</div>')
    
    # Join all parts
    html_content = "\n".join(html_parts)
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


if __name__ == "__main__":
    input_path = "text_blocks.json"
    output_path = "pseudo_html.html"
    convert_blocks_to_pseudo_html(input_path, output_path)