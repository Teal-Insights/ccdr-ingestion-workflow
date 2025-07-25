from transform.models import ContentBlock, BlockType


def create_image_block_html(block: ContentBlock, bboxes: bool = False, block_id: int = None) -> str:
    """
    Create an img element for a picture block.
    
    Args:
        block: ContentBlock of type PICTURE
        bboxes: If True, include bbox coordinates as data attributes
        block_id: Optional block ID for the element
        
    Returns:
        HTML img element as string
    """
    img_attrs = []
    
    # Add id if provided
    if block_id is not None:
        img_attrs.append(f'id="block-{block_id}"')
    
    # Add page data attribute
    img_attrs.append(f'data-page="{block.positional_data.page_pdf}"')
    
    # Add src attribute if storage_url is available
    if block.storage_url:
        img_attrs.append(f'src="{block.storage_url}"')
    
    # Add alt text from description or text_content
    alt_text = block.description or block.text_content or "Image"
    img_attrs.append(f'alt="{alt_text}"')
    
    # Add bbox data attribute if requested
    if bboxes and block.positional_data.bbox:
        bbox_values = [
            block.positional_data.bbox["x1"],
            block.positional_data.bbox["y1"], 
            block.positional_data.bbox["x2"],
            block.positional_data.bbox["y2"]
        ]
        bbox_str = ",".join(str(int(coord)) for coord in bbox_values)
        img_attrs.append(f'data-bbox="{bbox_str}"')
    
    attrs_str = " ".join(img_attrs)
    return f"<img {attrs_str} />"


def create_text_block_html(block: ContentBlock, bboxes: bool = False, block_id: int = None) -> str:
    """
    Create a p element for a text block.
    
    Args:
        block: ContentBlock with text content
        bboxes: If True, include bbox coordinates as data attributes
        block_id: Optional block ID for the element
        
    Returns:
        HTML p element as string
    """
    p_attrs = []
    
    # Add id if provided
    if block_id is not None:
        p_attrs.append(f'id="block-{block_id}"')
    
    # Add page data attribute
    p_attrs.append(f'data-page="{block.positional_data.page_pdf}"')
    
    # Add bbox data attribute if requested
    if bboxes and block.positional_data.bbox:
        bbox_values = [
            block.positional_data.bbox["x1"],
            block.positional_data.bbox["y1"],
            block.positional_data.bbox["x2"], 
            block.positional_data.bbox["y2"]
        ]
        bbox_str = ",".join(str(int(coord)) for coord in bbox_values)
        p_attrs.append(f'data-bbox="{bbox_str}"')
    
    attrs_str = " ".join(p_attrs)
    attrs_part = f" {attrs_str}" if attrs_str else ""
    
    # Use text_content or fallback to empty string
    text_content = block.text_content or ""
    
    return f"<p{attrs_part}>{text_content}</p>"


def convert_blocks_to_html(
    content_blocks: list[ContentBlock],
    bboxes: bool = False
) -> str:
    """
    Convert list of ContentBlocks to HTML format. Represent BlockType.PICTURE as
    img, and everything else as p.

    Args:
        content_blocks: list of ContentBlock objects to convert to HTML
        bboxes: If True, include bbox coordinates as data attributes

    Returns:
        The HTML representation as a string
    """
    # Generate HTML
    html_parts = []
    block_counter = 1  # Initialize block counter for fallback IDs

    # Process all blocks
    for block in content_blocks:
        # Create the block element directly (p or img)
        if block.block_type == BlockType.PICTURE:
            block_html = create_image_block_html(block, bboxes, block_counter)
        else:
            block_html = create_text_block_html(block, bboxes, block_counter)
        
        html_parts.append(block_html)
        block_counter += 1

    # Join all parts
    html_content = "\n".join(html_parts)

    return html_content


if __name__ == "__main__":
    import os
    import json
    
    with open(os.path.join("artifacts", "doc_601_content_blocks_with_styles.json"), "r") as fr:
        content_blocks: list[ContentBlock] = json.load(fr)
        content_blocks = [ContentBlock.model_validate(block) for block in content_blocks]

    html_content = convert_blocks_to_html(content_blocks, bboxes=True)
    with open(os.path.join("artifacts", "doc_601_html_content.html"), "w") as fw:
        fw.write(html_content)