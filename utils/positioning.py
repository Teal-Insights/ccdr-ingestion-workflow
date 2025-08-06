"""Utilities for position-based layout analysis."""

from utils.schema import BoundingBox
from transform.models import BlockType


def is_header_or_footer_by_position(bbox: BoundingBox, page_height: float) -> BlockType | None:
    """
    Determine if a block is a header or footer based on position.
    Excludes top and bottom inch of the page.
    
    Args:
        bbox: Object with x1, y1, x2, y2 coordinates
        page_height: Height of the page in points
    
    Returns:
        True if the block is entirely contained in header/footer area
    """
    # Convert inches to points (1 inch = 72 points)
    inch_in_points = 72
    
    # Header area: top inch (0 to 72 points from top)
    header_bottom = inch_in_points
    
    # Footer area: bottom inch (page_height - 72 to page_height)
    footer_top = page_height - inch_in_points
    
    # Check if block is entirely contained in header area
    if bbox.y2 <= header_bottom:
        return BlockType.PAGE_HEADER
    
    # Check if block is entirely contained in footer area  
    if bbox.y1 >= footer_top:
        return BlockType.PAGE_FOOTER
    
    return None


if __name__ == "__main__":
    block = {
        "left": 68.0,
        "top": 748.0,
        "width": 56.0,
        "height": 16.0,
        "page_number": 1,
        "page_width": 612.0,
        "page_height": 792.0,
        "text": "March 2024",
        "type": "Text",
        "logical_page_number": "null"
    }
    
    bbox = BoundingBox(x1=block['left'], y1=block['top'], x2=block['left'] + block['width'], y2=block['top'] + block['height'])
    page_height = block['page_height']
    print(is_header_or_footer_by_position(bbox, page_height))