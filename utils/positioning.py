"""Utilities for position-based layout analysis."""

from typing import Protocol


class BoundingBox(Protocol):
    """Protocol for objects with bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float


def is_header_or_footer_by_position(bbox: BoundingBox, page_height: float) -> bool:
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
        return True
    
    # Check if block is entirely contained in footer area  
    if bbox.y1 >= footer_top:
        return True
    
    return False