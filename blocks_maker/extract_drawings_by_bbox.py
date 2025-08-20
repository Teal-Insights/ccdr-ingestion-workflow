import os
from io import StringIO
from svgelements import SVG, Rect, SVGElement  # type: ignore[import-untyped]
from utils.schema import BoundingBox

def extract_geometries_in_bbox(svg_content: str, bbox: BoundingBox) -> list[SVGElement]:
    """
    Extract all SVG geometries contained within a given bounding box.

    Args:
        svg_content: SVG content as string or file path
        bbox: BoundingBox

    Returns:
        List of non-text geometry elements that are contained within the bbox
    """
    # Parse the SVG
    if isinstance(svg_content, str) and os.path.isfile(svg_content):
        # It's a file path
        svg = SVG.parse(svg_content, reify=True)
    else:
        # It's SVG content as string
        svg = SVG.parse(StringIO(svg_content), reify=True)

    geometries_in_bbox = []

    # Iterate through all elements in the SVG
    element: SVGElement
    for element in svg.elements():
        # Skip containers and non-geometric elements
        if isinstance(element, SVG) or not hasattr(element, 'bbox'):
            continue

        try:
            # Skip if element id matches "font_"
            if element.id and element.id.startswith("font_"):
                continue

            # Get the bounding box of the current element
            element_bbox = element.bbox()
            if element_bbox is None:
                continue

            # Convert to Rect for intersection testing
            elem_rect = Rect(element_bbox[0], element_bbox[1], 
                           element_bbox[2] - element_bbox[0], 
                           element_bbox[3] - element_bbox[1])

            # Check if element is completely contained within the target bbox
            if is_completely_contained(elem_rect, bbox):
                geometries_in_bbox.append(element)
   
        except Exception as e:
            # Some elements might not have proper bbox calculation
            print(f"Warning: Could not process element {type(element)}: {e}")
            continue

    return geometries_in_bbox


def is_completely_contained(elem_rect: Rect, target_bbox: BoundingBox) -> bool:
    """Check if elem_rect is completely contained within target_rect"""
    x1, y1, x2, y2 = int(target_bbox.x1), int(target_bbox.y1), int(target_bbox.x2), int(target_bbox.y2)
    try:
        # SVG rects have x, y, width, height
        return (elem_rect.x >= x1 and 
            elem_rect.y >= y1 and
            elem_rect.x + elem_rect.width <= x2 and
            elem_rect.y + elem_rect.height <= y2)
    except AttributeError:
        # image rects have x0, y0, x1, y1
        return (elem_rect.x0 >= x1 and 
            elem_rect.y0 >= y1 and
            elem_rect.x1 <= x2 and
            elem_rect.y1 <= y2)


def is_bbox_contained(element_bbox: tuple[float, float, float, float], target_bbox: BoundingBox) -> bool:
    """Check if element_bbox tuple is completely contained within target_bbox"""
    x1, y1, x2, y2 = int(target_bbox.x1), int(target_bbox.y1), int(target_bbox.x2), int(target_bbox.y2)
    elem_x1, elem_y1, elem_x2, elem_y2 = element_bbox
    return (elem_x1 >= x1 and elem_y1 >= y1 and elem_x2 <= x2 and elem_y2 <= y2)

# Example usage
if __name__ == "__main__":
    # Example SVG content
    svg_content = """
    <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
        <rect x="10" y="10" width="50" height="30" fill="red"/>
        <circle cx="100" cy="100" r="25" fill="blue"/>
        <path d="M150,50 L180,80 L150,110 Z" fill="green"/>
        <line x1="20" y1="150" x2="80" y2="180" stroke="black"/>
    </svg>
    """

    # Define bounding box (x_min, y_min, x_max, y_max)
    bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)

    # Extract geometries
    geometries = extract_geometries_in_bbox(svg_content, bbox)

    print(f"Found {len(geometries)} geometries in bbox {bbox}:")
    for i, geom in enumerate(geometries):
        print(f"{i+1}. {type(geom).__name__}: {geom}")