import os
from io import StringIO
from svgelements import SVG, Path, Rect

def extract_geometries_in_bbox(svg_content, bbox):
    """
    Extract all SVG geometries that intersect with a given bounding box.

    Args:
        svg_content: SVG content as string or file path
        bbox: Tuple (x_min, y_min, x_max, y_max) defining the bounding box

    Returns:
        List of geometry elements that intersect with the bbox
    """
    # Parse the SVG
    if isinstance(svg_content, str) and os.path.isfile(svg_content):
        # It's a file path
        svg = SVG.parse(svg_content, reify=True)
    else:
        # It's SVG content as string
        svg = SVG.parse(StringIO(svg_content), reify=True)

    x_min, y_min, x_max, y_max = bbox
    target_bbox = Rect(x_min, y_min, x_max - x_min, y_max - y_min)

    geometries_in_bbox = []

    # Iterate through all elements in the SVG
    for element in svg.elements():
        # Skip containers and non-geometric elements
        if isinstance(element, SVG) or not hasattr(element, 'bbox'):
            continue

        try:
            # Get the bounding box of the current element
            element_bbox = element.bbox()
            if element_bbox is None:
                continue

            # Convert to Rect for intersection testing
            elem_rect = Rect(element_bbox[0], element_bbox[1], 
                           element_bbox[2] - element_bbox[0], 
                           element_bbox[3] - element_bbox[1])

            # Check if element is completely contained within the target bbox
            if is_completely_contained(elem_rect, target_bbox):
                geometries_in_bbox.append(element)
   
        except Exception as e:
            # Some elements might not have proper bbox calculation
            print(f"Warning: Could not process element {type(element)}: {e}")
            continue

    return geometries_in_bbox

def is_completely_contained(elem_rect, target_rect):
    """Check if elem_rect is completely contained within target_rect"""
    return (elem_rect.x >= target_rect.x and 
            elem_rect.y >= target_rect.y and
            elem_rect.x + elem_rect.width <= target_rect.x + target_rect.width and
            elem_rect.y + elem_rect.height <= target_rect.y + target_rect.height)

def rectangles_intersect(rect1, rect2):
    """Check if two rectangles intersect"""
    return not (rect1.x + rect1.width < rect2.x or 
                rect2.x + rect2.width < rect1.x or
                rect1.y + rect1.height < rect2.y or 
                rect2.y + rect2.height < rect1.y)

def extract_geometries_precise_intersection(svg_content, bbox):
    """
    More precise extraction that checks actual geometry intersection,
    not just bounding box intersection.
    """
    import os
    if isinstance(svg_content, str) and os.path.isfile(svg_content):
        # It's a file path
        svg = SVG.parse(svg_content, reify=True)
    else:
        # It's SVG content as string
        from io import StringIO
        svg = SVG.parse(StringIO(svg_content), reify=True)

    x_min, y_min, x_max, y_max = bbox
    bbox_path = Path(f"M{x_min},{y_min} L{x_max},{y_min} L{x_max},{y_max} L{x_min},{y_max} Z")

    intersecting_geometries = []

    for element in svg.elements():
        if not hasattr(element, 'as_path'):
            continue

        try:
            # Convert element to path for intersection testing
            element_path = element.as_path()
            if element_path is None:
                continue

            # Check for intersection (this is more computationally expensive)
            if paths_intersect(element_path, bbox_path):
                intersecting_geometries.append(element)

        except Exception as e:
            print(f"Warning: Could not process element {type(element)}: {e}")
            continue

    return intersecting_geometries

def paths_intersect(path1, path2):
    """
    Simple intersection test - you might want to use a more sophisticated
    method depending on your needs
    """
    try:
        # This is a simplified check - for production use, consider
        # using more robust computational geometry libraries
        bbox1 = path1.bbox()
        bbox2 = path2.bbox()

        if bbox1 is None or bbox2 is None:
            return False

        return not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
                   bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
    except Exception:
        return False

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
    bbox = (0, 0, 100, 100)

    # Extract geometries
    geometries = extract_geometries_in_bbox(svg_content, bbox)

    print(f"Found {len(geometries)} geometries in bbox {bbox}:")
    for i, geom in enumerate(geometries):
        print(f"{i+1}. {type(geom).__name__}: {geom}")