# Use remove element by id to remove unused clip paths for better reliability than regex

import tempfile
import subprocess
import os
from PIL import Image
from typing import Literal
import re
import io
import svgelements
import numpy as np

def extract_elements(
        svg_text: str, elements_to_extract: list[Literal["g", "mask", "clipPath", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon"]] = ["g"], filter_by: list[str] = [], from_defs: bool = False
    ) -> list[str]:
    """
    Extract top-level <g> groups from SVG content, either from within <defs> or the main content.
    This function uses a stack-based approach to correctly handle nested groups.
    Args:
        svg_text: The SVG content as a string
        elements_to_extract: List of top-level element types to extract
        filter_by: List of substrings; only groups containing any of these will be returned
            (intended for filtering defs by "mask_" or "clip_" identifiers)
        from_defs: If True, extract groups from within <defs>; otherwise from main content
    Returns:
        List of top-level <g> group strings
    """
    
    # Handle switching between defs and main content
    start: int
    end: int

    if from_defs:
        start = svg_text.find('>', svg_text.find('<defs')) + 1
        # In case there's more than one defs, find the last one
        end = svg_text.rfind('</defs>')
        if start < 6 or end == -1:
            return [] # No defs found
    else:
        defs_end: int
        defs_end = svg_text.rfind('</defs>')
        if defs_end == -1:
            start = svg_text.find('>', svg_text.find('<svg')) + 1
        else:
            start = defs_end + 6
        end = svg_text.find('</svg>')

    # Extract top-level elements between the determined start and end
    content: str = svg_text[start:end]
    elements: list[str] = []
    stack: list[tuple[int, str]] = []
    i: int = 0
    element_start: int | None = None
    popped_element: tuple[int, str] | None = None

    while i < len(content):
        if content[i] == '<':
            # Check if this is a closing tag
            if content[i+1] == '/':
                # Find the tag name
                end_tag_start = i + 2
                end_tag_end = content.find('>', end_tag_start)
                if end_tag_end != -1:
                    tag_name = content[end_tag_start:end_tag_end]
                    
                    # Check if this closing tag matches one of our elements_to_extract
                    if tag_name in elements_to_extract and stack:
                        popped_element = stack.pop()
                        if not stack and popped_element[1] == tag_name:  # Stack empty = complete top-level element
                            group_end = end_tag_end + 1
                            if element_start is not None:
                                if filter_by:
                                    if any(f in content[element_start:group_end] for f in filter_by):
                                        elements.append(content[element_start:group_end])
                                else:
                                    # No filter, add all elements
                                    elements.append(content[element_start:group_end])
                                element_start = None
                    # Pop from stack for any closing tag (even if not in elements_to_extract)
                    elif stack:
                        stack.pop()
                    i = end_tag_end + 1
                else:
                    i += 1
            else:
                # This is an opening tag - find the tag name
                tag_start = i + 1
                tag_end = tag_start
                while tag_end < len(content) and content[tag_end] not in ' >\t\n':
                    tag_end += 1
                
                if tag_end < len(content):
                    tag_name = content[tag_start:tag_end]
                    
                    # Check if it's a self-closing tag
                    tag_close = content.find('>', tag_end)
                    if tag_close != -1 and content[tag_close-1] == '/':
                        # Self-closing tag - if it's a top-level element we want, extract it
                        if tag_name in elements_to_extract and not stack:
                            if filter_by:
                                if any(f in content[i:tag_close+1] for f in filter_by):
                                    elements.append(content[i:tag_close+1])
                            else:
                                elements.append(content[i:tag_close+1])
                        i = tag_close + 1
                    else:
                        # Opening tag - push ALL tags onto stack
                        if tag_name in elements_to_extract and not stack:
                            element_start = i  # Only set start for top-level elements we want
                        stack.append((i, tag_name))
                        i = tag_close + 1 if tag_close != -1 else i + 1
                else:
                    i += 1
        else:
            i += 1

    return elements


def filter_svg_content(
        svg_content: str, filter_text: bool = True, filter_images: bool = True
    ) -> str:
    """
    Filter SVG content to capture only vector drawings that are not text or background colors/gradients.
    
    Args:
        svg_content: The SVG content to filter
        filter_text: Whether to remove text-related elements
        filter_images: Whether to remove image elements
    """
    if filter_text:
        # Remove text-related elements
        # Handle both self-closing and paired font path elements
        svg_content = re.sub(r'<path[^>]*id="font_[^"]*"[^>]*/?>', "", svg_content)
        svg_content = re.sub(r"<use[^>]*data-text[^>]*>", "", svg_content)
        svg_content = re.sub(r"<text[^>]*>.*?</text>", "", svg_content, flags=re.DOTALL)

    if filter_images:
        # Remove image elements
        svg_content = re.sub(r"<image[^>]*>.*?</image>", "", svg_content, flags=re.DOTALL)
        svg_content = re.sub(r"<image[^>]*/>", "", svg_content)

    # Remove empty group tags
    # This needs to be done iteratively as removing inner groups may make outer groups empty
    prev_content = ""
    iteration = 0
    while prev_content != svg_content:
        prev_content = svg_content
        iteration += 1

        # Remove empty groups (both self-closing and paired)
        svg_content = re.sub(r"<g[^>]*>\s*</g>", "", svg_content)
        svg_content = re.sub(r"<g[^>]*/>", "", svg_content)

        if iteration > 10:  # Safety break
            raise Exception("Likely infinite loop in filter_svg_content")

    # Remove unused clipPath definitions (after text/images/groups are removed)
    svg_content = remove_unused_clippaths(svg_content)

    # Clean up empty lines and excessive whitespace
    svg_content = re.sub(r"\n\s*\n", "\n", svg_content)
    svg_content = re.sub(r"^\s*$", "", svg_content, flags=re.MULTILINE)

    return svg_content


def render_svg_to_pixels(svg_content: str, width: int = 200, height: int = 200) -> np.ndarray:
    """
    Render SVG content to a PIL Image using available SVG renderers.
    
    Args:
        svg_content: The SVG content as a string
        width: Width for rendering
        height: Height for rendering
        
    Returns:
        Numpy array of pixel data (RGB) or None if rendering failed
    """
    # Create a temporary SVG file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as temp_svg:
        temp_svg.write(svg_content)
        temp_svg_path = temp_svg.name
    
    # Use Inkscape
    temp_png_path = temp_svg_path.replace('.svg', '.png')
        
    result = subprocess.run([
        'inkscape', 
        '--export-type=png',
        '--export-filename=' + temp_png_path,
        f'--export-width={width}',
        f'--export-height={height}',
        '--export-background=white',  # Force white background
        '--export-background-opacity=1.0',  # Make background opaque
        temp_svg_path
    ], capture_output=True, text=True, timeout=10)
    
    pixels: np.ndarray | None = None
    if result.returncode == 0 and os.path.exists(temp_png_path):
        img = Image.open(temp_png_path)
        # Convert RGBA to RGB to ensure no alpha channel
        if img.mode in ('RGBA', 'LA'):
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            else:  # LA mode
                background.paste(img.convert('RGB'), mask=img.split()[-1])
            img_copy = background
        else:
            img_copy = img.convert('RGB')  # Ensure RGB mode
        if img_copy is None:
            raise Exception("Failed to render SVG - SVG content may be malformed")
    
        pixels = np.array(img_copy)
        img.close()
    else:
        raise Exception(f"Inkscape rendering failed: {result.stderr}")
    
    os.unlink(temp_png_path)
    os.unlink(temp_svg_path)

    if pixels is None:
        raise Exception("Failed to render SVG - SVG content may be malformed")

    return pixels


def extract_viewbox_values(svg_content: str) -> tuple[float, float, float, float] | None:
    """
    Extract viewBox values from SVG content.
    
    Returns:
        Tuple of (x, y, width, height) or None if not found
    """
    viewbox_match = re.search(r'viewBox="([^"]+)"', svg_content)
    if viewbox_match:
        try:
            values = [float(x) for x in viewbox_match.group(1).split()]
            if len(values) == 4:
                return tuple(values)  # type: ignore
        except ValueError:
            pass
    return None


def update_svg_viewbox_and_dimensions(svg_content: str, x: float, y: float, width: float, height: float) -> str:
    """
    Update SVG viewBox and dimensions in a single operation to avoid conflicts.
    
    Args:
        svg_content: The SVG content to update
        x, y, width, height: New viewBox values
        
    Returns:
        Updated SVG content with new viewBox and dimensions
    """
    new_viewbox = f'viewBox="{x} {y} {width} {height}"'
    new_width = f'width="{width:.2f}"'
    new_height = f'height="{height:.2f}"'
    
    # First, remove all existing viewBox, width, and height attributes
    svg_content = re.sub(r'\s+viewBox="[^"]*"', '', svg_content)
    svg_content = re.sub(r'\s+width="[^"]*"', '', svg_content)
    svg_content = re.sub(r'\s+height="[^"]*"', '', svg_content)
    
    # Then add the new attributes after the opening <svg tag
    svg_content = re.sub(
        r'(<svg[^>]*?)>', 
        rf'\1 {new_viewbox} {new_width} {new_height}>', 
        svg_content
    )
    
    return svg_content


def get_group_bounding_box(svg_content: str, group_id: str) -> tuple[float, float, float, float] | None:
    """
    Get the bounding box of a specific group by its ID.
    
    Args:
        svg_content: The SVG content as a string
        group_id: The ID of the group to get the bounding box for
        
    Returns:
        Bounding box as (x0, y0, x1, y1)
    """
    # Parse the SVG from string content using io.StringIO to simulate a file
    svg_file = io.StringIO(svg_content)
    svg = svgelements.SVG.parse(svg_file, reify=True)

    # Find the element with the specified ID
    target_element = None
    for element in svg.elements():
        if hasattr(element, 'values') and element.values.get('id') == group_id:
            target_element = element
            break
    
    if target_element is None:
        raise ValueError(f"Group with ID '{group_id}' not found in SVG")

    # Get the bounding box of the target element
    if hasattr(target_element, 'bbox') and callable(target_element.bbox):
        element_bbox = target_element.bbox()
        if element_bbox is not None:
            # Try to convert to list and validate
            bbox_list = list(element_bbox)  # type: ignore
            if len(bbox_list) == 4:
                x0, y0, x1, y1 = bbox_list
                # Ensure all values are numeric
                x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
                
                # Check if bbox is valid
                if x1 > x0 and y1 > y0:
                    return (x0, y0, x1, y1)
    
    print(f"Warning: Group with ID '{group_id}' has no valid bounding box: {svg_content}")
    print("This may happen if the group contains only a vertical or horizontal line.")
    return None


def get_svg_bounding_box(svg_content: str) -> tuple[float, float, float, float]:
    """
    Get the bounding box of all visible elements in an SVG.
    
    Args:
        svg_content: The SVG content as a string
        
    Returns:
        Bounding box as (x0, y0, x1, y1)
    """
    # Parse the SVG from string content using io.StringIO to simulate a file
    svg_file = io.StringIO(svg_content)
    svg = svgelements.SVG.parse(svg_file, reify=True)

    # Calculate the bounding box of all visible elements
    bbox = None
    for element in svg.elements():
        # Skip elements that are hidden or have no geometry
        try:
            if hasattr(element, 'values') and element.values.get('visibility') == 'hidden':
                continue
            if hasattr(element, 'values') and element.values.get('display') == 'none':
                continue
        except (KeyError, AttributeError):
            pass
        
        # Get element bbox if it has geometry
        if hasattr(element, 'bbox') and callable(element.bbox):
            element_bbox = element.bbox()
            # Check if bbox is valid - must be a sequence with 4 numeric values
            if element_bbox is not None:
                # Try to convert to list and validate
                bbox_list = list(element_bbox)  # type: ignore
                if len(bbox_list) == 4:
                    x0, y0, x1, y1 = bbox_list
                    # Ensure all values are numeric
                    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
                    
                    # Skip invalid or zero-size bboxes
                    if x1 > x0 and y1 > y0:
                        if bbox is None:
                            bbox = [x0, y0, x1, y1]
                        else:
                            bbox[0] = min(bbox[0], x0)  # min x
                            bbox[1] = min(bbox[1], y0)  # min y
                            bbox[2] = max(bbox[2], x1)  # max x
                            bbox[3] = max(bbox[3], y1)  # max y
    
    if bbox is None:
        raise ValueError("No visible elements with geometry found in SVG to determine content bounds")

    return tuple(bbox)  # type: ignore


def clip_svg_to_bounding_box(svg_content: str, bbox: tuple[float, float, float, float], padding: float = 2.0) -> str:
    """
    Clip SVG content to a specified bounding box.
    
    Args:
        svg_content: The original SVG content as a string
        bbox: Bounding box as (x0, y0, x1, y1)
        padding: Padding to add around the bounding box
        
    Returns:
        Clipped SVG content
    """
    x0, y0, x1, y1 = bbox
    
    # Add padding around the content
    x0 -= padding
    y0 -= padding
    x1 += padding
    y1 += padding
    
    # Calculate new dimensions
    width = x1 - x0
    height = y1 - y0
    
    print(f"  Content bounds: ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")
    print(f"  New dimensions: {width:.1f} x {height:.1f}")
    
    # Update viewBox and dimensions in a single operation
    return update_svg_viewbox_and_dimensions(svg_content, x0, y0, width, height)


def clip_svg_to_content_bounds(svg_content: str) -> tuple[str, tuple[float, float, float, float]]:
    """
    Clip SVG content to its actual content bounds, removing empty space around the drawing.
    
    Args:
        svg_content: The original SVG content as a string
        
    Returns:
        Tuple of (clipped SVG content, bounding box as (x0, y0, x1, y1))
    """
    # Get the bounding box of all visible elements
    bbox = get_svg_bounding_box(svg_content)
    
    # Clip the SVG to the bounding box
    clipped_svg = clip_svg_to_bounding_box(svg_content, bbox)
    
    # Calculate the final bounding box with padding for return value
    x0, y0, x1, y1 = bbox
    padding = 2.0
    final_bbox = (x0 - padding, y0 - padding, x1 + padding, y1 + padding)
    
    return clipped_svg, final_bbox


def remove_unused_clippaths(svg_content: str) -> str:
    """Remove clipPath definitions that are not referenced elsewhere in the document"""
    # Find all clipPath definitions and their IDs
    clippath_pattern = r'<clipPath[^>]*id="([^"]+)"[^>]*>.*?</clipPath>'
    clippath_matches = re.findall(clippath_pattern, svg_content, flags=re.DOTALL)

    if not clippath_matches:
        return svg_content

    removed_count = 0
    # Remove unused clipPath definitions
    for clippath_id in clippath_matches:
        # Count how many times this ID appears in the document
        # If it only appears once, it's just the definition and not used anywhere
        id_count = svg_content.count(clippath_id)

        if id_count == 1:
            # Remove this specific clipPath definition
            unused_clippath_pattern = (
                rf'<clipPath[^>]*id="{re.escape(clippath_id)}"[^>]*>.*?</clipPath>'
            )
            before_length = len(svg_content)
            svg_content = re.sub(
                unused_clippath_pattern, "", svg_content, flags=re.DOTALL
            )
            after_length = len(svg_content)
            if before_length != after_length:
                removed_count += 1

    return svg_content


def remove_element_by_id(svg_content: str, element_id: str) -> str:
    """
    Remove an SVG element by its ID using a stack-based approach to handle nested elements correctly.
    
    Args:
        svg_content: The SVG content as a string
        element_id: The ID of the element to remove
        
    Returns:
        SVG content with the element removed
    """
    i = 0
    stack = []
    element_start = None
    element_tag_name = None
    
    while i < len(svg_content):
        if svg_content[i] == '<':
            # Find the end of the tag
            tag_end = svg_content.find('>', i)
            if tag_end == -1:
                i += 1
                continue
                
            tag_content = svg_content[i:tag_end + 1]
            
            # Check if this is a self-closing tag
            if tag_content.endswith('/>'):
                # Handle self-closing tags
                if f'id="{element_id}"' in tag_content:
                    # Found target self-closing element - remove it
                    print(f"    Removed self-closing element {element_id}")
                    return svg_content[:i] + svg_content[tag_end + 1:]
                i = tag_end + 1
                continue
            
            # Check if this is a closing tag
            if tag_content.startswith('</'):
                tag_name = tag_content[2:-1].strip()
                if stack and stack[-1][1] == tag_name:
                    # Pop from stack
                    start_pos, popped_tag = stack.pop()
                    
                    # Check if this was our target element
                    if start_pos == element_start and popped_tag == element_tag_name:
                        # We've found the end of our target element - remove the entire element
                        print(f"    Removed paired element {element_id} (tag: {element_tag_name})")
                        return svg_content[:element_start] + svg_content[tag_end + 1:]
                
                i = tag_end + 1
                continue
            
            # This is an opening tag
            # Extract tag name
            tag_parts = tag_content[1:-1].split()
            if tag_parts:
                tag_name = tag_parts[0]
                
                # Check if this tag has our target ID
                if f'id="{element_id}"' in tag_content:
                    # Found our target element
                    element_start = i
                    element_tag_name = tag_name
                    stack.append((element_start, tag_name))
                    i = tag_end + 1
                    continue
                
                # Push to stack for all opening tags
                stack.append((i, tag_name))
                i = tag_end + 1
                continue
        
        i += 1
    
    # Element not found
    raise ValueError(f"Element {element_id} not found")


def extract_svg_header(svg_content: str, page_num: int) -> str:
    """Extract the SVG header from the SVG content"""
    svg_header_match = re.search(r'^(<svg[^>]*>)', svg_content)
    if not svg_header_match:
        raise ValueError(f"Failed to extract SVG header on page {page_num + 1}")
    svg_header: str = svg_header_match.group(1)
    if not svg_header.endswith("\n"):
        svg_header += "\n"
    return svg_header


def assign_ids_to_elements(svg_content: str) -> str:
    """
    Find all opening tags that don't have an ID and assign them a unique ID.
    
    Args:
        svg_content: The SVG content to modify
        
    Returns:
        Modified SVG content with IDs assigned
    """
    working_svg_content = svg_content
    id_counter = 0
    
    # Pattern to match opening tags (including self-closing) that don't have an id attribute
    # Exclude the svg tag itself
    pattern = r'<(?!svg\s|/)([\w-]+)(?![^>]*\sid\s*=)[^>]*?(?:/>|>)'
    
    # Find all matches with their positions (from end to start to avoid position shifts)
    matches = list(re.finditer(pattern, working_svg_content))
    
    # Process from end to start to avoid position shifts when inserting IDs
    for match in reversed(matches):
        full_tag = match.group(0)
        start_pos = match.start()
        end_pos = match.end()
        
        # Generate unique ID
        unique_id = f"temp_id_{id_counter}"
        id_counter += 1
        
        # Insert the ID attribute
        if full_tag.endswith('/>'):
            # Self-closing tag - insert before the />
            new_tag = full_tag[:-2] + f' id="{unique_id}"/>'
        else:
            # Opening tag - insert before the >
            new_tag = full_tag[:-1] + f' id="{unique_id}">'
        
        # Replace in the working content
        working_svg_content = working_svg_content[:start_pos] + new_tag + working_svg_content[end_pos:]
    
    return working_svg_content


def filter_svg_elements_by_visual_contribution(
        svg_content: str, groups: list[str], page_width: float,
        page_height: float, min_pixel_diff_threshold: int = 5
    ) -> list[str]:
    """
    Filter SVG elements by testing their visual contribution to the rendered output.

    Args:
        svg_content: The full baseline SVG content as a string
        groups: List of substrings whose visual contribution should be tested
        min_pixel_diff_threshold: Minimum number of changed pixels to consider element visible
        page_width: Page width in points for proper rendering scale
        page_height: Page height in points for proper rendering scale

    Returns:
       Groups list with only visually contributing elements
    """        
    # Render the baseline (full SVG)
    scale: float = 1.0
    render_width: int = int(page_width * scale)
    render_height: int = int(page_height * scale)

    # Test each group by removing it and comparing
    filtered_groups = []
    filtered_svg_content: str = svg_content
    re_render_baseline: bool = True
    baseline_pixels: np.ndarray = np.zeros((render_height, render_width, 3), dtype=np.uint8)

    group_id_match: re.Match[str] | None
    group_id: str
    test_pixels: np.ndarray
    modified_svg_content: str

    for group in groups:
        # Extract the group ID
        group_id_match = re.search(r'id="([^"]+)"', group)
        if not group_id_match:
            raise ValueError("All groups must have an ID assigned for visual contribution testing")
        group_id = group_id_match.group(1)

        # Re-render baseline only if we removed a group last time
        if re_render_baseline:
            baseline_pixels = render_svg_to_pixels(
                filtered_svg_content, width=render_width, height=render_height
            )

        # Render test image without the current group
        modified_svg_content = filtered_svg_content.replace(group, "")
        test_pixels = render_svg_to_pixels(
            modified_svg_content, width=render_width, height=render_height
        )

        # Count changed pixels
        if baseline_pixels.shape == test_pixels.shape:
            pixel_diff = np.sum(baseline_pixels != test_pixels)
            print(f"    Group {group_id}: {pixel_diff} pixels changed")

            if pixel_diff >= min_pixel_diff_threshold:
                filtered_groups.append(group)
                print(f"    → Keeping {group_id} (contributes {pixel_diff} pixels)")
                re_render_baseline = True
                filtered_svg_content = modified_svg_content
            else:
                re_render_baseline = False
                print(f"    → Removing {group_id} (contributes only {pixel_diff} pixels)")
        else:
            raise ValueError(f"Image size mismatch for group {group_id} - baseline: {baseline_pixels.shape}, test: {test_pixels.shape}")

    return filtered_groups


def bboxes_overlap(bbox1, bbox2):
    """Check if two bboxes overlap or are contained within each other."""
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    
    # Check if they don't overlap (easier to check the negative case)
    if x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1:
        return False
    return True


def union_bbox(bbox1, bbox2):
    """Calculate the union bbox of two bboxes."""
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    return (min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4))


def cluster_by_bbox_overlap(
        groups: list[str], defs: list[str],
        bboxes: list[tuple[float, float, float, float]], ids: list[str] | None = None
    ) -> tuple[list[str], list[str], list[tuple[float, float, float, float]]]:
    """
    Cluster groups based on bbox overlaps and return segments.
    
    Args:
        groups: List of group strings
        defs: List of definition strings  
        bboxes: List of bbox tuples (x1, y1, x2, y2)
        ids: Optional list of group ID strings for sorting within clusters
        
    Returns:
        Tuple of (clustered_groups, clustered_defs, clustered_bboxes)
    """
    n = len(groups)
    if n != len(defs) or n != len(bboxes):
        raise ValueError("All input lists must have the same length")
    if ids is not None and len(ids) != n:
        raise ValueError("ids list must have the same length as other inputs")
    
    # Build adjacency matrix for overlapping bboxes
    overlaps = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if bboxes_overlap(bboxes[i], bboxes[j]):
                overlaps[i][j] = overlaps[j][i] = True
    
    # Find connected components using DFS
    visited = [False] * n
    clusters = []
    
    def dfs(idx, cluster):
        visited[idx] = True
        cluster.append(idx)
        for j in range(n):
            if overlaps[idx][j] and not visited[j]:
                dfs(j, cluster)
    
    for i in range(n):
        if not visited[i]:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)
    
    # Build output
    result_groups = []
    result_defs = []
    result_bboxes = []
    
    for cluster in clusters:
        # Sort cluster indices by id if provided
        if ids is not None:
            cluster.sort(key=lambda idx: ids[idx])
        
        # Concatenate groups
        cluster_groups = []
        for idx in cluster:
            cluster_groups.append(groups[idx])
        result_groups.append('\n'.join(cluster_groups))
        
        # Concatenate unique defs
        cluster_defs = set()
        for idx in cluster:
            cluster_defs.add(defs[idx])
        result_defs.append('\n'.join(sorted(cluster_defs)))
        
        # Calculate union bbox
        cluster_bbox = bboxes[cluster[0]]
        for idx in cluster[1:]:
            cluster_bbox = union_bbox(cluster_bbox, bboxes[idx])
        result_bboxes.append(cluster_bbox)
    
    return result_groups, result_defs, result_bboxes