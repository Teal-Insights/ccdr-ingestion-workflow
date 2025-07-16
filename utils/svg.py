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
        svg_text: str, elements_to_extract: list[Literal["g", "mask", "clipPath"]] = ["g"], filter_by: list[str] = [], from_defs: bool = False
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
    matched: bool

    while i < len(content):
        matched = False
        for element, length in [(el, len(el)) for el in elements_to_extract]:
            if content[i:i+length+1] == f'<{element}':
                if not stack:  # Top-level group
                    element_start = i
                stack.append((i, element))
                i = content.find('>', i) + 1
                matched = True
                break

            elif content[i:i+length+3] == f'</{element}>':
                if stack:
                    popped_element = stack.pop()
                    if not stack:  # Stack empty = complete top-level group
                        group_end = i + length + 3
                        if not element_start or popped_element[1] != element:
                            raise ValueError(f"Mismatched tag found: </{popped_element[1]} has no corresponding opening tag")
                        if filter_by:
                            if any(f in content[element_start:group_end] for f in filter_by):
                                elements.append(content[element_start:group_end])
                        else:
                            # No filter, add all groups
                            elements.append(content[element_start:group_end])
                        element_start = None
                        popped_element = None
                else:
                    raise ValueError(f"Mismatched closing tag found: </{element}>")
                i = content.find('>', i) + 1
                matched = True
                break
        if not matched:
            i += 1

    return elements


def filter_svg_content(svg_content: str, filter_text: bool = True, filter_images: bool = True) -> str:
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


def render_svg_to_image(svg_content: str, width: int = 200, height: int = 200) -> Image.Image | None:
    """
    Render SVG content to a PIL Image using available SVG renderers.
    
    Args:
        svg_content: The SVG content as a string
        width: Width for rendering
        height: Height for rendering
        
    Returns:
        PIL Image object or None if rendering failed
    """
    # Create a temporary SVG file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as temp_svg:
        temp_svg.write(svg_content)
        temp_svg_path = temp_svg.name
    
    try:
        # Try to use Inkscape first (most accurate SVG rendering)
        temp_png_path = temp_svg_path.replace('.svg', '.png')
        
        # Try Inkscape first with white background
        try:
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
                img.close()
                os.unlink(temp_png_path)
                return img_copy
                
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
            
        # Fallback to rsvg-convert if Inkscape fails or isn't available
        try:
            result = subprocess.run([
                'rsvg-convert', 
                '-w', str(width), 
                '-h', str(height), 
                '-b', 'white',  # Set background color to white
                '-o', temp_png_path,
                temp_svg_path
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(temp_png_path):
                img = Image.open(temp_png_path)
                # Convert to RGB to ensure no alpha channel
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img.convert('RGB'), mask=img.split()[-1])
                    img_copy = background
                else:
                    img_copy = img.convert('RGB')
                img.close()
                os.unlink(temp_png_path)
                return img_copy
                
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
            
        # Fallback to cairosvg if available
        try:
            import cairosvg
            # cairosvg doesn't have a direct background option, but we'll handle it in PIL
            cairosvg.svg2png(url=temp_svg_path, write_to=temp_png_path, output_width=width, output_height=height)
            if os.path.exists(temp_png_path):
                img = Image.open(temp_png_path)
                # Convert to RGB with white background
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img.convert('RGB'), mask=img.split()[-1])
                    img_copy = background
                else:
                    img_copy = img.convert('RGB')
                img.close()
                os.unlink(temp_png_path)
                return img_copy
        except ImportError:
            pass
        except Exception:
            pass
            
        return None
        
    finally:
        # Clean up temporary SVG
        try:
            os.unlink(temp_svg_path)
        except OSError:
            pass


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


def clip_svg_to_content_bounds(svg_content: str) -> tuple[str, tuple[float, float, float, float]]:
    """
    Clip SVG content to its actual content bounds, removing empty space around the drawing.
    
    Args:
        svg_content: The original SVG content as a string
        
    Returns:
        Tuple of (clipped SVG content, bounding box as (x0, y0, x1, y1))
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

    # Add a small padding around the content
    padding = 2.0
    x0, y0, x1, y1 = bbox  # bbox is guaranteed to be not None here
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
    return update_svg_viewbox_and_dimensions(svg_content, x0, y0, width, height), (x0, y0, x1, y1)


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
    Remove an SVG element by its ID using regex to avoid namespace issues.
    
    Args:
        svg_content: The SVG content as a string
        element_id: The ID of the element to remove
        
    Returns:
        SVG content with the element removed
    """
    try:
        # Use regex to remove the element, handling both self-closing and paired tags
        # Pattern for ACTUAL self-closing tags: <tag ... id="element_id" ... />
        self_closing_pattern = rf'<[^>]*id="{re.escape(element_id)}"[^>]*\s*/>'
        
        # First try to match ACTUAL self-closing tags (must end with />)
        if re.search(self_closing_pattern, svg_content):
            result = re.sub(self_closing_pattern, '', svg_content)
            print(f"    Removed self-closing element {element_id}")
            return result
        
        # Pattern for paired tags: <tag ... id="element_id" ...>...</tag>
        # This is more complex as we need to find the matching closing tag
        start_pattern = rf'<([^>\s]+)[^>]*id="{re.escape(element_id)}"[^>]*>'
        start_match = re.search(start_pattern, svg_content)
        
        if start_match:
            tag_name = start_match.group(1)
            # Find the complete element including its content
            element_pattern = rf'<{re.escape(tag_name)}[^>]*id="{re.escape(element_id)}"[^>]*>.*?</{re.escape(tag_name)}>'
            result = re.sub(element_pattern, '', svg_content, flags=re.DOTALL)
            print(f"    Removed paired element {element_id} (tag: {tag_name})")
            return result
        
        print(f"    Element {element_id} not found")
        return svg_content
        
    except Exception as e:
        print(f"  Warning: Failed to remove element {element_id}: {e}")
        return svg_content


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
    scale = 1.0
    render_width = int(page_width * scale)
    render_height = int(page_height * scale)
    baseline_image = render_svg_to_image(
        svg_content, width=render_width, height=render_height
    )
    if baseline_image is None:
        raise Exception("Failed to render baseline SVG - SVG content may be malformed")

    baseline_pixels = np.array(baseline_image)
    baseline_image.close()

    # Test each group by removing it and comparing
    filtered_groups = []
    filtered_svg_content: str = svg_content
    for group in groups:
        # Extract the group ID for logging
        group_id: str
        group_id_match: re.Match[str] | None = re.search(r'id="([^"]+)"', group)
        if not group_id_match:
            raise ValueError("All groups must have an ID assigned for visual contribution testing")
        group_id = group_id_match.group(1)

        # Render without the group
        modified_svg_content: str = filtered_svg_content.replace(group, "")
        test_image: Image.Image | None = render_svg_to_image(modified_svg_content, width=render_width, height=render_height)
        if test_image is None:
            raise Exception(f"Failed to render test image for group {group_id} - SVG content may be malformed")

        # Count changed pixels
        test_pixels = np.array(test_image)
        if baseline_pixels.shape == test_pixels.shape:
            pixel_diff = np.sum(baseline_pixels != test_pixels)
            print(f"    Group {group_id}: {pixel_diff} pixels changed")

            if pixel_diff >= min_pixel_diff_threshold:
                filtered_groups.append(group)
                print(f"    → Keeping {group_id} (contributes {pixel_diff} pixels)")
            else:
                filtered_svg_content = modified_svg_content
                print(f"    → Removing {group_id} (contributes only {pixel_diff} pixels)")
        else:
            raise ValueError(f"Image size mismatch for group {group_id} - baseline: {baseline_pixels.shape}, test: {test_pixels.shape}")
        test_image.close()

    return filtered_groups
