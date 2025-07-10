# TODO: Use a clustering algorithm to group SVG elements that are part of the same drawing
# TODO: Store the file in S3 and capture the actual storage url

import dotenv
import os
import re
import io
import tempfile
import subprocess
import numpy as np
from typing import List, cast
import pymupdf
from .models import SvgBlock, BlocksDocument, Block
import svgelements
from PIL import Image


dotenv.load_dotenv(override=True)


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


def filter_svg_elements_by_visual_contribution(svg_content: str, min_pixel_diff_threshold: int = 5, page_width: float | None = None, page_height: float | None = None) -> str:
    """
    Filter SVG elements by testing their visual contribution to the rendered output.
    
    Args:
        svg_content: The SVG content as a string (should already have text filtered out)
        min_pixel_diff_threshold: Minimum number of changed pixels to consider element visible
        page_width: Page width in points for proper rendering scale
        page_height: Page height in points for proper rendering scale
        
    Returns:
        Filtered SVG content with only visually contributing elements
    """
    try:
        print("  Testing visual contribution of SVG elements...")
        
        # Find all group elements using regex
        group_pattern = r"<g[^>]*>.*?</g>"
        group_matches = []
        
        for match in re.finditer(group_pattern, svg_content, re.DOTALL):
            group_content = match.group(0)
            group_start = match.start()
            group_end = match.end()
            
            # Check if this group is inside <defs> by looking at the content before it
            content_before = svg_content[:group_start]
            
            # Count unclosed <defs> tags before this group
            defs_opens = len(re.findall(r'<defs[^>]*>', content_before))
            defs_closes = len(re.findall(r'</defs>', content_before))
            inside_defs = defs_opens > defs_closes
            
            if not inside_defs:
                # Check if this group contains image elements
                contains_image = bool(re.search(r'<image[^>]*/?>', group_content))
                
                if not contains_image:
                    # Extract the ID if it exists
                    id_match = re.search(r'id="([^"]*)"', group_content)
                    group_id = id_match.group(1) if id_match else None
                    
                    group_matches.append({
                        'content': group_content,
                        'start': group_start,
                        'end': group_end,
                        'id': group_id,
                        'match_obj': match
                    })
                else:
                    id_match = re.search(r'id="([^"]*)"', group_content)
                    group_id = id_match.group(1) if id_match else "no-id"
                    print(f"    Excluding group from testing (contains image): {group_id}")
        
        if not group_matches:
            print("  No drawing group elements found for testing, keeping original SVG")
            return svg_content
        
        print(f"  Found {len(group_matches)} drawing group elements to test for visual contribution")
        
        # Assign temporary IDs to groups that don't have them
        working_svg_content = svg_content
        temp_id_counter = 0
        
        # Process groups from end to start to avoid position shifts
        for group_info in reversed(group_matches):
            if group_info['id'] is None:
                temp_id = f"temp_group_{temp_id_counter}"
                temp_id_counter += 1
                
                # Insert the ID into the opening tag
                original_content = group_info['content']
                # Find the opening <g tag and insert id attribute
                opening_tag_match = re.match(r'(<g[^>]*)(>)', original_content)
                if opening_tag_match:
                    new_content = f'{opening_tag_match.group(1)} id="{temp_id}"{opening_tag_match.group(2)}{original_content[opening_tag_match.end():]}'
                    # Replace in the working content
                    working_svg_content = (
                        working_svg_content[:group_info['start']] + 
                        new_content + 
                        working_svg_content[group_info['end']:]
                    )
                    # Update the group info
                    group_info['id'] = temp_id
                    group_info['content'] = new_content
        
        # Calculate proper render dimensions
        if page_width and page_height:
            # Use page dimensions with reasonable scaling
            max_dimension = 800
            aspect_ratio = page_width / page_height
            
            if page_width > page_height:
                render_width = min(max_dimension, int(page_width))
                render_height = int(render_width / aspect_ratio)
            else:
                render_height = min(max_dimension, int(page_height))
                render_width = int(render_height * aspect_ratio)
            
            print(f"  Using render dimensions: {render_width} x {render_height} (from page: {page_width:.1f} x {page_height:.1f})")
        else:
            # Fallback to default dimensions
            render_width, render_height = 200, 200
            print(f"  Warning: No page dimensions provided, using default: {render_width} x {render_height}")
        
        # Render the baseline (full SVG)
        baseline_image = render_svg_to_image(working_svg_content, width=render_width, height=render_height)
        if baseline_image is None:
            print("  Warning: Could not render baseline SVG, keeping original")
            return svg_content
        
        baseline_pixels = np.array(baseline_image)
        visible_element_ids = set()
        
        # Test each group by removing it and comparing
        for i, group_info in enumerate(group_matches):
            group_id = group_info['id']
            print(f"    Testing group {i+1}/{len(group_matches)}: {group_id}")
            
            # Create SVG without this group
            svg_without_group = remove_element_by_id(working_svg_content, group_id)
            
            # Render without the group
            test_image = render_svg_to_image(svg_without_group, width=render_width, height=render_height)
            if test_image is None:
                print(f"    Warning: Could not render test image for {group_id}, assuming visible")
                visible_element_ids.add(group_id)
                continue
            
            # Compare pixels
            test_pixels = np.array(test_image)
            
            # Count changed pixels
            if baseline_pixels.shape == test_pixels.shape:
                pixel_diff = np.sum(baseline_pixels != test_pixels)
                print(f"    Group {group_id}: {pixel_diff} pixels changed")
                
                if pixel_diff >= min_pixel_diff_threshold:
                    visible_element_ids.add(group_id)
                    print(f"    → Keeping {group_id} (contributes {pixel_diff} pixels)")
                else:
                    print(f"    → Removing {group_id} (contributes only {pixel_diff} pixels)")
            else:
                print(f"    Warning: Image size mismatch for {group_id}, assuming visible")
                visible_element_ids.add(group_id)
        
        # Build filtered SVG by removing non-contributing groups
        filtered_svg = working_svg_content
        total_elements = len(group_matches)
        visible_count = len(visible_element_ids)
        
        for group_info in group_matches:
            group_id = group_info['id']
            if group_id and group_id not in visible_element_ids:
                filtered_svg = remove_element_by_id(filtered_svg, group_id)
        
        print(f"  Visual contribution analysis: kept {visible_count}/{total_elements} elements")
        return filtered_svg
        
    except Exception as e:
        print(f"  Warning: Visual contribution filtering failed: {e}")
        print("  Falling back to original SVG content")
        return svg_content


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


def clip_svg_to_content_bounds(svg_content: str) -> str:
    """
    Clip SVG content to its actual content bounds, removing empty space around the drawing.
    
    Args:
        svg_content: The original SVG content as a string
        
    Returns:
        Clipped SVG content with updated viewBox and dimensions
    """    
    try:
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
                try:
                    element_bbox = element.bbox()
                    # Check if bbox is valid - must be a sequence with 4 numeric values
                    if element_bbox is not None:
                        try:
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
                        except (TypeError, ValueError, IndexError):
                            # Skip if bbox can't be converted to 4 numeric values
                            pass
                except Exception:
                    # Skip elements that can't provide bbox
                    continue
        
        if bbox is None:
            print("  Warning: No valid content bounds found, returning original SVG")
            return svg_content
            
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
        return update_svg_viewbox_and_dimensions(svg_content, x0, y0, width, height)
        
    except Exception as e:
        print(f"  Warning: Failed to clip SVG content bounds: {e}")
        return svg_content


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


def extract_svg_header(svg_content: str) -> str:
    """Extract the SVG header from the SVG content (everything before the first
    <g>, but excluding any <defs> section).
    If no <g> is found, use the viewBox to determine the dimensions.
    If no viewBox is found, something has gone wrong; let's raise an error.
    """
    svg_header_match = re.match(r"(.*?)(?=<defs|<g[^>]*>)", svg_content, re.DOTALL)
    
    if svg_header_match:
        svg_header = svg_header_match.group(1)
    else:
        print("Warning: No header found in SVG content")
        # Create template string with formattable values
        svg_header_template = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="{width}px" height="{height}px" viewBox="0 0 {width} {height}">
"""
        # Fallback: create a minimal SVG header using extracted viewbox
        viewbox_values = extract_viewbox_values(svg_content)
        if viewbox_values and len(viewbox_values) == 4:
            width = viewbox_values[2]
            height = viewbox_values[3]
            svg_header = svg_header_template.format(width=width, height=height)
        else:
            raise ValueError("No viewBox found in SVG content")

    # Make sure the header ends with a newline for clean formatting
    if not svg_header.endswith("\n"):
        svg_header += "\n"

    return svg_header


def segment_svg_groups(svg_content: str) -> list[str]:
    """
    Segment SVG content into separate SVG files based on groups.
    Each <g> tag is treated as a separate drawing.
    """
    try:
        # Find all <g> elements using regex
        group_pattern = r"<g[^>]*>.*?</g>"
        groups = re.findall(group_pattern, svg_content, re.DOTALL)

        print(f"  DEBUG: Found {len(groups)} <g> elements using regex")

        if not groups:
            # No groups found, clip the entire SVG and remove defs section
            # since PyMuPDF only references clip paths from groups
            svg_without_defs = re.sub(r"<defs[^>]*>.*?</defs>", "", svg_content, flags=re.DOTALL).strip()
            clipped_svg = clip_svg_to_content_bounds(svg_without_defs)
            return [clipped_svg]

        segments = []

        svg_header = extract_svg_header(svg_content)

        # Extract the defs section if it exists
        defs_match = re.search(r"<defs[^>]*>.*?</defs>", svg_content, re.DOTALL)
        defs_content = defs_match.group(0) if defs_match else ""

        print(f"  DEBUG: Found defs section: {len(defs_content)} characters")

        for i, group in enumerate(groups):
            print(f"  DEBUG: Processing group {i + 1}/{len(groups)}")

            # Find clip-path references in this group
            clip_path_ids = []
            clip_matches = re.findall(r'clip-path="url\(#([^)]+)\)"', group)
            clip_path_ids.extend(clip_matches)

            print(f"  DEBUG: Found clip-path references: {clip_path_ids}")

            # Create a minimal defs section with only the required clipPaths
            minimal_defs = ""
            if clip_path_ids and defs_content:
                minimal_defs = "<defs>\n"
                for clip_id in clip_path_ids:
                    # Find the specific clipPath definition
                    clip_pattern = (
                        rf'<clipPath[^>]*id="{re.escape(clip_id)}"[^>]*>.*?</clipPath>'
                    )
                    clip_match = re.search(clip_pattern, defs_content, re.DOTALL)
                    if clip_match:
                        minimal_defs += clip_match.group(0) + "\n"
                        print(f"  DEBUG: Added clipPath definition: {clip_id}")
                    else:
                        print(
                            f"  DEBUG: WARNING - Could not find clipPath definition for: {clip_id}"
                        )
                minimal_defs += "</defs>\n"

            # Create the complete SVG for this segment
            segment_content = svg_header
            if minimal_defs:
                segment_content += minimal_defs
            segment_content += group + "\n</svg>"

            # Clip the segment to its content bounds
            clipped_segment = clip_svg_to_content_bounds(segment_content)
            segments.append(clipped_segment)
            print(
                f"  DEBUG: Segment {i + 1} created with {len(clipped_segment)} characters (clipped)"
            )

        print(f"  DEBUG: Created {len(segments)} segments from {len(groups)} groups")
        return segments

    except Exception as e:
        print(f"Warning: Error during SVG segmentation: {e}")
        return [svg_content]


def has_meaningful_content(svg_content: str) -> bool:
    """
    Check if SVG content has anything meaningful beyond defs and whitespace.
    
    Args:
        svg_content: The SVG content to check
        
    Returns:
        True if the SVG contains anything other than defs and whitespace, False otherwise
    """
    # Remove all unwanted parts in one go: XML declaration, SVG tags, and defs sections
    cleaned = re.sub(r'<\?xml[^>]*>|</?svg[^>]*>|<defs[^>]*>.*?</defs>', '', svg_content, flags=re.DOTALL)
    
    # If anything non-whitespace remains, it's meaningful content
    return bool(cleaned.strip())


def extract_svgs_from_pdf(
    pdf_path: str,
    output_filename: str,
    svgs_dir: str | None = None,
) -> str:
    """
    Extract SVGs from a PDF and save them as JSON blocks.

    Args:
        pdf_path: Path to the PDF file to process
        output_filename: Full path to the output JSON file
        svgs_dir: Directory to which to save the SVGs

    Returns:
        Path to the output JSON file
    """
    # Create temporary directory if not provided
    if svgs_dir is None:
        svgs_dir = tempfile.mkdtemp()

    os.makedirs(svgs_dir, exist_ok=True)
    svg_dir = os.path.join(svgs_dir, "svg")
    os.makedirs(svg_dir, exist_ok=True)

    svg_blocks = []

    try:
        doc = pymupdf.open(pdf_path)
        print(f"Processing PDF: {pdf_path}")
        print(f"Number of pages: {len(doc)}")

        total_pages = len(doc)

        # Extract all SVGs
        print("\n=== Extracting SVGs ===")
        for page_num in range(total_pages):
            page = doc[page_num]
            page_name = f"page_{page_num + 1}"

            print(f"Processing page {page_num + 1}...")

            # Extract SVG with filtered mode (graphics only)
            try:
                svg_content = page.get_svg_image()
                
                # Get page dimensions for proper rendering
                page_rect = page.rect
                page_width = page_rect.width
                page_height = page_rect.height
                print(f"  Page dimensions: {page_width:.1f} x {page_height:.1f}")
                
                # Step 1: Filter text early in the pipeline (but keep images for visibility testing)
                text_filtered_svg = filter_svg_content(svg_content, filter_text=True, filter_images=False)

                print(
                    f"  DEBUG: Text-filtered SVG content contains {len(re.findall(r'<g[^>]*>', text_filtered_svg))} <g> opening tags"
                )

                # Check if there's actual content (not just empty SVG)
                if (
                    "<path" in text_filtered_svg
                    or "<rect" in text_filtered_svg
                    or "<circle" in text_filtered_svg
                    or "<polygon" in text_filtered_svg
                ):
                    # Step 2: Apply visual contribution filtering (images are kept but excluded from testing)
                    print("  Applying visual contribution filtering...")
                    visually_filtered_svg = filter_svg_elements_by_visual_contribution(
                        text_filtered_svg, 
                        page_width=page_width, 
                        page_height=page_height
                    )
                    
                    # Skip if visual contribution analysis determined the SVG has no meaningful content
                    if not visually_filtered_svg.strip():
                        print(f"  Skipped page {page_num + 1} - SVG has no visually contributing elements")
                        continue
                    
                    # Step 3: Filter images after visibility testing but before segmentation
                    print("  Removing image elements from final output...")
                    final_filtered_svg = filter_svg_content(visually_filtered_svg, filter_text=False, filter_images=True)
                    
                    # Segment the SVG using the final filtered content
                    svg_segments = segment_svg_groups(final_filtered_svg)

                    for segment_idx, segment_content in enumerate(svg_segments):
                        
                        # Check if this segment has meaningful content before processing
                        if not has_meaningful_content(segment_content):
                            print(f"  Skipped segment {segment_idx + 1} on page {page_num + 1} - no meaningful drawable content")
                            continue
                        
                        # Try to get bbox from the segmented SVG
                        bbox = None
                        if len(svg_segments) > 1:
                            # For segmented SVGs, try to extract bbox from viewBox
                            viewbox_values = extract_viewbox_values(segment_content)
                            if viewbox_values:
                                bbox = list(viewbox_values)

                        # Fallback to full page bbox if no specific bbox found
                        if bbox is None:
                            page_rect = page.rect
                            bbox = [
                                page_rect.x0,
                                page_rect.y0,
                                page_rect.x1,
                                page_rect.y1,
                            ]

                        # Create storage path for this segment
                        segment_filename = (
                            f"{page_name}_graphics_only_segment_{segment_idx + 1}.svg"
                        )
                        segment_path = os.path.join(svg_dir, segment_filename)

                        # Save the segment (only save if it has meaningful content)
                        with open(segment_path, "w", encoding="utf-8") as f:
                            f.write(segment_content)

                        # Create the block without description
                        svg_block = {
                            "block_type": "svg",
                            "page_number": page_num + 1,
                            "bbox": bbox,
                            "storage_url": segment_path,
                            "description": None,
                        }

                        svg_blocks.append(svg_block)
                        print(
                            f"  Added SVG block for page {page_num + 1}, segment {segment_idx + 1}"
                        )
                else:
                    print(
                        f"  Skipped page {page_num + 1} - no significant vector graphics"
                    )
            except Exception as e:
                print(f"  Error extracting SVG from page {page_num + 1}: {e}")

        doc.close()

        # Convert dict blocks to Pydantic models
        svg_pydantic_blocks: List[SvgBlock] = []
        for block_data in svg_blocks:
            svg_block = SvgBlock(**block_data)
            svg_pydantic_blocks.append(svg_block)

        # Create output document using Pydantic model
        output_data = BlocksDocument(
            pdf_path=pdf_path,
            total_pages=total_pages,
            total_blocks=len(svg_pydantic_blocks),
            blocks=cast(List[Block], svg_pydantic_blocks),
        )

        # Save the SVG blocks to JSON
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(output_data.model_dump_json(indent=2, exclude_none=True))

        print(
            f"\n✓ Extracted {len(svg_blocks)} SVG blocks to {output_filename}"
        )
        return output_filename

    except Exception as e:
        raise Exception(f"Failed to extract SVGs from PDF: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run -m transform.extract_svgs <pdf_file>")
        print("Example: uv run -m transform.extract_svgs document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Create a real temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="svgs_test_")
    output_filename = os.path.join(temp_dir, "svgs.json")

    try:
        output_path = extract_svgs_from_pdf(
            pdf_path=pdf_path,
            output_filename=output_filename,
            svgs_dir=temp_dir,
        )

        print("SVGs extracted successfully!")
        print(f"Output file: {output_path}")
        print(f"Temporary directory: {temp_dir}")
        print(f"Note: Clean up temporary directory when done: rm -rf {temp_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
