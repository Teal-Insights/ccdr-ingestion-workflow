import tempfile
import subprocess
import os
from PIL import Image
import re
import io
import svgelements


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