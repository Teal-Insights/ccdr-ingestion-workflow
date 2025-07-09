# TODO: add a check to ensure that the segmented SVG has a visible drawing
#   - if we convert to PNG, does it have any contrast?
#   - If we grab the portion of the page image that corresponds to the bounding box, does it have any contrast?
# TODO: Use a clustering algorithm to group SVG elements that are part of the same drawing
# TODO: Store the file in S3 and capture the actual storage url

import dotenv
import os
import re
import tempfile
import subprocess
import numpy as np
from typing import List, cast
import pymupdf
from .models import SvgBlock, BlocksDocument, Block
import svgelements
from PIL import Image

dotenv.load_dotenv(override=True)


def has_visible_content(svg_content: str, min_variance_threshold: float = 100.0) -> bool:
    """
    Check if an SVG has visible content by converting it to PNG and analyzing pixel variance.
    
    Args:
        svg_content: The SVG content as a string
        min_variance_threshold: Minimum pixel variance to consider content "visible"
        
    Returns:
        True if the SVG has visible content (sufficient contrast), False otherwise
    """
    try:
        # Create a temporary SVG file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as temp_svg:
            temp_svg.write(svg_content)
            temp_svg_path = temp_svg.name
        
        try:
            # Try to use Inkscape first (most accurate SVG rendering)
            temp_png_path = temp_svg_path.replace('.svg', '.png')
            
            # Try Inkscape first
            try:
                result = subprocess.run([
                    'inkscape', 
                    '--export-type=png',
                    '--export-filename=' + temp_png_path,
                    '--export-width=200',  # Small size for efficiency
                    '--export-height=200',
                    temp_svg_path
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, 'inkscape')
                    
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                # Fallback to rsvg-convert if Inkscape fails or isn't available
                try:
                    result = subprocess.run([
                        'rsvg-convert', 
                        '-w', '200', 
                        '-h', '200', 
                        '-o', temp_png_path,
                        temp_svg_path
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode != 0:
                        raise subprocess.CalledProcessError(result.returncode, 'rsvg-convert')
                        
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    # Fallback to cairosvg if available
                    try:
                        import cairosvg
                        cairosvg.svg2png(url=temp_svg_path, write_to=temp_png_path, output_width=200, output_height=200)
                    except ImportError:
                        print("  Warning: No SVG renderer available (tried inkscape, rsvg-convert, cairosvg)")
                        return True  # Assume visible if we can't check
                    except Exception as e:
                        print(f"  Warning: Failed to render SVG with cairosvg: {e}")
                        return True  # Assume visible if rendering fails
            
            # Check if PNG file was created
            if not os.path.exists(temp_png_path):
                print("  Warning: SVG rendering did not produce PNG file")
                return True  # Assume visible if we can't check
            
            # Load the PNG and analyze pixels
            try:
                with Image.open(temp_png_path) as img:
                    # Convert to grayscale for variance analysis
                    if img.mode in ('RGBA', 'LA'):
                        # Handle transparency by compositing on white background
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'RGBA':
                            background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                        else:  # LA mode
                            background.paste(img.convert('RGB'), mask=img.split()[-1])
                        img = background
                    
                    gray_img = img.convert('L')
                    pixels = np.array(gray_img)
                    
                    # Calculate pixel variance
                    variance = np.var(pixels)
                    
                    print(f"  SVG pixel variance: {variance:.1f} (threshold: {min_variance_threshold})")
                    
                    # Clean up temporary PNG
                    try:
                        os.unlink(temp_png_path)
                    except OSError:
                        pass
                    
                    return bool(variance >= min_variance_threshold)
                    
            except Exception as e:
                print(f"  Warning: Failed to analyze PNG: {e}")
                return True  # Assume visible if analysis fails
                
        finally:
            # Clean up temporary SVG
            try:
                os.unlink(temp_svg_path)
            except OSError:
                pass
                
    except Exception as e:
        print(f"  Warning: Error checking SVG visibility: {e}")
        return True  # Assume visible if check fails completely


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
    if svgelements is None:
        print("  Warning: svgelements not available, returning original SVG")
        return svg_content
    
    try:
        # Parse the SVG from string content using io.StringIO to simulate a file
        import io
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


def filter_svg_content(svg_content: str) -> str:
    """
    Filter SVG content to capture only vector drawings that are not text or background colors/gradients.
    """
    # Remove text-related elements
    # Handle both self-closing and paired font path elements
    svg_content = re.sub(r'<path[^>]*id="font_[^"]*"[^>]*/?>', "", svg_content)
    svg_content = re.sub(r"<use[^>]*data-text[^>]*>", "", svg_content)
    svg_content = re.sub(r"<text[^>]*>.*?</text>", "", svg_content, flags=re.DOTALL)

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


def extract_svg_from_page(page: pymupdf.Page, page_name: str, output_dir: str) -> str:
    """Extract filtered SVG representation of the page (graphics only)"""
    try:
        svg_content = page.get_svg_image()

        # Always filter to get graphics-only content
        filtered_svg_content = filter_svg_content(svg_content)

        return filtered_svg_content

    except Exception as e:
        return f"Error extracting SVG from page: {str(e)}"


def _clean_namespaces(element):
    """Remove namespace prefixes from element and its children for cleaner SVG output"""
    # Remove namespace from tag name
    if "}" in element.tag:
        element.tag = element.tag.split("}")[1]

    # Clean up children recursively
    for child in element:
        _clean_namespaces(child)


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
            # No groups found, clip the entire SVG
            clipped_svg = clip_svg_to_content_bounds(svg_content)
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


def get_group_bbox(group_element) -> tuple[float, float, float, float] | None:
    """
    Calculate the bounding box of a group element.
    Returns (x, y, width, height) or None if cannot be determined.
    """
    try:
        # This is a simplified approach - in reality, calculating SVG bounding boxes
        # is complex and would require parsing all the path data, transforms, etc.
        # For now, we'll look for explicit bbox attributes or use a fallback

        # Check if there's a transform attribute that might give us positioning info
        transform = group_element.attrib.get("transform", "")

        # Look for translate values in transform
        translate_match = re.search(r"translate\(([^)]+)\)", transform)
        if translate_match:
            coords = translate_match.group(1).split(",")
            if len(coords) >= 2:
                try:
                    x = float(coords[0].strip())
                    y = float(coords[1].strip())
                    # Use a default size for now - this is naive but works as a starting point
                    return (x, y, 100, 100)
                except ValueError:
                    pass

        # Fallback: return None to use original SVG dimensions
        return None

    except Exception:
        return None


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
            svg_info = extract_svg_from_page(page, page_name, svg_dir)

            if "error" not in svg_info:
                # Get the filtered SVG content
                filtered_svg_content = filter_svg_content(svg_info)

                print(
                    f"  DEBUG: Filtered SVG content contains {len(re.findall(r'<g[^>]*>', filtered_svg_content))} <g> opening tags"
                )

                # Check if there's actual content (not just empty SVG)
                if (
                    "<path" in filtered_svg_content
                    or "<rect" in filtered_svg_content
                    or "<circle" in filtered_svg_content
                    or "<polygon" in filtered_svg_content
                ):
                    # Segment the SVG using the in-memory filtered content
                    svg_segments = segment_svg_groups(filtered_svg_content)

                    for segment_idx, segment_content in enumerate(svg_segments):
                        # Check if this segment has visible content
                        if not has_visible_content(segment_content):
                            print(
                                f"  Skipped page {page_num + 1}, segment {segment_idx + 1} - no visible content (low contrast)"
                            )
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

                        # Save the segment (always save it to ensure the file exists)
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
            elif svg_info.startswith("Error"):
                print(f"  Error extracting SVG from page {page_num + 1}: {svg_info}")
            else:
                print(f"  Skipped page {page_num + 1} - no SVG saved")

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
