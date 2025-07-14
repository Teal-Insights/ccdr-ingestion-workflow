# TODO: Need to handle excluding groups inside the defs from our segmentation regex
# TODO: Need to make sure we keep mask_* defs as well as clip_* defs
# TODO: Need to fix bug where "<g><g></g></g>" returns 1 match: "<g><g></g>"
# TODO: Related: Need to figure out how to match only top-level groups, not nested ones
# TODO: Need to make sure that if we match a nested group, we keep all the defs that are referenced inside it
# <g clip-path="url(#clip_61)">
#  <g mask="url(#mask_62)">
#    <image x="42" y="38" width="42" height="41" xlink:href="..."/>
#  </g>
# </g>
#
#
# TODO: (Short-term) treat SVG segments as part of same image if one is contained in the other's bounding box
# TODO: Make extraction configurable so we can extract as either SVG or PNG, and make PNG the default for now
# TODO: Include necessary background image elements in the extracted content, but keep bounding box scoped to vector graphics
# TODO: Store the file in S3 and capture the actual storage url
# TODO: Either don't extract non-grouped paths at all, or do some triage to, e.g., only extract if contributing to a drawing
# TODO: Do some triage to keep text that's part of a drawing, e.g., axis labels
# TODO: Check to make sure we're not extracting tables as drawings
# TODO: (Long-term) Use a clustering algorithm to group SVG elements that are part of the same drawing

import dotenv
import os
import re
import tempfile
import numpy as np
from typing import List, cast
import pymupdf
from .models import SvgBlock, BlocksDocument, Block
from utils.svg import (
    render_svg_to_image, extract_viewbox_values,
    clip_svg_to_content_bounds, remove_unused_clippaths,
    remove_element_by_id
)

dotenv.load_dotenv(override=True)


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
            raise Exception("Failed to render baseline SVG - SVG content may be malformed")
        
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
                raise Exception(f"Failed to render test image for group {group_id} - SVG content may be malformed")
            
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
                raise Exception(f"Image size mismatch for group {group_id} - baseline: {baseline_pixels.shape}, test: {test_pixels.shape}")
        
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
        print(f"  Error: Visual contribution filtering failed: {e}")
        raise Exception(f"Visual contribution filtering failed: {e}") from e


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
        group_pattern = r"<g[^>]*>.*?</g>(?![^<]*</defs>)"
        groups = re.findall(group_pattern, svg_content, re.DOTALL)

        print(f"  DEBUG: Found {len(groups)} <g> elements using regex")

        if not groups:
            # No groups found, clip the entire SVG
            filtered_content = remove_unused_clippaths(svg_content)
            clipped_svg = clip_svg_to_content_bounds(filtered_content)
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
