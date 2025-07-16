# TODO: Deduplicate nested group matching and move to a shared utility function in utils/svg.py
# TODO: Diagnose why we're only successfully extracting a segment with ungrouped paths in the case where the page has no groups
# TODO: Need to make sure that if we match a nested group, we keep all the defs that are referenced inside it
# <g clip-path="url(#clip_61)">
#  <g mask="url(#mask_62)">
#    <image x="42" y="38" width="42" height="41" xlink:href="..."/>
#  </g>
# </g>
# Make sure we keep mask_* defs as well as clip_* defs
#
# TODO: (Short-term) treat SVG segments as part of same image if one is contained in the other's bounding box
# TODO: Make extraction configurable so we can extract as either SVG or PNG, and make PNG the default for now
# TODO: Include necessary background image elements in the extracted content, but keep bounding box scoped to vector graphics
# TODO: Store the file in S3 and capture the actual storage url
# TODO: Either don't extract non-grouped paths at all, or do some triage to, e.g., only extract if contributing to a drawing
# TODO: Do some triage to keep text that's part of a drawing, e.g., axis labels
# TODO: Check to make sure we're not extracting tables as drawings
# TODO: (Long-term) Use a clustering algorithm to group SVG elements that are part of the same drawing

"""
### SVG Processing Pipeline

The SVG extraction follows a specific order of operations:

1. Extract raw SVG from PDF page using `page.get_svg_image()`
2. Delete text elements early (to simplify/speed up subsequent processing)
3. Extract top-level groups in the SVG body (ignoring those in `<defs>`)
4. Keep only groups that contain at least one visible drawing element (path, rect, circle, polygon), not just text or images
5. Render page with and without each group to test visual contribution
6. For groups that contribute visually, extract `<defs>` they reference
7. Extract SVG header
8. Calculate the bbox of each group
9. Concatenate groups with overlapping or contained bboxes (and take the union of associated defs) to get segment contents
10. Write SVG segments to separate files, using the same header for each
11. Clip each segment to its content bounds
12. Save segments as separate SVG files
13. Also save PNG page crop of bounding box corresponding to each SVG segment
14. Create JSON blocks for each SVG segment (without description)

**Critical SVG Design Decisions**:
- Only test `<g>` elements outside `<defs>` (actual drawing units)
- Exclude image elements from extracted SVG code, but not from visual contribution testing or PNG snapshots
- Use pixel-level comparison to detect visual impact
- Remove elements causing fewer than 5 pixel changes
- Preserve original SVG structure exactly as PyMuPDF generates it
- No XML parsing, just regex and string manipulation
- Fail the pipeline if any step fails, so we don't produce partial results
"""

import dotenv
import os
import re
import tempfile
from typing import List, cast
import pymupdf
from .models import SvgBlock, BlocksDocument, Block
from utils.svg import (
    extract_viewbox_values, segment_svg_groups,
    filter_svg_content, has_meaningful_content,
    filter_svg_elements_by_visual_contribution
)

dotenv.load_dotenv(override=True)


def extract_svgs_from_pdf(
    pdf_path: str,
    output_filename: str,
    output_dir: str | None = None,
    extract_paths: bool = False,
) -> str:
    """
    Extract SVGs from a PDF and save them as JSON blocks.

    Args:
        pdf_path: Path to the PDF file to process
        output_filename: Full path to the output JSON file
        output_dir: Directory to which to save the SVGs
        extract_paths: Whether to extract paths (currently unused)

    Returns:
        Path to the output JSON file
    """
    # Create temporary directory if no output dir provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for SVG and PNG files if they don't exist
    svg_dir: str = os.path.join(output_dir, "svg_files")
    png_dir: str = os.path.join(output_dir, "png_files")

    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # Open the PDF document
    doc: pymupdf.Document = pymupdf.open(pdf_path)
    total_pages: int = len(doc)

    print(f"Processing PDF: {pdf_path}")
    print(f"Number of pages: {total_pages}")

    # Loop through pages and extract all SVGs as blocks
    svg_blocks: List[dict] = []
    print("\n=== Extracting SVGs ===")

    for page_num in range(total_pages):
        # Get raw SVG content from the pymupdf page object
        page: pymupdf.Page = doc[page_num]
        one_indexed_page_num: int = page_num + 1
        svg_content: str = page.get_svg_image()  

        print(f"Processing page {str(one_indexed_page_num)}...")

        # Get page dimensions for proper rendering
        page_rect: pymupdf.Rect = cast(pymupdf.Rect, page.rect)
        page_width: float = cast(float, page_rect.width)
        page_height: float = cast(float, page_rect.height)
        print(f"  Page dimensions: {page_width:.1f} x {page_height:.1f}")

        # Step 1: Remove text early in the pipeline (but keep images for visibility testing)
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
                    f"page_{str(one_indexed_page_num)}_graphics_only_segment_{segment_idx + 1}.svg"
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


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Extract SVGs from PDF files")
    parser.add_argument("pdf_file", help="Path to the PDF file to process")
    parser.add_argument("--extract-paths", action="store_true", 
                        help="Whether to extract paths (currently unused)")
    
    args = parser.parse_args()

    pdf_path = args.pdf_file

    # Create a real temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="svgs_test_")
    output_filename = os.path.join(temp_dir, "svgs.json")

    try:
        output_path = extract_svgs_from_pdf(
            pdf_path=pdf_path,
            output_filename=output_filename,
            output_dir=temp_dir,
            extract_paths=args.extract_paths,
        )

        print("SVGs extracted successfully!")
        print(f"Output file: {output_path}")
        print(f"Temporary directory: {temp_dir}")
        print(f"Note: Clean up temporary directory when done: rm -rf {temp_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
