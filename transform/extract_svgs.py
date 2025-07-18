# TODO: Processing input2.pdf errors on page 4: "Error: No visible elements with geometry found in SVG to determine content bounds"; calculating bbox of group in context of full SVG may solve this?
# TODO: Finish implementing steps 8-14 in the SVG processing pipeline below
# TODO: (Short-term) treat SVG segments as part of same image if one is contained in the other's bounding box
# TODO: Extract both SVG and PNG, updating the Pydantic model accordingly
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
2. Delete text and image elements early (to simplify/speed up subsequent processing)
3. Extract top-level groups in the SVG body (ignoring those in `<defs>`)
4. Render page with and without each group to test visual contribution
5. Keep only groups that contain at least one visible drawing element (path, rect, circle, polygon), not just images
6. For groups that contribute visually, extract `<defs>` they reference
7. Extract SVG header
8. Calculate the bbox of each group
9. Concatenate groups with overlapping or contained bboxes (and take the union of associated defs) to get segment contents
10. Clip each segment to its content bounds
11. Write SVG segments to separate SVG files
12. Also save PNG page crop of bounding box corresponding to each SVG segment
13. Create JSON blocks for each SVG segment (without description)

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
    filter_svg_content, extract_svg_header,
    filter_svg_elements_by_visual_contribution,
    extract_elements, assign_ids_to_elements,
    clip_svg_to_bounding_box, get_group_bounding_box,
    cluster_by_bbox_overlap, remove_element_by_id
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
        # 1. Extract raw SVG from PDF page using `page.get_svg_image()`
        page: pymupdf.Page = doc[page_num]
        one_indexed_page_num: int = page_num + 1
        svg_content: str = assign_ids_to_elements(
            page.get_svg_image()
        )

        print(f"Processing page {str(one_indexed_page_num)}...")

        # Get page dimensions for proper rendering
        page_rect: pymupdf.Rect = cast(pymupdf.Rect, page.rect)
        page_width: float = cast(float, page_rect.width)
        page_height: float = cast(float, page_rect.height)
        print(f"  Page dimensions: {page_width:.1f} x {page_height:.1f}")

        # 2. Delete text elements early to simplify/speed up subsequent processing
        #    (This is the baseline for visual contribution testing)
        print("  Removing text elements...")
        text_filtered_svg: str = filter_svg_content(svg_content, filter_text=True, filter_images=False)

        # Delete image elements (baseline for extracting drawing groups)
        image_filtered_svg: str = filter_svg_content(
            text_filtered_svg, filter_text=False, filter_images=True
        )

        # TODO: If we wanted to keep top-level paths, we could group them here (but would need to be careful about rendering order; alternatively, do this at the end)
        top_level_paths: list[str] = extract_elements(image_filtered_svg, ["path", "rect", "circle", "ellipse", "line", "polyline", "polygon"])
        if extract_paths:
            print("  Note: Top-level path extraction not currently implemented, skipping...")
        else:
            # Filter out top-level paths
            for path in top_level_paths:
                # Get the id attribute of the path
                match = re.search(r'id="([^"]+)"', path)
                if not match:
                    raise ValueError("Path without ID found after assigning IDs")
                path_id = match.group(1)
                image_filtered_svg = remove_element_by_id(image_filtered_svg, path_id)

        # 3. Extract top-level groups in the SVG body (ignoring those in `<defs>`)
        print("  Extracting top-level groups...")
        top_level_groups: list[str] = extract_elements(image_filtered_svg, ["g"])

        # 4. Render page with and without each group to test visual contribution, and filter accordingly
        print("  Applying visual contribution filtering...")
        visually_filtered_groups: list[str] = filter_svg_elements_by_visual_contribution(
            text_filtered_svg,
            top_level_groups,
            page_width=page_width, 
            page_height=page_height,
            min_pixel_diff_threshold=5,
        )

        # 5. Keep only groups that contain a visible drawing, not just images
        image_filtered_groups: list[str] = []
        for group in visually_filtered_groups:
            image_filtered_group: str = filter_svg_content(group, filter_text=False, filter_images=True).strip()
            if image_filtered_group:
                image_filtered_groups.append(group)

        # Skip page if no visually contributing groups
        if not image_filtered_groups:
            print(f"  Skipped page {page_num + 1} - SVG has no visually contributing elements")
            continue

        # 6. For groups that contribute visually, extract `<defs>` they reference
        print("  Extracting referenced defs...")
        image_filtered_svg: str = filter_svg_content(text_filtered_svg, filter_text=False, filter_images=True)
        defs: list[str] = []
        for group in image_filtered_groups:
            # Get all substrings in group matching "mask_", "clip_"
            identifiers: list[str] = re.findall(r'(mask|clip)_\d+', group)
            if identifiers:
                # Extract corresponding defs from the full text-filtered SVG
                referenced_defs: list[str] = extract_elements(
                    image_filtered_svg,
                    ["g", "mask", "clipPath"],
                    filter_by=identifiers,
                    from_defs=True
                )

                joined_defs: str = "\n".join(referenced_defs).strip()
                defs.append(joined_defs)
        assert len(defs) == len(image_filtered_groups), "Mismatch in number of groups and defs"

        # 7. Extract SVG header
        svg_header: str = extract_svg_header(svg_content, page_num)
        
        def wrap_with_svg_tags(body: str) -> str:
            return f"{svg_header}\n{body}\n</svg>"

        # 8. Calculate the bbox of each group
        group_ids: list[str] = []
        for group in image_filtered_groups:
            match = re.search(r'id="([^"]+)"', group)
            if match:
                group_ids.append(match.group(1))
            else:
                raise ValueError("Group without ID found after assigning IDs")
        assert len(group_ids) == len(image_filtered_groups), "Mismatch in number of groups and IDs"

        print("  Calculating bounding boxes for groups...")
        bboxes: list[tuple[float, float, float, float]] = []
        for group_id in group_ids:
            group_bbox: tuple[float, float, float, float] = get_group_bounding_box(
                image_filtered_svg, group_id
            )
            bboxes.append(group_bbox)

        # 9. Concatenate groups with overlapping or contained bboxes
        # (and take the union of associated defs) to get segment content
        print("  Grouping overlapping/contained bounding boxes into segments...")
        clustered_groups, clustered_defs, unified_bbox = cluster_by_bbox_overlap(
            image_filtered_groups,
            defs,
            bboxes,
            group_ids
        )
        clustered_groups = cast(list[str], clustered_groups)
        clustered_defs = cast(list[str], clustered_defs)
        unified_bboxes = cast(list[tuple[float, float, float, float]], unified_bbox)

        # Construct SVG segments
        for i in range(len(clustered_groups)):
            print(f"  Processing segment {i + 1} of {len(clustered_groups)}...")

            unified_bbox = unified_bboxes[i]
            concatenated_groups = clustered_groups[i]
            concatenated_defs = clustered_defs[i]

            segment: str = (
                wrap_with_svg_tags(f"<defs>\n{concatenated_defs}\n</defs>\n{concatenated_groups}")
            )

            # 10. Clip each segment to its content bounds
            clipped_content = clip_svg_to_bounding_box(segment, unified_bbox)

            # 11. Save segments as separate SVG files
            # Create storage path for this segment
            segment_filename = (
                f"page_{str(one_indexed_page_num)}_graphics_only_segment_{i + 1}.svg"
            )
            segment_path = os.path.join(svg_dir, segment_filename)

            with open(segment_path, "w", encoding="utf-8") as f:
                f.write(clipped_content)
            print(f"    Saved SVG segment to {segment_path}")

            # 12. TODO: Also save PNG page crop of bounding box corresponding to each SVG segment

            # 13. Create JSON blocks for each SVG segment (without description)
            svg_block = {
                "block_type": "svg",
                "page_number": page_num + 1,
                "bbox": unified_bbox,
                "storage_url": segment_path,
                "description": None,
            }

            svg_blocks.append(svg_block)
            print(
                f"  Added SVG block for page {page_num + 1}, segment {i + 1}"
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
    parser.add_argument(
        "--extract-paths", action="store_true", 
        help="Whether to extract top-level path elements (currently unused)"
    )
    
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
