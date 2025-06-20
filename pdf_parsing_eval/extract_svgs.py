# TODO: clip canvas for each segmented SVG to the drawing area
# TODO: add a check to ensure that the segmented SVG has a visible drawing
# TODO: Store the file in S3 and catpure the actual storage url

import dotenv
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, Type
import pymupdf
from tenacity import retry, stop_after_attempt, wait_exponential
from litellm import completion, Choices
from litellm.files.main import ModelResponse
from pydantic import BaseModel

dotenv.load_dotenv(override=True)


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
            unused_clippath_pattern = rf'<clipPath[^>]*id="{re.escape(clippath_id)}"[^>]*>.*?</clipPath>'
            before_length = len(svg_content)
            svg_content = re.sub(unused_clippath_pattern, '', svg_content, flags=re.DOTALL)
            after_length = len(svg_content)
            if before_length != after_length:
                removed_count += 1
    
    return svg_content


def filter_svg_content(svg_content: str) -> str:
    """
    Filter SVG content to capture only vector drawings that are not text or background colors/gradients.
    """
    print(f"  DEBUG: Before filtering - SVG contains {len(re.findall(r'<g[^>]*>', svg_content))} <g> opening tags")
    
    # Remove text-related elements
    # Handle both self-closing and paired font path elements
    svg_content = re.sub(r'<path[^>]*id="font_[^"]*"[^>]*/?>', '', svg_content)
    svg_content = re.sub(r'<use[^>]*data-text[^>]*>', '', svg_content)
    svg_content = re.sub(r'<text[^>]*>.*?</text>', '', svg_content, flags=re.DOTALL)
    
    # Remove image elements
    svg_content = re.sub(r'<image[^>]*>.*?</image>', '', svg_content, flags=re.DOTALL)
    svg_content = re.sub(r'<image[^>]*/>', '', svg_content)
    
    print(f"  DEBUG: After text/image removal - SVG contains {len(re.findall(r'<g[^>]*>', svg_content))} <g> opening tags")
    
    # Remove empty group tags
    # This needs to be done iteratively as removing inner groups may make outer groups empty
    prev_content = ""
    iteration = 0
    while prev_content != svg_content:
        prev_content = svg_content
        iteration += 1
        groups_before = len(re.findall(r'<g[^>]*>', svg_content))
        
        # Remove empty groups (both self-closing and paired)
        svg_content = re.sub(r'<g[^>]*>\s*</g>', '', svg_content)
        svg_content = re.sub(r'<g[^>]*/>', '', svg_content)
        
        groups_after = len(re.findall(r'<g[^>]*>', svg_content))
        print(f"  DEBUG: Iteration {iteration} - groups before: {groups_before}, after: {groups_after}")
        
        if iteration > 10:  # Safety break
            print("  DEBUG: Breaking after 10 iterations to prevent infinite loop")
            break
    
    print(f"  DEBUG: After group removal - SVG contains {len(re.findall(r'<g[^>]*>', svg_content))} <g> opening tags")
    
    # Remove unused clipPath definitions (after text/images/groups are removed)
    svg_content = remove_unused_clippaths(svg_content)
    
    # Clean up empty lines and excessive whitespace
    svg_content = re.sub(r'\n\s*\n', '\n', svg_content)
    svg_content = re.sub(r'^\s*$', '', svg_content, flags=re.MULTILINE)
    
    print(f"  DEBUG: Final filtered SVG contains {len(re.findall(r'<g[^>]*>', svg_content))} <g> opening tags")
    
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
    if '}' in element.tag:
        element.tag = element.tag.split('}')[1]
    
    # Clean up children recursively
    for child in element:
        _clean_namespaces(child)


def segment_svg_groups(svg_content: str) -> list[str]:
    """
    Segment SVG content into separate SVG files based on groups.
    Each <g> tag is treated as a separate drawing.
    """
    try:
        # Find all <g> elements using regex
        group_pattern = r'<g[^>]*>.*?</g>'
        groups = re.findall(group_pattern, svg_content, re.DOTALL)
        
        print(f"  DEBUG: Found {len(groups)} <g> elements using regex")
        
        if not groups:
            return [svg_content]
        
        segments = []
        
        # Extract the SVG header (everything before the first <g>, but excluding any <defs> section)
        svg_header_match = re.match(r'(.*?)(?=<defs|<g[^>]*>)', svg_content, re.DOTALL)
        if svg_header_match:
            svg_header = svg_header_match.group(1)
            # Make sure the header ends with a newline for clean formatting
            if not svg_header.endswith('\n'):
                svg_header += '\n'
        else:
            # Fallback: create a minimal SVG header
            svg_header = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="612" height="792" viewBox="0 0 612 792">
'''
        
        # Extract the defs section if it exists
        defs_match = re.search(r'<defs[^>]*>.*?</defs>', svg_content, re.DOTALL)
        defs_content = defs_match.group(0) if defs_match else ""
        
        print(f"  DEBUG: Found defs section: {len(defs_content)} characters")
        
        for i, group in enumerate(groups):
            print(f"  DEBUG: Processing group {i + 1}/{len(groups)}")
            
            # Find clip-path references in this group
            clip_path_ids = []
            clip_matches = re.findall(r'clip-path="url\(#([^)]+)\)"', group)
            clip_path_ids.extend(clip_matches)
            
            print(f"    DEBUG: Found clip-path references: {clip_path_ids}")
            
            # Create a minimal defs section with only the required clipPaths
            minimal_defs = ""
            if clip_path_ids and defs_content:
                minimal_defs = "<defs>\n"
                for clip_id in clip_path_ids:
                    # Find the specific clipPath definition
                    clip_pattern = rf'<clipPath[^>]*id="{re.escape(clip_id)}"[^>]*>.*?</clipPath>'
                    clip_match = re.search(clip_pattern, defs_content, re.DOTALL)
                    if clip_match:
                        minimal_defs += clip_match.group(0) + "\n"
                        print(f"    DEBUG: Added clipPath definition: {clip_id}")
                    else:
                        print(f"    DEBUG: WARNING - Could not find clipPath definition for: {clip_id}")
                minimal_defs += "</defs>\n"
            
            # Create the complete SVG for this segment
            segment_content = svg_header
            if minimal_defs:
                segment_content += minimal_defs
            segment_content += group + "\n</svg>"
            
            segments.append(segment_content)
            print(f"    DEBUG: Segment {i + 1} created with {len(segment_content)} characters")
        
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
        transform = group_element.attrib.get('transform', '')
        
        # Look for translate values in transform
        translate_match = re.search(r'translate\(([^)]+)\)', transform)
        if translate_match:
            coords = translate_match.group(1).split(',')
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


def extract_json_from_markdown(content: str) -> str:
    """Extract JSON content from markdown code fence if present."""
    if '```json' in content:
        # Take whatever is between ```json ... ```
        return content.split('```json')[1].split('```')[0].strip()
    return content.strip().strip('"\'')  # Fallback to raw string if no JSON code fence


def parse_llm_json_response(content: str, model_class: Type[BaseModel]) -> BaseModel:
    """Parse JSON from LLM response, handling both direct JSON and markdown-fenced output."""
    try:
        return model_class.model_validate(json.loads(content))
    except json.JSONDecodeError:
        # If direct parse fails, try to extract from code fences
        json_str = extract_json_from_markdown(content)
        return model_class.model_validate(json.loads(json_str))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def describe_svg_with_llm(svg_content: str) -> str:
    response = completion(
        model="deepseek/deepseek-chat", 
        messages=[
                {"role": "user",
                "content": """Describe in detail what the SVG represents. Return a JSON object with the following fields: 'label', 'description'.
                Label should be one of: 'chart', 'graph', 'diagram', 'map', 'logo', 'icon', 'table', or 'text_box'.
                Description should be a detailed description of what the SVG shows/communicates."""}
        ],
        response_format={"type": "json_object"},
    )
    if response and isinstance(response, ModelResponse) and isinstance(response.choices[0], Choices) and response.choices[0].message.content:
        return response.choices[0].message.content
    else:
        raise Exception("No valid response from DeepSeek")


async def extract_svgs_from_pdf(pdf_path: str, output_filename: str, temp_dir: str | None = None) -> str | None:
    """
    Extract SVGs from a PDF and save them as JSON blocks.

    Args:
        pdf_path: Path to the PDF file to process
        output_filename: Full path to the output JSON file
        temp_dir: Directory to use for temporary files (optional, creates one if not provided)

    Returns:
        Path to the output JSON file
    """
    # Create temporary directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    os.makedirs(temp_dir, exist_ok=True)
    svg_dir = os.path.join(temp_dir, "svg")
    os.makedirs(svg_dir, exist_ok=True)
    
    svg_blocks = []
    
    try:
        doc = pymupdf.open(pdf_path)
        print(f"Processing PDF: {pdf_path}")
        print(f"Number of pages: {len(doc)}")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_name = f"page_{page_num + 1}"
            
            print(f"Processing page {page_num + 1}...")
            
            # Extract SVG with filtered mode (graphics only)
            svg_info = extract_svg_from_page(page, page_name, svg_dir)
            
            if "error" not in svg_info:
                # Get the filtered SVG content
                filtered_svg_content = filter_svg_content(svg_info)
                
                print(f"  DEBUG: Filtered SVG content contains {len(re.findall(r'<g[^>]*>', filtered_svg_content))} <g> opening tags")
                
                # Check if there's actual content (not just empty SVG)
                if "<path" in filtered_svg_content or "<rect" in filtered_svg_content or "<circle" in filtered_svg_content or "<polygon" in filtered_svg_content:
                    # Segment the SVG using the in-memory filtered content
                    svg_segments = segment_svg_groups(filtered_svg_content)
                    
                    for segment_idx, segment_content in enumerate(svg_segments):
                        # Try to get bbox from the segmented SVG
                        bbox = None
                        if len(svg_segments) > 1:
                            # For segmented SVGs, try to extract bbox from viewBox
                            viewbox_match = re.search(r'viewBox="([^"]+)"', segment_content)
                            if viewbox_match:
                                try:
                                    viewbox_values = [float(x) for x in viewbox_match.group(1).split()]
                                    if len(viewbox_values) == 4:
                                        bbox = viewbox_values
                                except ValueError:
                                    pass
                        
                        # Fallback to full page bbox if no specific bbox found
                        if bbox is None:
                            page_rect = page.rect
                            bbox = [page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1]
                        
                        # Generate description using LLM
                        try:
                            description = await describe_svg_with_llm(segment_content)
                        except Exception as e:
                            print(f"  Warning: Failed to generate description for SVG segment: {e}")
                            description = "Description generation failed"
                        
                        # Create storage path for this segment
                        segment_filename = f"{page_name}_graphics_only_segment_{segment_idx + 1}.svg"
                        
                        segment_path = os.path.join(svg_dir, segment_filename)
                        
                        # Save the segment (always save it to ensure the file exists)
                        with open(segment_path, "w", encoding="utf-8") as f:
                            f.write(segment_content)
                        
                        # Create the properly formatted block
                        svg_block = {
                            "block_type": "svg",
                            "page_number": page_num + 1,
                            "bbox": bbox,
                            "storage_url": segment_path,
                            "description": description
                        }
                        
                        svg_blocks.append(svg_block)
                        print(f"  Added SVG block for page {page_num + 1}, segment {segment_idx + 1}")
                else:
                    print(f"  Skipped page {page_num + 1} - no significant vector graphics")
            elif svg_info.startswith("Error"):
                print(f"  Error extracting SVG from page {page_num + 1}: {svg_info}")
            else:
                print(f"  Skipped page {page_num + 1} - no SVG saved")
        
        doc.close()
        
        # Save the SVG blocks to JSON
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(svg_blocks, f, indent=2, ensure_ascii=False)
        
        print(f"Extracted {len(svg_blocks)} SVG blocks to {output_filename}")
        return output_filename
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None


if __name__ == "__main__":
    import asyncio
    
    asyncio.run(extract_svgs_from_pdf(
        pdf_path="input.pdf",
        output_filename="svgs.json",
        temp_dir="temp"
    ))