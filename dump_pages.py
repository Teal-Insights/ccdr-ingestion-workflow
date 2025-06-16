import pymupdf
import json
import os
import sys
import re
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Union[str, Any]:
        if isinstance(obj, bytes):
            return f"<bytes: {len(obj)} bytes>"
        elif hasattr(obj, '__dict__'):
            return str(obj)
        return super().default(obj)

def extract_images_from_page(page: pymupdf.Page, page_name: str, output_dir: str) -> List[Dict[str, Any]]:
    """Extract and save images from a page"""
    images_info: List[Dict[str, Any]] = []
    
    # Get image list from page
    image_list = page.get_images(full=True)
    
    print(f"    PyMuPDF detected {len(image_list)} images in page")
    
    for img_index, img in enumerate(image_list):
        print(f"    Processing image {img_index + 1}: xref={img[0]}, bbox info available")
        # Extract image
        xref = img[0]  # xref is the first element
        pix = pymupdf.Pixmap(page.parent, xref)
        
        if pix.n - pix.alpha < 4:  # Can convert to PNG
            image_filename = f"{page_name}_image_{img_index + 1}.png"
            image_path = f"{output_dir}/{image_filename}"
            pix.save(image_path)
            
            # Get image info
            bbox = page.get_image_bbox(img)
            image_info = {
                "filename": image_filename,
                "xref": xref,
                "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                "width": pix.width,
                "height": pix.height,
                "colorspace": pix.colorspace.name if pix.colorspace else "unknown",
                "alpha": bool(pix.alpha),
                "size_bytes": len(pix.pil_tobytes("PNG"))
            }
            images_info.append(image_info)
        else:
            # Convert CMYK to RGB first
            pix1 = pymupdf.Pixmap(pymupdf.csRGB, pix)
            image_filename = f"{page_name}_image_{img_index + 1}.png"
            image_path = f"{output_dir}/{image_filename}"
            pix1.save(image_path)
            
            bbox = page.get_image_bbox(img)
            image_info = {
                "filename": image_filename,
                "xref": xref,
                "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                "width": pix1.width,
                "height": pix1.height,
                "colorspace": "RGB (converted from CMYK)",
                "alpha": bool(pix1.alpha),
                "size_bytes": len(pix1.pil_tobytes("PNG"))
            }
            images_info.append(image_info)
            pix1 = None
        
        pix = None
    
    return images_info

def extract_drawings_from_page(page: pymupdf.Page) -> List[Dict[str, Any]]:
    """Extract drawing/vector graphics information from a page"""
    drawings_info: List[Dict[str, Any]] = []
    
    try:
        # Get page drawings - returns list of dicts with drawing path info
        drawings = page.get_drawings()
        
        for draw_index, drawing in enumerate(drawings):
            # Based on PyMuPDF source: each drawing is a dict with keys like:
            # "type", "rect", "items", "fill", "color", "width", etc.
            drawing_info = {
                "drawing_id": draw_index + 1,
                "type": drawing.get("type", "unknown"),
                "bbox": [drawing["rect"].x0, drawing["rect"].y0, drawing["rect"].x1, drawing["rect"].y1] if "rect" in drawing else [0, 0, 0, 0],
                "fill": drawing.get("fill", None),
                "color": drawing.get("color", None),
                "width": drawing.get("width", None),
                "stroke_opacity": drawing.get("stroke_opacity", None),
                "fill_opacity": drawing.get("fill_opacity", None),
                "items_count": len(drawing.get("items", [])),
                "items": []
            }
            
            # Extract drawing items - each item is a tuple like ("cmd", Point, Point, ...)
            for item in drawing.get("items", []):
                if isinstance(item, tuple) and len(item) > 0:
                    cmd = item[0]  # Command like "l" (line), "c" (curve), "re" (rect), etc.
                    item_info = {
                        "command": cmd,
                        "points": []
                    }
                    
                    # Extract points from the remaining tuple elements
                    for point_like in item[1:]:
                        if hasattr(point_like, 'x') and hasattr(point_like, 'y'):
                            # It's a Point object
                            item_info["points"].append([float(point_like.x), float(point_like.y)])
                        elif hasattr(point_like, 'x0'):
                            # It's a Rect object
                            item_info["rect"] = [float(point_like.x0), float(point_like.y0), 
                                               float(point_like.x1), float(point_like.y1)]
                        elif hasattr(point_like, 'ul'):
                            # It's a Quad object
                            item_info["quad"] = {
                                "ul": [float(point_like.ul.x), float(point_like.ul.y)],
                                "ur": [float(point_like.ur.x), float(point_like.ur.y)],
                                "ll": [float(point_like.ll.x), float(point_like.ll.y)],
                                "lr": [float(point_like.lr.x), float(point_like.lr.y)]
                            }
                        else:
                            # Fallback for other types
                            item_info["other_data"] = str(point_like)
                    
                    drawing_info["items"].append(item_info)
            
            drawings_info.append(drawing_info)
        
    except Exception as e:
        # If drawings extraction fails, return info about the error
        return [{"error": f"Could not extract drawings: {str(e)}"}]
    
    return drawings_info



def create_filtered_svg(svg_content: str) -> str:
    """Remove text and image elements from SVG, keeping only vector graphics"""
    # Remove text-related elements
    svg_content = re.sub(r'<path[^>]*id="font_[^"]*"[^>]*>[^<]*</path>', '', svg_content)
    svg_content = re.sub(r'<use[^>]*data-text[^>]*>', '', svg_content)
    svg_content = re.sub(r'<text[^>]*>.*?</text>', '', svg_content, flags=re.DOTALL)
    
    # Remove image elements
    svg_content = re.sub(r'<image[^>]*>.*?</image>', '', svg_content, flags=re.DOTALL)
    svg_content = re.sub(r'<image[^>]*/>', '', svg_content)
    
    # Clean up empty lines and excessive whitespace
    svg_content = re.sub(r'\n\s*\n', '\n', svg_content)
    svg_content = re.sub(r'^\s*$', '', svg_content, flags=re.MULTILINE)
    
    return svg_content

def extract_svg_from_page(page: pymupdf.Page, page_name: str, output_dir: str, svg_mode: str = "filtered") -> Dict[str, Any]:
    """Extract SVG representation of the page
    
    Args:
        svg_mode: "full", "filtered", "both", or "none"
    """
    try:
        svg_content = page.get_svg_image()
        result = {}
        
        # Always calculate filtered content for size comparison
        filtered_svg_content = create_filtered_svg(svg_content)
        
        # Save based on mode
        if svg_mode in ["full", "both"]:
            svg_filename = f"{page_name}.svg"
            svg_path = f"{output_dir}/{svg_filename}"
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg_content)
            
            result["full_svg"] = {
                "filename": svg_filename,
                "size_chars": len(svg_content),
                "contains_text": "<text" in svg_content or "data-text" in svg_content,
                "contains_images": "<image" in svg_content,
                "contains_paths": "<path" in svg_content,
                "saved": True
            }
        else:
            result["full_svg"] = {
                "filename": f"{page_name}.svg",
                "size_chars": len(svg_content),
                "contains_text": "<text" in svg_content or "data-text" in svg_content,
                "contains_images": "<image" in svg_content,
                "contains_paths": "<path" in svg_content,
                "saved": False
            }
        
        if svg_mode in ["filtered", "both"]:
            filtered_svg_filename = f"{page_name}_graphics_only.svg"
            filtered_svg_path = f"{output_dir}/{filtered_svg_filename}"
            with open(filtered_svg_path, "w", encoding="utf-8") as f:
                f.write(filtered_svg_content)
            
            result["graphics_only_svg"] = {
                "filename": filtered_svg_filename,
                "size_chars": len(filtered_svg_content),
                "size_reduction": f"{((len(svg_content) - len(filtered_svg_content)) / len(svg_content) * 100):.1f}%",
                "saved": True
            }
        else:
            result["graphics_only_svg"] = {
                "filename": f"{page_name}_graphics_only.svg",
                "size_chars": len(filtered_svg_content),
                "size_reduction": f"{((len(svg_content) - len(filtered_svg_content)) / len(svg_content) * 100):.1f}%",
                "saved": False
            }
        
        result["svg_mode"] = svg_mode
        return result
        
    except Exception as e:
        return {"error": str(e)}

def extract_svg_images(svg_content: str, page_name: str, output_dir: str) -> List[Dict[str, Any]]:
    """Extract base64 images from SVG content as fallback"""
    import base64
    svg_images = []
    
    # Find base64 encoded images in SVG
    image_pattern = r'<image[^>]*xlink:href="data:image/([^;]+);base64,([^"]+)"[^>]*>'
    matches = re.findall(image_pattern, svg_content)
    
    for idx, (image_format, base64_data) in enumerate(matches):
        try:
            # Decode base64 data
            image_data = base64.b64decode(base64_data)
            
            # Save the image
            image_filename = f"{page_name}_svg_image_{idx + 1}.{image_format}"
            image_path = f"{output_dir}/{image_filename}"
            
            with open(image_path, "wb") as f:
                f.write(image_data)
            
            # Extract additional info from SVG element
            element_pattern = rf'<image[^>]*xlink:href="data:image/{re.escape(image_format)};base64,{re.escape(base64_data[:50])}'
            element_match = re.search(element_pattern, svg_content)
            
            width = height = x = y = None
            if element_match:
                full_element = svg_content[element_match.start():svg_content.find('>', element_match.start()) + 1]
                width_match = re.search(r'width="(\d+)"', full_element)
                height_match = re.search(r'height="(\d+)"', full_element)
                x_match = re.search(r'x="(\d+)"', full_element)
                y_match = re.search(r'y="(\d+)"', full_element)
                
                width = int(width_match.group(1)) if width_match else None
                height = int(height_match.group(1)) if height_match else None
                x = int(x_match.group(1)) if x_match else None
                y = int(y_match.group(1)) if y_match else None
            
            svg_image_info = {
                "filename": image_filename,
                "format": image_format,
                "size_bytes": len(image_data),
                "source": "svg_base64",
                "svg_position": {"x": x, "y": y},
                "svg_dimensions": {"width": width, "height": height},
                "base64_preview": base64_data[:100] + "..." if len(base64_data) > 100 else base64_data
            }
            svg_images.append(svg_image_info)
            
        except Exception as e:
            print(f"    Error extracting SVG image {idx + 1}: {e}")
    
    return svg_images

def dump_page_data(doc: pymupdf.Document, output_dir: str = "page_dumps", extract_svg_images: bool = False, svg_mode: str = "filtered") -> None:
    """Dump page text data and visual content in various formats for inspection"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different content types
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/svg", exist_ok=True)
    
    for page_num in range(len(doc)):
        page: pymupdf.Page = doc[page_num]
        page_name: str = f"page_{page_num + 1}"
        
        print(f"Processing page {page_num + 1}...")
        
        # Dump raw text
        with open(f"{output_dir}/{page_name}_text.txt", "w", encoding="utf-8") as f:
            f.write(page.get_text())
        
        # Dump text blocks with bounding boxes, font and size info
        with open(f"{output_dir}/{page_name}_blocks.json", "w", encoding="utf-8") as f:
            # Get detailed text structure to extract font information
            text_dict = page.get_text("dict")
            
            # Each tuple: (x0, y0, x1, y1, text, block_no, block_type)
            blocks: List[Tuple[float, float, float, float, str, int, int]] = page.get_text("blocks")
            blocks_data: List[Dict[str, Union[str, List[float], int]]] = []
            
            for blk in blocks:
                x0, y0, x1, y1, text, block_no, block_type = blk
                
                # Extract font information for this block from the detailed structure
                fonts_in_block = set()
                sizes_in_block = set()
                
                # Find corresponding block in detailed structure
                for page_block in text_dict.get("blocks", []):
                    if page_block.get("number") == block_no and page_block.get("type") == 0:  # text block
                        for line in page_block.get("lines", []):
                            for span in line.get("spans", []):
                                if span.get("font"):
                                    fonts_in_block.add(span["font"])
                                if span.get("size"):
                                    sizes_in_block.add(round(span["size"], 1))
                
                # Convert sets to sorted lists for consistent output
                fonts_list = sorted(list(fonts_in_block)) if fonts_in_block else []
                sizes_list = sorted(list(sizes_in_block)) if sizes_in_block else []
                
                blocks_data.append({
                    "text": text.strip(),
                    "bbox": [x0, y0, x1, y1],
                    "block_no": block_no,
                    "fonts": fonts_list,
                    "font_sizes": sizes_list
                })
            json.dump(blocks_data, f, indent=2, ensure_ascii=False)
        
        # Dump detailed structure (with custom encoder for bytes)
        with open(f"{output_dir}/{page_name}_structure.json", "w", encoding="utf-8") as f:
            structure: Dict[str, Any] = page.get_text("dict")
            json.dump(structure, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        
        # Extract and save images
        images_info = extract_images_from_page(page, page_name, f"{output_dir}/images")
        if images_info:
            with open(f"{output_dir}/{page_name}_images.json", "w", encoding="utf-8") as f:
                json.dump(images_info, f, indent=2, ensure_ascii=False)
            print(f"  Found {len(images_info)} images")
        
        # Extract drawing/vector graphics info (full detail)
        try:
            drawings_info = extract_drawings_from_page(page)
            if drawings_info and not (len(drawings_info) == 1 and "error" in drawings_info[0]):
                with open(f"{output_dir}/{page_name}_drawings.json", "w", encoding="utf-8") as f:
                    json.dump(drawings_info, f, indent=2, ensure_ascii=False)
                print(f"  Found {len(drawings_info)} vector drawings")
            elif drawings_info and "error" in drawings_info[0]:
                print(f"  Drawings extraction skipped: {drawings_info[0]['error']}")
        except Exception as e:
            print(f"  Drawings extraction failed: {e}")
        

        # Extract SVG representation
        svg_info = extract_svg_from_page(page, page_name, f"{output_dir}/svg", svg_mode)
        
        # Extract images from SVG as fallback (for images PyMuPDF might miss)
        # Note: Usually these are background colors/gradients, disabled by default
        svg_images = []
        if extract_svg_images and "error" not in svg_info:
            svg_content = None
            try:
                with open(f"{output_dir}/svg/{page_name}.svg", "r", encoding="utf-8") as f:
                    svg_content = f.read()
                svg_images = extract_svg_images(svg_content, page_name, f"{output_dir}/images")
                if svg_images:
                    print(f"  Found {len(svg_images)} additional images in SVG (backgrounds/gradients)")
                    # Add SVG images to the images info
                    if images_info:
                        images_info.extend(svg_images)
                    else:
                        images_info = svg_images
                    # Update the images JSON file
                    with open(f"{output_dir}/{page_name}_images.json", "w", encoding="utf-8") as f:
                        json.dump(images_info, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  SVG image extraction failed: {e}")
        elif not extract_svg_images and "error" not in svg_info:
            # Still check for count but don't extract
            try:
                with open(f"{output_dir}/svg/{page_name}.svg", "r", encoding="utf-8") as f:
                    svg_content = f.read()
                svg_image_matches = re.findall(r'<image[^>]*xlink:href="data:image/([^;]+);base64,([^"]+)"[^>]*>', svg_content)
                if svg_image_matches:
                    print(f"  Skipped {len(svg_image_matches)} SVG images (backgrounds/gradients, use --extract-svg-images to include)")
            except:
                pass
        
        # Save SVG info with image counts
        svg_info["svg_images_found"] = len(svg_images)
        with open(f"{output_dir}/{page_name}_svg_info.json", "w", encoding="utf-8") as f:
            json.dump(svg_info, f, indent=2, ensure_ascii=False)
        
        if "error" not in svg_info:
            reduction = svg_info["graphics_only_svg"]["size_reduction"]
            saved_types = []
            if svg_info["full_svg"]["saved"]:
                saved_types.append("full")
            if svg_info["graphics_only_svg"]["saved"]:
                saved_types.append("graphics-only")

            if saved_types:
                types_str = " + ".join(saved_types)
                print(f"  Generated SVG ({types_str}, {reduction} size reduction)")
            else:
                print(f"  Analyzed SVG ({reduction} size reduction, no files saved)")
        
        print(f"Dumped page {page_num + 1} data to {output_dir}/")

def process_pdf(pdf_path: str, output_dir: str = "page_dumps", extract_svg_images: bool = False, svg_mode: str = "filtered") -> None:
    """Process a PDF file and dump all page data including images and graphics"""
    try:
        doc: pymupdf.Document = pymupdf.open(pdf_path)
        print(f"Processing PDF: {pdf_path}")
        print(f"Number of pages: {len(doc)}")
        extraction_note = " (excluding SVG backgrounds)" if not extract_svg_images else " (including SVG backgrounds)"
        svg_note = f" (SVG: {svg_mode})" if svg_mode != "none" else " (no SVG files)"
        print(f"Extracting: text, blocks, images{extraction_note}, drawings{svg_note}")
        print("-" * 60)
        
        dump_page_data(doc, output_dir, extract_svg_images, svg_mode)
        doc.close()
        
        print("-" * 60)
        print(f"Processing complete! Data dumped to '{output_dir}/' directory")
        print(f"Content organized in:")
        print(f"  • Text files: {output_dir}/page_N_text.txt")
        print(f"  • Block data: {output_dir}/page_N_blocks.json")
        print(f"  • Structure: {output_dir}/page_N_structure.json")
        print(f"  • Images: {output_dir}/images/")
        if svg_mode in ["full", "both"]:
            print(f"  • Full SVG: {output_dir}/svg/page_N.svg")
        if svg_mode in ["filtered", "both"]:
            print(f"  • Graphics-only SVG: {output_dir}/svg/page_N_graphics_only.svg")
        if svg_mode == "none":
            print(f"  • SVG analysis: {output_dir}/page_N_svg_info.json (no SVG files saved)")
        print(f"  • All vector drawings: {output_dir}/page_N_drawings.json")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)

def main() -> None:
    """Main function with command-line interface"""
    if len(sys.argv) < 2:
        print("Usage: python dump_pages.py <pdf_file> [output_directory] [options]")
        print("Example: python dump_pages.py document.pdf page_dumps")
        print("Example: python dump_pages.py document.pdf --svg-mode filtered")
        print("Example: python dump_pages.py document.pdf page_dumps --extract-svg-images --svg-mode full")
        print("\nThis script extracts:")
        print("  • Text content and positioning data")
        print("  • Images (saved as PNG files)")
        print("  • Vector drawings and graphics metadata")
        print("  • SVG representations (configurable)")
        print("\nOptions:")
        print("  --extract-svg-images     Also extract SVG embedded images (usually backgrounds/gradients)")
        print("  --svg-mode <mode>        SVG output mode:")
        print("                           'full'     - Save only full SVG (with text/images)")
        print("                           'filtered' - Save only graphics-only SVG (default)") 
        print("                           'both'     - Save both versions")
        print("                           'none'     - No SVG files (just analysis)")
        sys.exit(1)
    
    # Parse arguments
    pdf_path: str = sys.argv[1]
    output_dir: str = "page_dumps"
    extract_svg_images: bool = False
    svg_mode: str = "filtered"
    
    # Process remaining arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--extract-svg-images":
            extract_svg_images = True
        elif arg == "--svg-mode":
            if i + 1 < len(sys.argv):
                svg_mode = sys.argv[i + 1]
                if svg_mode not in ["full", "filtered", "both", "none"]:
                    print(f"Error: Invalid svg-mode '{svg_mode}'. Must be 'full', 'filtered', 'both', or 'none'")
                    sys.exit(1)
                i += 1  # Skip the next argument (the mode value)
            else:
                print("Error: --svg-mode requires a value")
                sys.exit(1)
        elif not arg.startswith("--"):
            output_dir = arg
        i += 1
    
    # Check if PDF file exists
    if not Path(pdf_path).exists():
        print(f"Error: PDF file '{pdf_path}' not found")
        sys.exit(1)
    
    process_pdf(pdf_path, output_dir, extract_svg_images, svg_mode)

if __name__ == "__main__":
    main()
