# TODO: Ignore text blocks that aren't visible (e.g., transparent, behind a vector image)

import pymupdf
import os
import tempfile
import re
import html
from typing import Any, Dict, List, Tuple, Union, cast, Literal, overload
from pathlib import Path
from models import TextBlock, BlocksDocument, Block


"""Type stub for PyMuPDF dynamically added method"""
class Page(pymupdf.Page):
    @overload
    def get_text(self, option: Literal["text"] = "text", **kwargs) -> str: ...

    @overload 
    def get_text(self, option: Literal["blocks"], **kwargs) -> List[Tuple[float, float, float, float, str, int, int]]: ...

    @overload
    def get_text(self, option: Literal["dict"], **kwargs) -> Dict[str, Any]: ...

    @overload
    def get_text(self, option: Literal["html"], **kwargs) -> str: ...

    def get_text(self, option: str = "text", **kwargs) -> Union[str, List[Tuple[float, float, float, float, str, int, int]], Dict[str, Any]]: ...


def _parse_html_elements(html_content: str) -> List[Dict[str, Any]]:
    """
    Parse HTML content to extract all elements with their positions and styling.
    Returns a list of elements with their text, styling, and bounding box info.
    """
    elements = []
    
    # Find all p tags with their content and style attributes
    p_pattern = r'<p\s+style="([^"]*)"[^>]*>(.*?)</p>'
    p_matches = re.findall(p_pattern, html_content, re.DOTALL)
    
    for style, content in p_matches:
        # Extract position from style
        position = _extract_position_from_style(style)
        if position:
            # Parse the content to preserve inner HTML structure
            cleaned_content = _clean_inner_html(content)
            if cleaned_content.strip():
                elements.append({
                    'type': 'p',
                    'content': cleaned_content,
                    'style': style,
                    'position': position,
                    'raw_html': f'<p style="{style}">{content}</p>'
                })
    
    return elements


def _extract_position_from_style(style: str) -> Dict[str, float] | None:
    """Extract top and left positions from CSS style string"""
    try:
        position = {}
        
        # Extract top position
        top_match = re.search(r'top:\s*([0-9.]+)pt', style)
        if top_match:
            position['top'] = float(top_match.group(1))
        
        # Extract left position  
        left_match = re.search(r'left:\s*([0-9.]+)pt', style)
        if left_match:
            position['left'] = float(left_match.group(1))
            
        # Extract line-height for height estimation
        height_match = re.search(r'line-height:\s*([0-9.]+)pt', style)
        if height_match:
            position['height'] = float(height_match.group(1))
        
        return position if 'top' in position and 'left' in position else None
    except:
        return None


def _clean_inner_html(content: str) -> str:
    """Clean inner HTML content while preserving structure"""
    # Remove excessive whitespace but preserve HTML tags
    content = re.sub(r'\s+', ' ', content)
    content = content.strip()
    return content


def _normalize_text_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching"""
    # Remove HTML tags for comparison
    text_only = re.sub(r'<[^>]+>', '', text)
    # Replace multiple whitespace with single space, strip, lowercase
    normalized = re.sub(r'\s+', ' ', text_only.strip().lower())
    return normalized


def _find_matching_html_elements(block_text: str, block_bbox: List[float], html_elements: List[Dict[str, Any]]) -> str:
    """
    Find HTML elements that match the given text block based on content and position.
    Returns the combined HTML content for the matching elements.
    """
    x0, y0, x1, y1 = block_bbox
    normalized_block_text = _normalize_text_for_matching(block_text)
    
    matching_elements = []
    
    for element in html_elements:
        # Check if element position overlaps with block bbox
        pos = element['position']
        elem_top = pos['top']
        elem_left = pos['left']
        elem_height = pos.get('height', 12)  # Default height if not specified
        
        # Simple overlap check - element should be within or near the block bounds
        if (elem_left >= x0 - 5 and elem_left <= x1 + 5 and  # Allow small margin
            elem_top >= y0 - 5 and elem_top <= y1 + elem_height + 5):
            
            # Also check if text content matches
            elem_text = _normalize_text_for_matching(element['content'])
            
            # If texts match or one contains the other, it's likely a match
            if (elem_text in normalized_block_text or 
                normalized_block_text in elem_text or
                _text_similarity(elem_text, normalized_block_text) > 0.8):
                matching_elements.append(element)
    
    if matching_elements:
        # Sort by position (top, then left) to maintain reading order
        matching_elements.sort(key=lambda e: (e['position']['top'], e['position']['left']))
        
        # Combine the HTML content
        combined_html = ''.join(elem['raw_html'] for elem in matching_elements)
        return combined_html
    
    # No matches found, return escaped plain text
    return html.escape(block_text)


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity between two strings"""
    if not text1 or not text2:
        return 0.0
    
    # Simple similarity based on common words
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def extract_text_blocks_with_styling(pdf_path: str, output_filename: str, temp_dir: str | None = None) -> str:
    """
    Extract text blocks with bounding boxes and complete HTML styling information from a PDF.
    
    This version parses the full page HTML and matches elements to text blocks based on 
    position and content, since the clip parameter doesn't work with HTML extraction.
    
    Args:
        pdf_path: Path to the PDF file to process
        output_filename: Full path to the output JSON file
        temp_dir: Directory to use for temporary files (optional, creates one if not provided)
    
    Returns:
        Path to the output JSON file
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If there's an error processing the PDF
    """
    # Check if PDF file exists
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file '{pdf_path}' not found")
    
    # Use provided temp_dir or create one
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="pdf_text_blocks_")
        output_path = os.path.join(temp_dir, output_filename)
        cleanup_temp = True
    else:
        output_path = output_filename  # Use full path as provided
        cleanup_temp = False
    
    try:
        # Open the PDF document
        doc: pymupdf.Document = pymupdf.open(pdf_path)
        
        # Container for all text blocks from all pages
        all_text_blocks: List[TextBlock] = []
        
        # Store total pages before processing
        total_pages = len(doc)
        
        # Process each page
        for page_num in range(total_pages):
            page: Page = cast(Page, doc[page_num])
            
            # Get full page HTML once
            page_html = page.get_text("html")
            
            # Parse HTML to extract all elements with positions
            html_elements = _parse_html_elements(page_html)
            
            # Get text blocks for semantic grouping
            blocks: List[Tuple[float, float, float, float, str, int, int]] = page.get_text("blocks", sort=True)
            
            # Process each text block and find matching HTML elements
            for blk in blocks:
                x0, y0, x1, y1, block_text, block_no, block_type = blk
                
                # Skip empty blocks and non-text blocks
                if not block_text.strip() or block_type != 0:
                    continue
                
                # Find HTML elements that match this text block
                styled_html = _find_matching_html_elements(
                    block_text.strip(), 
                    [x0, y0, x1, y1], 
                    html_elements
                )
                
                # Create text block using Pydantic model
                text_block = TextBlock(
                    page_number=page_num + 1,
                    text=styled_html,  # Matched HTML with complete styling
                    plain_text=block_text.strip(),  # Clean plain text from block
                    bbox=[x0, y0, x1, y1]
                )
                
                all_text_blocks.append(text_block)
        
        # Close the document
        doc.close()
        
        # Create output data structure using Pydantic model
        output_data = BlocksDocument(
            pdf_path=pdf_path,
            total_pages=total_pages,
            total_blocks=len(all_text_blocks),
            blocks=cast(List[Block], all_text_blocks)
        )
        
        # Write to JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_data.model_dump_json(indent=2, exclude_none=True))
        
        print(f"Extracted {len(all_text_blocks)} text blocks from {total_pages} pages")
        print(f"Output saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        # Clean up temp directory on error only if we created it
        if cleanup_temp:
            if os.path.exists(output_path):
                os.remove(output_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        raise Exception(f"Error processing PDF: {e}")


def main():
    """Example usage of the text block extraction function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: uv run extract_text_blocks.py <pdf_file> [output_filename]")
        print("Example: uv run extract_text_blocks.py document.pdf")
        print("Example: uv run extract_text_blocks.py document.pdf my_blocks.json")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else "text_blocks.json"
    
    # Create a real temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="text_blocks_test_")
    output_path = os.path.join(temp_dir, output_filename)
    
    try:
        result_path = extract_text_blocks_with_styling(pdf_path, output_path, temp_dir)
        print(f"Text blocks extracted successfully!")
        print(f"Output file: {result_path}")
        print(f"Temporary directory: {temp_dir}")
        print(f"Note: Clean up temporary directory when done: rm -rf {temp_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
