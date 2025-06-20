# TODO: Ignore text blocks that aren't visible (e.g., transparent, behind a vector image)

import pymupdf
import json
import os
import tempfile
import re
import html
from typing import Any, Dict, List, Tuple, Union, cast, Literal, overload
from pathlib import Path


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


def _extract_spans_from_html(html_content: str) -> List[Dict[str, str]]:
    """Extract all span elements from HTML with their text and styling"""
    spans = []
    # Find all span tags with their content and style attributes
    span_pattern = r'<span([^>]*)>(.*?)</span>'
    matches = re.findall(span_pattern, html_content, re.DOTALL)
    
    for attrs, text_content in matches:
        # Extract style attribute
        style_match = re.search(r'style="([^"]*)"', attrs)
        style = style_match.group(1) if style_match else ""
        
        # Decode HTML entities in text
        clean_text = html.unescape(text_content.strip())
        
        if clean_text:  # Skip empty spans
            spans.append({
                "text": clean_text,
                "style": style,
                "raw_html": f'<span{attrs}>{text_content}</span>'
            })
    
    return spans


def _normalize_text_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching"""
    # Replace multiple whitespace with single space, strip, lowercase
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    # Remove common problematic characters
    normalized = re.sub(r'[\n\r\t\b]', ' ', normalized)
    return re.sub(r'\s+', ' ', normalized).strip()


def _match_block_to_spans(block_text: str, spans: List[Dict[str, str]]) -> str:
    """Match a text block to HTML spans and reconstruct styled HTML"""
    if not spans:
        return html.escape(block_text)
    
    # Normalize the block text for matching
    normalized_block = _normalize_text_for_matching(block_text)
    
    # Build a list of all span texts concatenated to see if it matches
    all_spans_text = ' '.join(span["text"] for span in spans)
    normalized_spans = _normalize_text_for_matching(all_spans_text)
    
    # If the normalized texts are very similar, use the spans
    if normalized_block in normalized_spans or normalized_spans in normalized_block:
        # Try to match spans in order
        result_html = ""
        remaining_text = block_text
        
        for span in spans:
            span_text = span["text"]
            normalized_span = _normalize_text_for_matching(span_text)
            normalized_remaining = _normalize_text_for_matching(remaining_text)
            
            # Check if this span's text appears at the start of remaining text
            if normalized_remaining.startswith(normalized_span):
                # Add this span to result
                result_html += f'<span style="{span["style"]}">{html.escape(span_text)}</span>'
                
                # Remove the matched text from remaining (roughly)
                # This is approximate - we find where this text appears and remove it
                span_len = len(span_text)
                # Find the actual position in the original text (accounting for normalization differences)
                pos = 0
                for i in range(min(len(remaining_text), span_len * 2)):
                    if _normalize_text_for_matching(remaining_text[i:i+span_len]) == normalized_span:
                        pos = i + span_len
                        break
                if pos > 0:
                    remaining_text = remaining_text[pos:]
                else:
                    # Fallback: remove approximately the same amount
                    remaining_text = remaining_text[span_len:]
        
        # If we have leftover text, add it without styling
        if remaining_text.strip():
            result_html += html.escape(remaining_text.strip())
        
        return result_html if result_html else html.escape(block_text)
    else:
        # No good match found, return plain escaped text
        return html.escape(block_text)


def extract_text_blocks_with_styling(pdf_path: str, output_filename: str, temp_dir: str | None = None) -> str:
    """
    Extract text blocks with bounding boxes and font styling information from a PDF.
    
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
        all_text_blocks: List[Dict[str, Any]] = []
        
        # Store total pages before processing
        total_pages = len(doc)
        
        # Process each page
        for page_num in range(total_pages):
            page: Page = cast(Page, doc[page_num])
            
            # Get HTML representation of the page
            html_content = page.get_text("html")
            
            # Extract all spans from HTML with their styling
            spans = _extract_spans_from_html(html_content)
            
            # Get text blocks for semantic grouping
            blocks: List[Tuple[float, float, float, float, str, int, int]] = page.get_text("blocks", sort=True)
            
            # Process each text block and match it to HTML spans
            for blk in blocks:
                x0, y0, x1, y1, block_text, block_no, block_type = blk
                
                # Skip empty blocks and non-text blocks
                if not block_text.strip() or block_type != 0:
                    continue
                
                # Try to match this block's text to spans and reconstruct styled HTML
                styled_html = _match_block_to_spans(block_text.strip(), spans)
                
                # Create text block data structure with 5 keys
                text_block = {
                    "block_type": "text",
                    "page_number": page_num + 1,
                    "text": styled_html,  # Reconstructed HTML with semantic grouping
                    "plain_text": block_text.strip(),  # Clean plain text from block
                    "bbox": [x0, y0, x1, y1]
                }
                
                all_text_blocks.append(text_block)
        
        # Close the document
        doc.close()
        
        # Create output data structure
        output_data = {
            "pdf_path": pdf_path,
            "total_pages": total_pages,
            "total_text_blocks": len(all_text_blocks),
            "text_blocks": all_text_blocks
        }
        
        # Write to JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
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
        print("Usage: python extract_text_blocks.py <pdf_file> [output_filename]")
        print("Example: python extract_text_blocks.py document.pdf")
        print("Example: python extract_text_blocks.py document.pdf my_blocks.json")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else "text_blocks.json"
    
    try:
        output_path = extract_text_blocks_with_styling(pdf_path, output_filename)
        print(f"Text blocks extracted successfully!")
        print(f"Output file: {output_path}")
        print(f"Temporary directory: {os.path.dirname(output_path)}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
