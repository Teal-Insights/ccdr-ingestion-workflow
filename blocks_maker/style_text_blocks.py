# TODO: In the event there are overlapping or redundant style tags in the result, we should remove them mechanically

import pymupdf
import re
from typing import Dict, List, Tuple

from utils.models import ContentBlock, BlockType
from utils.positioning import is_header_or_footer_by_position


def normalize_text_for_matching(text: str) -> str:
    """Normalize text for more robust matching between PyMuPDF and LayoutLM extractions."""
    import re
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize whitespace: collapse multiple spaces/newlines into single spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove common punctuation that might differ
    text = re.sub(r'[.,;:!?]$', '', text)
    
    return text


def find_best_span_match(span_text: str, content_text: str) -> tuple[bool, int]:
    """
    Find the best match for a span in content text using multiple strategies.
    
    Returns:
        (found_match, match_count) - whether a match was found and how many times
    """
    span_normalized = normalize_text_for_matching(span_text)
    content_normalized = normalize_text_for_matching(content_text)
    
    # Strategy 1: Exact normalized match
    if span_normalized in content_normalized:
        return True, content_normalized.count(span_normalized)
    
    # Strategy 2: Word-based fuzzy matching for multi-word spans
    span_words = span_normalized.split()
    if len(span_words) > 1:
        # Check if all words from the span appear in the content
        content_words = content_normalized.split()
        if all(word in content_words for word in span_words if len(word) > 2):  # Skip very short words
            # Check if words appear in sequence (with possible gaps)
            content_str = ' '.join(content_words)
            # Create a pattern that allows for gaps between words
            pattern = r'\b' + r'\b.*?\b'.join(re.escape(word) for word in span_words) + r'\b'
            matches = re.findall(pattern, content_str)
            if matches:
                return True, len(matches)
    
    # Strategy 3: Partial match for longer spans (at least 70% of words match)
    if len(span_words) >= 3:
        matching_words = sum(1 for word in span_words if word in content_normalized.split())
        if matching_words / len(span_words) >= 0.7:
            return True, 1  # Conservative count for partial matches
    
    return False, 0


def apply_style_flags_to_text(text: str, flags: int) -> str:
    """Apply PyMuPDF style flags to text by wrapping with HTML tags."""
    styled_text = text
    
    # Apply formatting based on flags (same logic as extract_text_blocks.py)
    if flags & 16:  # Bold
        styled_text = f"<b>{styled_text}</b>"
    if flags & 2:   # Italic
        styled_text = f"<i>{styled_text}</i>"
        
    return styled_text


def style_text_blocks(content_blocks: list[ContentBlock], pdf_path: str, temp_dir: str | None = None) -> list[ContentBlock]:
    """
    Style text blocks with pymupdf HTML by finding styled spans and applying them to ContentBlock text.
    
    Args:
        content_blocks: List of ContentBlocks to style
        pdf_path: Path to the PDF file
        temp_dir: Temporary directory (unused but kept for compatibility)
    
    Returns:
        List of ContentBlocks with styled text content
    """
    doc: pymupdf.Document = pymupdf.open(pdf_path)
    
    try:
        # Group content blocks by page for efficient processing
        blocks_by_page: Dict[int, List[Tuple[int, ContentBlock]]] = {}
        for idx, block in enumerate(content_blocks):
            page_num = block.positional_data.page_pdf
            if page_num not in blocks_by_page:
                blocks_by_page[page_num] = []
            blocks_by_page[page_num].append((idx, block))
        
        # Create result list - we'll build this by filtering and processing blocks
        styled_blocks: List[ContentBlock] = []
        total_styled_blocks = 0
        total_styled_spans_applied = 0
        excluded_count = 0
        
        # Process each page
        for page_num in range(1, len(doc) + 1):  # 1-indexed page numbers
            if page_num not in blocks_by_page:
                continue
                
            page = doc[page_num - 1]  # Convert to 0-indexed for PyMuPDF
            page_json = page.get_text("dict")
            page_height = page.rect.height
            
            # Extract styled spans from this page
            styled_spans_for_page: List[Tuple[str, str]] = []
            
            for json_block in page_json["blocks"]:
                if json_block["type"] == 0:  # Text block
                    for line in json_block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            flags = span.get("flags", 0)
                            
                            # Only consider spans with styling
                            if flags & (16 | 2) and text:  # Bold or Italic
                                styled_text = apply_style_flags_to_text(text, flags)
                                styled_spans_for_page.append((text, styled_text))
            
            if not styled_spans_for_page:
                # Even if no styled spans, we still need to process blocks for filtering
                pass
            else:
                print(f"Page {page_num}: Found {len(styled_spans_for_page)} styled spans")
            
            # Track which styled spans were successfully used
            used_spans = set()
            
            # Apply styling and filtering to ContentBlocks on this page
            for original_idx, content_block in blocks_by_page[page_num]:
                # Skip blocks that should be excluded by type
                if content_block.block_type in [BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER]:
                    excluded_count += 1
                    continue
                    
                # Apply position-based header/footer filtering
                if is_header_or_footer_by_position(content_block.positional_data.bbox, page_height):
                    excluded_count += 1
                    print(f"Heuristic exclusion: '{content_block.text_content[:50] if content_block.text_content else 'Empty'}...' on page {content_block.positional_data.page_pdf}")
                    continue
                
                # Keep picture blocks as-is
                if content_block.block_type == BlockType.PICTURE:
                    styled_blocks.append(content_block)
                    continue
                
                if not content_block.text_content:
                    # Keep blocks without text content
                    styled_blocks.append(content_block)
                    continue
                
                # Apply styled spans to this block's text
                original_text = content_block.text_content
                styled_text = original_text
                
                # Track which spans are used in this block
                spans_used_in_block = set()
                
                # Sort spans by length (longest first) to avoid partial replacements
                styled_spans_sorted = sorted(styled_spans_for_page, key=lambda x: len(x[0]), reverse=True)
                
                for original_span_text, styled_span_text in styled_spans_sorted:
                    if not original_span_text.strip():
                        continue
                    
                    # Use improved matching algorithm
                    found_match, match_count = find_best_span_match(original_span_text, styled_text)
                    
                    if found_match:
                        # For fuzzy matches, use the original regex approach as fallback
                        # Try exact replacement first
                        escaped_original = re.escape(original_span_text.strip())
                        exact_matches = re.findall(escaped_original, styled_text, re.IGNORECASE)
                        
                        if exact_matches:
                            # Exact match found - use regex replacement
                            # Escape backslashes in replacement to prevent backreference errors
                            escaped_replacement = styled_span_text.strip().replace('\\', '\\\\')
                            styled_text = re.sub(escaped_original, escaped_replacement, styled_text, flags=re.IGNORECASE)
                            spans_used_in_block.add(original_span_text)
                            used_spans.add(original_span_text)
                            
                            if len(exact_matches) > 1:
                                print(f"Warning: Multiple matches ({len(exact_matches)}) found for styled span: '{original_span_text.strip()}'")
                        else:
                            # Fuzzy match found but no exact replacement possible
                            # Mark as used to avoid warning, but don't modify text
                            spans_used_in_block.add(original_span_text)
                            used_spans.add(original_span_text)
                
                # Check if any styling was applied to this block
                if styled_text != original_text:
                    total_styled_blocks += 1
                    # Count number of style tags added
                    style_tags_added = styled_text.count('<b>') + styled_text.count('<i>')
                    total_styled_spans_applied += style_tags_added
                    
                    # Create new ContentBlock with styled text
                    styled_blocks.append(ContentBlock(
                        positional_data=content_block.positional_data,
                        block_type=content_block.block_type,
                        text_content=styled_text,
                        embedding_source=content_block.embedding_source,
                    ))
                else:
                    # No styling applied - keep original block
                    styled_blocks.append(content_block)
            
            # Warn about unused spans on this page (only once per span)
            if styled_spans_for_page:
                for original_span_text, _ in styled_spans_for_page:
                    if original_span_text not in used_spans:
                        print(f"Warning: No match found for styled span: '{original_span_text.strip()}'")
        
        # Discard empty text blocks as a sanity check
        final_styled_blocks = [
            block for block in styled_blocks if block.block_type == BlockType.PICTURE or 
            (block.text_content and block.text_content.strip())
        ]
        
        print("\nStyling summary:")
        print(f"  Total content blocks: {len(content_blocks)}")
        print(f"  Excluded blocks (headers/footers): {excluded_count}")
        print(f"  Blocks with styling applied: {total_styled_blocks}")
        print(f"  Total style spans applied: {total_styled_spans_applied}")
        print(f"  Final output blocks: {len(final_styled_blocks)}")

        return final_styled_blocks
        
    finally:
        doc.close()


if __name__ == "__main__":
    import os
    import json

    pdf_path: str = "./artifacts/wkdir/doc_601.pdf"
    temp_dir: str = "./artifacts"

    with open(os.path.join("artifacts", "doc_601_content_blocks_with_descriptions.json"), "r") as fr:
        content_blocks: list[ContentBlock] = json.load(fr)
        content_blocks = [ContentBlock.model_validate(block) for block in content_blocks]
    
    print(f"Loaded {len(content_blocks)} content blocks before styling")
    
    content_blocks_with_styles: list[ContentBlock] = style_text_blocks(content_blocks, pdf_path, temp_dir)
    print(f"Styled {len(content_blocks_with_styles)} content blocks")
    
    with open(os.path.join("artifacts", "doc_601_content_blocks_with_styles.json"), "w") as fw:
        json.dump([block.model_dump() for block in content_blocks_with_styles], fw, indent=2)