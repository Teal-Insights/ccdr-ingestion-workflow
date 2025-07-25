#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import pymupdf

from transform.models import ContentBlock, BlockType, PositionalData
from utils.schema import EmbeddingSource

def is_header_or_footer_by_position(bbox: dict[str, int], page_height: float) -> bool:
    """
    Determine if a block is a header or footer based on position.
    Assumes 8.5 x 11 inch page, excludes top and bottom inch.
    
    Args:
        bbox: Bounding box with x1, y1, x2, y2 coordinates
        page_height: Height of the page in points
    
    Returns:
        True if the block is entirely contained in header/footer area
    """
    # Convert inches to points (1 inch = 72 points)
    inch_in_points = 72
    
    # Header area: top inch (0 to 72 points from top)
    header_bottom = inch_in_points
    
    # Footer area: bottom inch (page_height - 72 to page_height)
    footer_top = page_height - inch_in_points
    
    # Check if block is entirely contained in header area
    if bbox['y2'] <= header_bottom:
        return True
    
    # Check if block is entirely contained in footer area  
    if bbox['y1'] >= footer_top:
        return True
    
    return False

def find_text_matches_for_blocks(content_blocks: list[ContentBlock], doc: pymupdf.Document) -> dict[int, tuple]:
    """
    Find text matches for content blocks, returning a mapping of original indices to matches.
    """
    
    # Build a lookup of all text spans across all pages
    all_spans = {}  # (page_num, block_idx, line_idx, span_idx) -> span_data
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_json = page.get_text("dict")
        page_height = page.rect.height
        
        for block_idx, json_block in enumerate(page_json["blocks"]):
            if json_block["type"] == 0:  # Text block
                for line_idx, line in enumerate(json_block["lines"]):
                    for span_idx, span in enumerate(line["spans"]):
                        # Filter PyMuPDF spans that are in header/footer areas
                        span_bbox = {
                            'x1': int(span["bbox"][0]),
                            'y1': int(span["bbox"][1]), 
                            'x2': int(span["bbox"][2]),
                            'y2': int(span["bbox"][3])
                        }
                        
                        if is_header_or_footer_by_position(span_bbox, page_height):
                            continue  # Skip header/footer spans
                        
                        span_key = (page_num, block_idx, line_idx, span_idx)
                        span_data = {
                            'text': span["text"].strip(),
                            'bbox': span["bbox"],
                            'full_span': span,
                            'line': line
                        }
                        all_spans[span_key] = span_data
    
    print(f"Built lookup with {len(all_spans)} non-header/footer spans across {len(doc)} pages")
    
    # Find matches for each content block (keeping original indices)
    matches = {}  # original_index -> (page_num, span_data)
    
    for original_idx, content_block in enumerate(content_blocks):
        # Skip blocks that should be excluded
        if content_block.block_type in [BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER, BlockType.PICTURE]:
            continue
            
        # Apply heuristic filtering
        page_num = content_block.positional_data.page_pdf - 1
        if 0 <= page_num < len(doc):
            page = doc[page_num]
            page_height = page.rect.height
            
            if is_header_or_footer_by_position(content_block.positional_data.bbox, page_height):
                continue  # Skip header/footer blocks
        
        cb_text = (content_block.text_content or "").strip()
        if not cb_text:
            continue
        
        cb_bbox = content_block.positional_data.bbox
        cb_page = content_block.positional_data.page_pdf
        
        # Try to find a match (same logic as before)
        expected_page = cb_page - 1  # Convert to 0-indexed
        found_match = False
        
        # First, try exact text match on the expected page
        if 0 <= expected_page < len(doc):
            for span_key, span_data in all_spans.items():
                page_num, _, _, _ = span_key
                if page_num == expected_page and span_data['text']:
                    # Check for exact match or substring match
                    if (cb_text.lower() == span_data['text'].lower() or 
                        cb_text.lower() in span_data['text'].lower() or
                        span_data['text'].lower() in cb_text.lower()):
                        
                        # Verify coordinate proximity
                        span_bbox = span_data['bbox']
                        y_diff = abs(cb_bbox['y1'] - span_bbox[1])
                        
                        if y_diff < 50:  # Y-coordinates should be close
                            matches[original_idx] = (page_num, span_data)
                            found_match = True
                            break
        
        # If not found on expected page, search all pages
        if not found_match:
            best_match = None
            best_score = float('inf')
            
            for span_key, span_data in all_spans.items():
                page_num, _, _, _ = span_key
                if not span_data['text']:
                    continue
                
                # Text similarity score
                text_score = 0
                if cb_text.lower() == span_data['text'].lower():
                    text_score = 0  # Perfect match
                elif cb_text.lower() in span_data['text'].lower() or span_data['text'].lower() in cb_text.lower():
                    text_score = 1  # Substring match
                else:
                    # Word-based similarity for multi-word texts
                    cb_words = set(cb_text.lower().split())
                    span_words = set(span_data['text'].lower().split())
                    if cb_words and span_words:
                        intersection = cb_words & span_words
                        union = cb_words | span_words
                        if len(intersection) > 0:
                            text_score = 2 + (1 - len(intersection) / len(union))  # Jaccard-based
                        else:
                            text_score = 10  # No word overlap
                    else:
                        text_score = 10
                
                if text_score < 3:  # Only consider reasonable text matches
                    # Coordinate score (y-coordinate similarity is most important)
                    span_bbox = span_data['bbox']
                    y_diff = abs(cb_bbox['y1'] - span_bbox[1])
                    coord_score = y_diff / 100.0  # Normalize
                    
                    # Page penalty (prefer correct page)
                    page_penalty = 0 if page_num == expected_page else 0.5
                    
                    total_score = text_score + coord_score + page_penalty
                    
                    if total_score < best_score:
                        best_score = total_score
                        best_match = (page_num, span_data)
            
            if best_match and best_score < 5:  # Reasonable threshold
                matches[original_idx] = best_match
    
    return matches

def extract_text_blocks_with_heuristic_filtering(
    content_blocks: list[ContentBlock], pdf_path: str, temp_dir: str | None = None
) -> list[ContentBlock]:
    """
    Text extraction with heuristic header/footer filtering that preserves original block order.
    """
    
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file '{pdf_path}' not found")

    doc: pymupdf.Document = pymupdf.open(pdf_path)
    
    # Find matches while preserving original indices
    matches = find_text_matches_for_blocks(content_blocks, doc)
    
    # Count statistics
    filtered_blocks_count = 0
    excluded_count = 0
    
    for original_idx, content_block in enumerate(content_blocks):
        if content_block.block_type in [BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER]:
            excluded_count += 1
            continue
        elif content_block.block_type == BlockType.PICTURE:
            continue  # Pictures are kept as-is
        else:
            # Apply heuristic filtering
            page_num = content_block.positional_data.page_pdf - 1
            if 0 <= page_num < len(doc):
                page = doc[page_num]
                page_height = page.rect.height
                
                if is_header_or_footer_by_position(content_block.positional_data.bbox, page_height):
                    excluded_count += 1
                    print(f"Heuristic exclusion: '{content_block.text_content[:50] if content_block.text_content else 'Empty'}...' on page {content_block.positional_data.page_pdf}")
                    continue
            
            filtered_blocks_count += 1
    
    print(f"Excluded {excluded_count} blocks using heuristic header/footer filtering")
    print(f"Found matches for {len(matches)} out of {filtered_blocks_count} filtered text blocks")
    
    # Process results IN ORIGINAL ORDER
    modified_content_blocks: list[ContentBlock] = []
    matched_blocks = 0
    
    for original_idx, content_block in enumerate(content_blocks):
        # Handle different block types while preserving order
        if content_block.block_type == BlockType.PICTURE:
            # Keep picture blocks as-is in their original position
            modified_content_blocks.append(content_block)
            
        elif content_block.block_type in [BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER]:
            # Skip header/footer blocks (excluded from output)
            continue
            
        else:
            # Apply heuristic filtering
            page_num = content_block.positional_data.page_pdf - 1
            if 0 <= page_num < len(doc):
                page = doc[page_num]
                page_height = page.rect.height
                
                if is_header_or_footer_by_position(content_block.positional_data.bbox, page_height):
                    # Skip heuristically filtered blocks
                    continue
            
            # Check if this block has a match
            if original_idx in matches:
                matched_blocks += 1
                page_num, span_data = matches[original_idx]
                
                # Create HTML-like text content from span
                span = span_data['full_span']
                html_text = span["text"]
                flags = span.get("flags", 0)
                
                # Apply formatting based on flags
                if flags & 16:  # Bold
                    html_text = f"<b>{html_text}</b>"
                if flags & 2:   # Italic
                    html_text = f"<i>{html_text}</i>"
                
                # Create new content block with matched text IN ORIGINAL POSITION
                modified_content_blocks.append(
                    ContentBlock(
                        positional_data=PositionalData(
                            page_pdf=content_block.positional_data.page_pdf,
                            page_logical=content_block.positional_data.page_logical,
                            bbox=content_block.positional_data.bbox,
                        ),
                        block_type=content_block.block_type,
                        text_content=html_text.strip(),
                        embedding_source=EmbeddingSource.TEXT_CONTENT,
                    )
                )
            else:
                # No match found - skip this block but preserve order for others
                print(f"Warning: No match found for filtered content block: '{content_block.text_content[:50] if content_block.text_content else 'Empty'}...'")
    
    doc.close()
    
    print("\nFinal statistics:")
    print(f"  Original content blocks: {len(content_blocks)}")
    print(f"  Filtered blocks processed: {filtered_blocks_count}")
    print(f"  Successfully matched: {matched_blocks}")
    print(f"  Final output blocks: {len(modified_content_blocks)}")
    if filtered_blocks_count > 0:
        match_rate = (matched_blocks / filtered_blocks_count * 100)
        print(f"  Match rate: {match_rate:.1f}%")
    
    return modified_content_blocks


if __name__ == "__main__":
    import os
    import json

    pdf_path: str = "./artifacts/wkdir/doc_601.pdf"
    temp_dir: str = "./artifacts"

    with open(os.path.join("artifacts", "doc_601_content_blocks_with_descriptions.json"), "r") as fr:
        content_blocks: list[ContentBlock] = json.load(fr)
        content_blocks = [ContentBlock.model_validate(block) for block in content_blocks]
    
    print(f"Loaded {len(content_blocks)} content blocks before text extraction")
    
    # Count blocks by type
    type_counts = defaultdict(int)
    for block in content_blocks:
        type_counts[block.block_type] += 1
    
    print("Block type distribution:")
    for block_type, count in type_counts.items():
        print(f"  {block_type}: {count}")
    
    # Run on full PDF
    content_blocks_subset = content_blocks
    print(f"\nProcessing full PDF with {len(content_blocks_subset)} blocks")
    
    content_blocks_with_text: list[ContentBlock] = extract_text_blocks_with_heuristic_filtering(content_blocks_subset, pdf_path, temp_dir)
    print(f"Extracted text for {len(content_blocks_with_text)} content blocks")
    
    with open(os.path.join("artifacts", "doc_601_content_blocks_with_text.json"), "w") as fw:
        json.dump([block.model_dump() for block in content_blocks_with_text], fw, indent=2)