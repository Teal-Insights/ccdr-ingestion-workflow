import pymupdf
import json
from Levenshtein import distance

def find_text_in_blocks(page, search_text, threshold=0.2):
    """Find text in page blocks, handling rotated text and using fuzzy matching"""
    blocks = page.get_text("dict")
    found_rects = []
    
    for block in blocks["blocks"]:
        if "lines" in block:  # text block
            block_text = ""
            # Collect all text from the block
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                block_text += line_text + "\n"
            
            # Calculate normalized Levenshtein distance
            block_text = block_text.strip()
            if not block_text:
                continue
                
            # Calculate similarity ratio (0 to 1, where 1 is identical)
            max_len = max(len(search_text), len(block_text))
            if max_len == 0:
                continue
                
            similarity = 1 - (distance(search_text.lower(), block_text.lower()) / max_len)
            
            # If similarity is above threshold, consider it a match
            if similarity >= (1 - threshold):
                bbox = block["bbox"]
                rect = pymupdf.Rect(bbox)
                found_rects.append((rect, similarity))
    
    # Sort by similarity score and return the best match
    if found_rects:
        found_rects.sort(key=lambda x: x[1], reverse=True)
        return [found_rects[0][0]]
    return []

def extract_bounding_boxes(pdf_path, content_nodes_path, similarity_threshold=0.2):
    # Load the PDF
    doc = pymupdf.open(pdf_path)

    # Load content nodes
    with open(content_nodes_path, 'r') as f:
        nodes = json.load(f)

    for node in nodes:
        # Skip if content is None or empty string
        if not node.get("content") or not node["content"].strip():
            continue
            
        # Skip if bounding box already exists
        if (node.get("positional_data") and 
            len(node["positional_data"]) > 0 and 
            node["positional_data"][0].get("bounding_box")):
            continue

        page_num = node["positional_data"][0]["page_pdf"] - 1  # Convert to 0-based
        page = doc[page_num]
        search_text = node["content"].strip()

        # 1. Search whole text with fuzzy matching
        text_instances = page.search_for(search_text)
        if text_instances:
            bbox = text_instances[0]  # Take first match
            node["positional_data"][0]["bounding_box"] = {
                "x0": round(bbox.x0, 2),
                "y0": round(bbox.y0, 2),
                "x1": round(bbox.x1, 2), 
                "y1": round(bbox.y1, 2)
            }
            continue

        # 2. Search whole text in blocks with fuzzy matching
        block_rects = find_text_in_blocks(page, search_text, similarity_threshold)
        if block_rects:
            bbox = block_rects[0]
            node["positional_data"][0]["bounding_box"] = {
                "x0": round(bbox.x0, 2),
                "y0": round(bbox.y0, 2),
                "x1": round(bbox.x1, 2), 
                "y1": round(bbox.y1, 2)
            }
            print(f"Found fuzzy match in block: {search_text[:50]}...")
            continue

        # 3. Search lines (split by newlines) with fuzzy matching
        lines = [line.strip() for line in search_text.split('\n') if line.strip()]
        found_bboxes = []
        
        for line in lines:
            line_instances = page.search_for(line)
            if line_instances:
                # Take the first match for each line
                bbox = line_instances[0]
                found_bboxes.append({
                    "page_pdf": page_num + 1,  # Convert back to 1-based
                    "page_logical": None,
                    "bounding_box": {
                        "x0": round(bbox.x0, 2),
                        "y0": round(bbox.y0, 2),
                        "x1": round(bbox.x1, 2), 
                        "y1": round(bbox.y1, 2)
                    }
                })

        if found_bboxes and len(found_bboxes) == len(lines):
            # Replace the positional_data with all found bboxes
            node["positional_data"] = found_bboxes
            print(f"Found {len(found_bboxes)} bboxes for multi-line text: {search_text[:50]}...")
            continue

        # 4. Search lines in blocks with fuzzy matching
        found_bboxes = []
        for line in lines:
            block_rects = find_text_in_blocks(page, line, similarity_threshold)
            if block_rects:
                bbox = block_rects[0]  # Take first match
                found_bboxes.append({
                    "page_pdf": page_num + 1,
                    "page_logical": None,
                    "bounding_box": {
                        "x0": round(bbox.x0, 2),
                        "y0": round(bbox.y0, 2),
                        "x1": round(bbox.x1, 2), 
                        "y1": round(bbox.y1, 2)
                    }
                })

        if found_bboxes:
            node["positional_data"] = found_bboxes
            print(f"Found {len(found_bboxes)} fuzzy rotated/transformed line bboxes: {search_text[:50]}...")
            continue

        # 5. Print warning if nothing worked
        print(f"WARNING: Could not find any matches for: {search_text[:50]}...")

    doc.close()
    return nodes


if __name__ == "__main__":
    pdf_path = "input.pdf"
    content_nodes_path = "sample_data/content_nodes.json"
    # You can adjust the similarity threshold here (0.2 = 80% similarity required)
    nodes = extract_bounding_boxes(pdf_path, content_nodes_path, similarity_threshold=0.2)
    with open("content_nodes_with_bbox.json", "w") as f:
        json.dump(nodes, f, indent=2)