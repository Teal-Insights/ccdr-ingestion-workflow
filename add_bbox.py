import pymupdf
import json

def find_text_in_blocks(page, search_text):
    """Find text in page blocks, handling rotated text"""
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
            
            # Check if our search text is in this block
            if search_text.lower() in block_text.lower():
                # Return the block's bounding box
                bbox = block["bbox"]
                rect = pymupdf.Rect(bbox)
                found_rects.append(rect)
    
    return found_rects

def extract_bounding_boxes(pdf_path, content_nodes_path):
    # Load the PDF
    doc = pymupdf.open(pdf_path)

    # Load content nodes
    with open(content_nodes_path, 'r') as f:
        nodes = json.load(f)

    for node in nodes:
        page_num = node["positional_data"][0]["page_pdf"] - 1  # Convert to 0-based
        page = doc[page_num]
        search_text = node["content"].strip()

        # 1. Search whole text
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

        # 2. Search whole text in blocks (handles rotated text)
        block_rects = find_text_in_blocks(page, search_text)
        if block_rects:
            bbox = block_rects[0]
            node["positional_data"][0]["bounding_box"] = {
                "x0": round(bbox.x0, 2),
                "y0": round(bbox.y0, 2),
                "x1": round(bbox.x1, 2), 
                "y1": round(bbox.y1, 2)
            }
            print(f"Found whole text in block: {search_text[:50]}...")
            continue

        # 3. Search lines (split by newlines)
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

        # 4. Search lines in blocks (for rotated individual lines)
        found_bboxes = []
        for line in lines:
            block_rects = find_text_in_blocks(page, line)
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
            print(f"Found {len(found_bboxes)} rotated/transformed line bboxes: {search_text[:50]}...")
            continue

        # 5. Print warning if nothing worked
        print(f"WARNING: Could not find any matches for: {search_text[:50]}...")

    doc.close()
    return nodes


if __name__ == "__main__":
    pdf_path = "output.pdf"
    content_nodes_path = "content_nodes.json"
    nodes = extract_bounding_boxes(pdf_path, content_nodes_path)
    with open("content_nodes_with_bbox.json", "w") as f:
        json.dump(nodes, f, indent=2)