# TODO: Aggregate same-page bounding boxes for the same output node

from utils.models import ContentBlock, StructuredNode, BlockType
from bs4 import BeautifulSoup, Tag, NavigableString
from utils.schema import TagName, PositionalData, BoundingBox
from utils.range_parser import parse_range_string
from dataclasses import dataclass
from typing import Any, Iterable

ALLOWED_TAGS = [tag.value for tag in TagName] + ["b", "i", "u", "s", "sup", "sub", "br"]

INLINE_STYLE = {"b","i","u","s","sup","sub","br"}

@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    node: str | None = None  # e.g., "<section data-children='...'>"
    path: str | None = None  # optional CSS-like path

def parse_input_ids(input_html: str) -> set[int]:
    """Extract IDs from input HTML elements."""
    soup = BeautifulSoup(input_html, "html.parser")
    ids_in_input: set[int] = set()
    for element in soup.find_all():
        if not isinstance(element, Tag):
            continue
        id_attr = element.get("id")
        if isinstance(id_attr, str):
            try:
                ids_in_input.add(int(id_attr))
            except ValueError:
                pass
    return ids_in_input

def collect_output_ranges(output_soup: BeautifulSoup) -> dict[str, Any]:
    """Collect IDs and nodes from data-sources and data-children attributes."""
    ids_from_sources: set[int] = set()
    ids_from_children: set[int] = set()
    nodes_with_both: list[Tag] = []
    children_nodes: list[Tag] = []
    source_nodes: list[Tag] = []

    for element in output_soup.find_all():
        if not isinstance(element, Tag):
            continue

        data_sources = element.get("data-sources")
        data_children = element.get("data-children")

        has_sources = isinstance(data_sources, str)
        has_children_attr = isinstance(data_children, str)

        if has_sources:
            ids_from_sources.update(parse_range_string(data_sources))
            source_nodes.append(element)

        if has_children_attr:
            ids_from_children.update(parse_range_string(data_children))
            children_nodes.append(element)

        if has_sources and has_children_attr:
            nodes_with_both.append(element)

    return {
        "ids_from_sources": ids_from_sources,
        "ids_from_children": ids_from_children,
        "nodes_with_both": nodes_with_both,
        "children_nodes": children_nodes,
        "source_nodes": source_nodes,
    }

def validate_coverage(ids_in_input: set[int], ids_from_sources: set[int], ids_from_children: set[int]) -> list[ValidationIssue]:
    """Check if all input IDs are covered by data-sources or children."""
    missing_ids = ids_in_input - (ids_from_sources | ids_from_children)
    issues: list[ValidationIssue] = []
    for missing_id in missing_ids:
        issues.append(ValidationIssue(
            code="MISSING_COVERAGE",
            message=f"ID {missing_id} is not covered by data-sources or children.",
            node=f"<element id='{missing_id}' />"
        ))
    return issues

def validate_disjoint(ids_from_sources: set[int], ids_from_children: set[int]) -> list[ValidationIssue]:
    """Check if data-sources and children overlap."""
    overlapping_ids = ids_from_sources & ids_from_children
    issues: list[ValidationIssue] = []
    for overlapping_id in overlapping_ids:
        issues.append(ValidationIssue(
            code="DISJOINT_SOURCES_AND_CHILDREN",
            message=f"ID {overlapping_id} is present in both data-sources and children.",
            node=f"<element id='{overlapping_id}' />"
        ))
    return issues

def validate_mutual_exclusive_attrs(output_soup: BeautifulSoup) -> list[ValidationIssue]:
    """Check that no element has both data-sources and data-children attributes."""
    issues: list[ValidationIssue] = []
    for element in output_soup.find_all():
        if not isinstance(element, Tag):
            continue
        if isinstance(element.get("data-sources"), str) and isinstance(element.get("data-children"), str):
            issues.append(ValidationIssue(
                code="MUTUAL_EXCLUSIVE_ATTRS",
                message=f"Element must not have both data-sources and data-children: {element.name}",
                node=f"<{element.name} data-sources='...' data-children='...'>"
            ))
    return issues

def validate_children_empty(children_nodes: list[Tag]) -> list[ValidationIssue]:
    """Check that all nodes with data-children are empty containers (no text or child elements)."""
    issues: list[ValidationIssue] = []
    for child in children_nodes:
        has_text = bool(child.get_text(strip=True))
        has_elements = any(isinstance(c, Tag) for c in child.contents)
        if has_text or has_elements:
            issues.append(ValidationIssue(
                code="CHILDREN_MUST_BE_EMPTY",
                message=f"data-children node must be empty: {child.name}",
                node=f"<{child.name}>"
            ))
    return issues

def _has_meaningful_content(node: Tag) -> bool:
    """Determine if a node has meaningful content (text with non-whitespace or non-inline children)."""
    if node.string and node.string.strip():
        return True
    for child in node.children:
        if isinstance(child, NavigableString):
            if child.strip():
                return True
        elif isinstance(child, Tag) and child.name.lower() not in INLINE_STYLE:
            return True
    return False

def validate_sources_populated(source_nodes: list[Tag]) -> list[ValidationIssue]:
    """Check if source nodes have meaningful content or are images."""
    issues: list[ValidationIssue] = []
    for source in source_nodes:
        if not _has_meaningful_content(source):
            # Allow img tags to be empty if they are sources
            if source.name.lower() == "img":
                continue
            issues.append(ValidationIssue(
                code="SOURCES_EMPTY",
                message=f"Source node with no meaningful content: {source.name}",
                node=f"<{source.name}>"
            ))
    return issues

def validate_sources_and_children(input_html: str, output_html: str) -> tuple[bool, list[ValidationIssue], dict[str, Any]]:
    """Orchestrates validation of data-sources and children attributes."""
    in_ids = parse_input_ids(input_html)
    out_soup = BeautifulSoup(output_html, "html.parser")

    coll = collect_output_ranges(out_soup)
    issues: list[ValidationIssue] = []
    issues += validate_coverage(in_ids, coll["ids_from_sources"], coll["ids_from_children"])
    issues += validate_disjoint(coll["ids_from_sources"], coll["ids_from_children"])
    issues += validate_mutual_exclusive_attrs(out_soup)
    issues += validate_children_empty(coll["children_nodes"])
    issues += validate_sources_populated(coll["source_nodes"])

    is_valid = len(issues) == 0
    meta = {
        "input_id_count": len(in_ids),
        "covered_by_sources": len(coll["ids_from_sources"]),
        "covered_by_children": len(coll["ids_from_children"]),
    }
    return is_valid, issues, meta


def pretty_print_html(html: str) -> str:
    """Return a prettified HTML string using BeautifulSoup.

    Uses formatter="minimal" to avoid over-escaping inline content.
    Falls back to original string on parse errors.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        # prettify adds newlines and indentation
        return soup.prettify(formatter="minimal")
    except Exception:
        return html

def create_nodes_from_html(html: str, content_blocks: list[ContentBlock]) -> list[StructuredNode]:
    """Helper to create StructuredNodes from an HTML string
    
    HTML elements will have a `data-sources` attribute that is a comma-separated
    range of ContentBlock indices. This is used to map the positional data of the
    content blocks to the output StructuredNodes.

    Args:
        html: HTML to create nodes from
        content_blocks: List of content blocks to map to the HTML elements

    Returns:
        List of StructuredNodes
    """
    soup = BeautifulSoup(html, 'html.parser')
    nodes: list[StructuredNode] = []

    def _strip_control_chars(s: str) -> str:
        CONTROL_CHARS = "".join(chr(c) for c in range(32) if c not in (9, 10, 13)) + chr(127)
        TRANSLATOR = str.maketrans("", "", CONTROL_CHARS)
        return s.translate(TRANSLATOR)

    def _get_positional_data(indices: list[int]) -> list:
        """Get positional data from content blocks at given indices."""
        return [content_blocks[i].positional_data for i in indices if i < len(content_blocks)]
    
    def _aggregate_positional_data_by_page(pos_list: list[PositionalData]) -> list[PositionalData]:
        """Merge positional data that share the same page into a single bbox per page.

        For each unique page_pdf in pos_list, compute the bounding rectangle that
        encompasses all bboxes for that page using min(x1,y1) and max(x2,y2).
        """
        if not pos_list:
            return []

        by_page: dict[int, list[PositionalData]] = {}
        for pd in pos_list:
            by_page.setdefault(pd.page_pdf, []).append(pd)

        aggregated: list[PositionalData] = []
        for page_pdf, group in by_page.items():
            x1 = min(p.bbox.x1 for p in group)
            y1 = min(p.bbox.y1 for p in group)
            x2 = max(p.bbox.x2 for p in group)
            y2 = max(p.bbox.y2 for p in group)
            # Choose the first non-null logical page label if any
            page_logical = next((p.page_logical for p in group if p.page_logical is not None), None)
            aggregated.append(
                PositionalData(
                    page_pdf=page_pdf,
                    page_logical=page_logical,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        # Keep deterministic order by page number
        aggregated.sort(key=lambda p: p.page_pdf)
        return aggregated
    
    def _convert_element_to_node(element: Tag) -> StructuredNode:
        """Convert a BeautifulSoup Tag to a StructuredNode."""
        # Inline styling tags to treat as flat text
        inline_style_tags = {'b', 'i', 'u', 's', 'sup', 'sub', 'br'}
        
        # Get tag name and convert to TagName enum
        tag_name = TagName(element.name.lower())
        
        # Parse data-sources attribute
        data_sources_str = element.get('data-sources', '')
        if not isinstance(data_sources_str, str):
            raise ValueError(f"data-sources attribute is not a string: {data_sources_str}")
        source_indices = parse_range_string(data_sources_str)
        positional_data = _get_positional_data(source_indices)
        positional_data = _aggregate_positional_data_by_page(positional_data)
        
        # Handle img tags specially to extract storage_url, description, and caption
        if element.name.lower() == 'img':
            # Extract storage_url, description, and caption from source ContentBlocks
            storage_url = None
            description = None
            caption = None
            
            if source_indices:
                # Get the first valid source block for the image
                for idx in source_indices:
                    if idx < len(content_blocks) and content_blocks[idx].block_type == BlockType.PICTURE:
                        source_block = content_blocks[idx]
                        storage_url = source_block.storage_url
                        description = source_block.description
                        caption = source_block.caption
                        break
            
            return StructuredNode(
                tag=tag_name,
                children=[],
                text=None,
                positional_data=positional_data,
                storage_url=storage_url,
                description=description,
                caption=caption
            )
        
        # Get text content, flattening inline style tags
        text_content = None
        children = []
        
        # Check if this element has mixed content (text + inline styling)
        has_inline_styles = any(
            isinstance(child, Tag) and child.name.lower() in inline_style_tags
            for child in element.children
        )
        
        if has_inline_styles or (element.string and not any(isinstance(child, Tag) and child.name.lower() not in inline_style_tags for child in element.children)):
            # Flatten the content, preserving inline styling tags as HTML
            text_content = ""
            for child in element.children:
                if isinstance(child, NavigableString):
                    text_content += str(child)
                elif isinstance(child, Tag) and child.name.lower() in inline_style_tags:
                    # Preserve the inline styling tag as HTML
                    text_content += str(child)
                elif isinstance(child, Tag):
                    # This is a structural tag, process as child
                    children.append(_convert_element_to_node(child))
            text_content = _strip_control_chars(text_content.strip()) if text_content else None
        else:
            # Process children normally (no inline styling)
            for child in element.children:
                if isinstance(child, Tag):
                    children.append(_convert_element_to_node(child))
                elif isinstance(child, NavigableString) and child.strip():
                    if not text_content:
                        text_content = _strip_control_chars(child.strip())

        return StructuredNode(
            tag=tag_name,
            children=children,
            text=text_content,
            positional_data=positional_data
        )
    
    # Process all top-level elements
    for element in soup.children:
        if isinstance(element, Tag):
            nodes.append(_convert_element_to_node(element))
    
    return nodes


def validate_data_sources(input_html: str, output_html: str) -> tuple[set[int], set[int]]:
    # Parse HTML
    input_soup = BeautifulSoup(input_html, "html.parser")
    output_soup = BeautifulSoup(output_html, "html.parser")

    # Extract IDs from input
    ids_in_input: set[int] = set()
    for element in input_soup.find_all():
        if not isinstance(element, Tag):
            continue
        id_attr = element.get("id")
        if isinstance(id_attr, str):
            try:
                ids_in_input.add(int(id_attr))
            except ValueError:
                pass

    # Extract IDs from output data-sources attributes
    ids_in_output: set[int] = set()
    for element in output_soup.find_all():
        if not isinstance(element, Tag):
            continue
        data_sources = element.get("data-sources")
        if isinstance(data_sources, str):
            ids_in_output.update(parse_range_string(data_sources))

    # Check coverage
    missing_ids = ids_in_input - ids_in_output
    extra_ids = ids_in_output - ids_in_input

    # Always return the sets, regardless of whether they're empty or not
    return missing_ids, extra_ids


def validate_html_tags(html: str, exclude: Iterable[str] | None = None) -> tuple[bool, list[str]]:
    """Validate that HTML is well-formed and contains only allowed tags.
    
    Behavior:
    - If a <body> element exists, validate only the tags inside the body.
    - Otherwise, validate against the whole content (useful for HTML partials).
    
    Args:
        html: HTML string to validate (can be a full document or partial)
        exclude: Optional iterable of tag names to treat as disallowed even if
            they appear in the global allowed list (case-insensitive)
        
    Returns:
        Tuple of (is_valid, invalid_tags) where:
        - is_valid: True if HTML parses and all tags are valid
        - invalid_tags: List of disallowed tag names found in the validated scope
    """
    try:
        # Try to parse the HTML
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        # If parsing fails, HTML is not valid
        return False, []
    
    # Determine scope: prefer body descendants when body exists
    scope_elements: list[Tag] = []
    if soup.body is not None:
        scope_elements = [el for el in soup.body.find_all() if isinstance(el, Tag)]
    else:
        scope_elements = [el for el in soup.find_all() if isinstance(el, Tag)]
    
    # Build effective allowed set with optional exclusions
    allowed_set = {t.lower() for t in ALLOWED_TAGS}
    if exclude is not None:
        excluded = {t.lower() for t in exclude}
        allowed_set = allowed_set - excluded

    # Check for disallowed tags within the chosen scope
    disallowed_tags: set[str] = set()
    for element in scope_elements:
        if element.name and element.name.lower() not in allowed_set:
            disallowed_tags.add(element.name.lower())
    
    # Return validation result
    is_valid = len(disallowed_tags) == 0
    return is_valid, sorted(list(disallowed_tags))


def test_create_nodes_from_html_list_merging():
    """Test case for merging paragraph blocks into a list structure.
    
    This test demonstrates the scenario where multiple paragraph blocks with
    multiline text content are merged into a single <ul> element, with each
    line becoming a separate <li> item. The data-sources attributes track
    which original content blocks each element came from.
    """
    from utils.schema import PositionalData, EmbeddingSource, TagName, BoundingBox
    from utils.models import BlockType
    
    # Create input ContentBlocks - two paragraphs with multiline text
    content_blocks = [
        ContentBlock(
            positional_data=PositionalData(
                page_pdf=1,
                page_logical=1,
                bbox=BoundingBox(x1=100, y1=200, x2=500, y2=250)
            ),
            block_type=BlockType.TEXT,
            embedding_source=EmbeddingSource.TEXT_CONTENT,
            text_content="First item from block 0\nSecond item from block 0\nThird item from block 0"
        ),
        ContentBlock(
            positional_data=PositionalData(
                page_pdf=1,
                page_logical=1,
                bbox=BoundingBox(x1=100, y1=260, x2=500, y2=310)
            ),
            block_type=BlockType.TEXT,
            embedding_source=EmbeddingSource.TEXT_CONTENT,
            text_content="Fourth item from block 1\nFifth item from block 1"
        ),
        ContentBlock(
            positional_data=PositionalData(
                page_pdf=1,
                page_logical=1,
                bbox=BoundingBox(x1=100, y1=320, x2=500, y2=370)
            ),
            block_type=BlockType.PICTURE,
            embedding_source=EmbeddingSource.DESCRIPTION,
            description="A picture of a cat",
            storage_url="https://example.com/cat.jpg",
            caption="A picture of a cat"
        )
    ]
    
    # HTML string where the paragraphs are merged into a single ul
    # with data-sources tracking the original blocks, plus an img tag
    html = """<ul data-sources="0,1">
    <li data-sources="0">First item from <i>block 0</i></li>
    <li data-sources="0">Second item from block 0</li>
    <li data-sources="0">Third <b>item</b> from block 0</li>
    <li data-sources="1">Fourth item from block 1</li>
    <li data-sources="1">Fifth item from block <sup>1</sup></li>
</ul>
<img data-sources="2" src="https://example.com/cat.jpg" alt="A picture of a cat" />"""
    
    # Expected output: StructuredNodes representing the list structure and image
    expected_nodes = [
        StructuredNode(
            tag=TagName.UL,
            children=[
                StructuredNode(
                    tag=TagName.LI,
                    text="First item from <i>block 0</i>",
                    positional_data=[content_blocks[0].positional_data]
                ),
                StructuredNode(
                    tag=TagName.LI,
                    text="Second item from block 0",
                    positional_data=[content_blocks[0].positional_data]
                ),
                StructuredNode(
                    tag=TagName.LI,
                    text="Third <b>item</b> from block 0",
                    positional_data=[content_blocks[0].positional_data]
                ),
                StructuredNode(
                    tag=TagName.LI,
                    text="Fourth item from block 1",
                    positional_data=[content_blocks[1].positional_data]
                ),
                StructuredNode(
                    tag=TagName.LI,
                    text="Fifth item from block <sup>1</sup>",
                    positional_data=[content_blocks[1].positional_data]
                )
            ],
            positional_data=[
                PositionalData(
                    page_pdf=1,
                    page_logical=1,
                    bbox=BoundingBox(x1=100, y1=200, x2=500, y2=310),
                )
            ]
        ),
        StructuredNode(
            tag=TagName.IMG,
            storage_url="https://example.com/cat.jpg",
            description="A picture of a cat",
            caption="A picture of a cat",
            positional_data=[content_blocks[2].positional_data]
        )
    ]
    
    # Test the function
    result = create_nodes_from_html(html, content_blocks)

    # Assertions comparing result to expected structure
    assert len(result) == len(expected_nodes), f"Expected {len(expected_nodes)} nodes, got {len(result)}"
    
    # Test the UL element
    expected_ul = expected_nodes[0]
    result_ul = result[0]
    
    assert result_ul.tag == expected_ul.tag, f"Expected {expected_ul.tag} tag, got {result_ul.tag}"
    assert len(result_ul.children) == len(expected_ul.children), f"Expected {len(expected_ul.children)} children, got {len(result_ul.children)}"
    
    # Check each child LI element
    for i, (result_li, expected_li) in enumerate(zip(result_ul.children, expected_ul.children)):
        assert result_li.tag == expected_li.tag, f"Child {i} should be {expected_li.tag}, got {result_li.tag}"
        assert result_li.text == expected_li.text, f"Child {i} text mismatch: expected '{expected_li.text}', got '{result_li.text}'"
        assert result_li.positional_data == expected_li.positional_data, f"Child {i} positional data mismatch"
    
    # Check that positional data is correctly aggregated at the UL level
    assert result_ul.positional_data == expected_ul.positional_data, "UL positional data mismatch"
    
    # Test the IMG element
    expected_img = expected_nodes[1]
    result_img = result[1]
    
    assert result_img.tag == expected_img.tag, f"Expected {expected_img.tag} tag, got {result_img.tag}"
    assert result_img.storage_url == expected_img.storage_url, f"IMG storage_url mismatch: expected '{expected_img.storage_url}', got '{result_img.storage_url}'"
    assert result_img.description == expected_img.description, f"IMG description mismatch: expected '{expected_img.description}', got '{result_img.description}'"
    assert result_img.caption == expected_img.caption, f"IMG caption mismatch: expected '{expected_img.caption}', got '{result_img.caption}'"
    assert result_img.positional_data == expected_img.positional_data, "IMG positional data mismatch"


def test_validate_html_tags():
    """Test cases for the validate_html_tags function."""
    
    # Test valid HTML
    valid_html = "<p>Hello <b>world</b></p><ul><li>Item 1</li></ul>"
    is_valid, invalid_tags = validate_html_tags(valid_html)
    assert is_valid, "Valid HTML should pass validation"
    assert invalid_tags == [], f"Valid HTML should have no invalid tags, got {invalid_tags}"
    
    # Test invalid HTML with disallowed tags
    invalid_html = "<div><span>Hello</span><p>World</p></div>"
    is_valid, invalid_tags = validate_html_tags(invalid_html)
    assert not is_valid, "Invalid HTML should fail validation"
    assert set(invalid_tags) == {"div", "span"}, f"Expected ['div', 'span'], got {invalid_tags}"
    
    # Test empty HTML
    empty_html = ""
    is_valid, invalid_tags = validate_html_tags(empty_html)
    assert is_valid, "Empty HTML should be valid"
    assert invalid_tags == [], "Empty HTML should have no invalid tags"
    
    # Test HTML with only text (no tags)
    text_only = "Just some text"
    is_valid, invalid_tags = validate_html_tags(text_only)
    assert is_valid, "Text-only content should be valid"
    assert invalid_tags == [], "Text-only content should have no invalid tags"
    
    print("✅ validate_html_tags tests passed!")


if __name__ == "__main__":
    try:
        test_create_nodes_from_html_list_merging()
        print("✅ create_nodes_from_html test passed!")
        
        test_validate_html_tags()
        print("✅ validate_html_tags test passed!")

        # Ad hoc tests for sources/children validation
        flat_input = """
<p id="1">A</p>
<p id="2">B</p>
<p id="3">C</p>
""".strip()

        # Happy path: data-children covers 1-2, data-sources covers 3, disjoint, empty children nodes
        good_output = """
<section data-children="1-2"></section>
<p data-sources="3">C</p>
""".strip()
        ok, issues, meta = validate_sources_and_children(flat_input, good_output)
        assert ok, f"Expected valid, got issues: {[i.code for i in issues]}"
        print("✅ sources/children happy path passed!")

        # Violations: overlap ids, children not empty, node has both, sources empty
        bad_output = """
<section data-children="1-2">Not empty</section>
<div data-sources="2-3" data-children="1"></div>
<p data-sources="3"></p>
""".strip()
        ok2, issues2, meta2 = validate_sources_and_children(flat_input, bad_output)
        assert not ok2 and len(issues2) > 0, "Expected violations in bad output"
        print("✅ sources/children violation case surfaced issues:", ", ".join(sorted({i.code for i in issues2})))
        
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)