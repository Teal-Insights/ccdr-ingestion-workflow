from transform.models import ContentBlock, StructuredNode
from transform.detect_top_level_structure import parse_range_string
from bs4 import BeautifulSoup, Tag, NavigableString
from utils.schema import TagName

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
    

    
    def _get_positional_data(indices: list[int]) -> list:
        """Get positional data from content blocks at given indices."""
        return [content_blocks[i].positional_data for i in indices if i < len(content_blocks)]
    
    def _convert_element_to_node(element: Tag) -> StructuredNode:
        """Convert a BeautifulSoup Tag to a StructuredNode."""
        # Get tag name and convert to TagName enum
        tag_name = TagName(element.name.lower())
        
        # Parse data-sources attribute
        data_sources_str = element.get('data-sources', '')
        source_indices = parse_range_string(data_sources_str)
        positional_data = _get_positional_data(source_indices)
        
        # Get text content (only direct text, not from children)
        text_content = None
        if element.string:
            text_content = element.string.strip()
        elif len(element.contents) == 1 and isinstance(element.contents[0], NavigableString):
            text_content = str(element.contents[0]).strip()
        
        # Process children
        children = []
        for child in element.children:
            if isinstance(child, Tag):
                children.append(_convert_element_to_node(child))
            elif isinstance(child, NavigableString) and child.strip():
                # Skip if we already captured this as text_content
                if not text_content:
                    text_content = child.strip()
        
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


def test_create_nodes_from_html_list_merging():
    """Test case for merging paragraph blocks into a list structure.
    
    This test demonstrates the scenario where multiple paragraph blocks with
    multiline text content are merged into a single <ul> element, with each
    line becoming a separate <li> item. The data-sources attributes track
    which original content blocks each element came from.
    """
    print("Starting test_create_nodes_from_html_list_merging...")
    from utils.schema import PositionalData, EmbeddingSource, TagName
    from transform.models import BlockType
    
    # Create input ContentBlocks - two paragraphs with multiline text
    content_blocks = [
        ContentBlock(
            positional_data=PositionalData(
                page_pdf=1,
                page_logical=1,
                bbox={"x1": 100, "y1": 200, "x2": 500, "y2": 250}
            ),
            block_type=BlockType.TEXT,
            embedding_source=EmbeddingSource.TEXT_CONTENT,
            text_content="First item from block 0\nSecond item from block 0\nThird item from block 0"
        ),
        ContentBlock(
            positional_data=PositionalData(
                page_pdf=1,
                page_logical=1,
                bbox={"x1": 100, "y1": 260, "x2": 500, "y2": 310}
            ),
            block_type=BlockType.TEXT,
            embedding_source=EmbeddingSource.TEXT_CONTENT,
            text_content="Fourth item from block 1\nFifth item from block 1"
        )
    ]
    
    # HTML string where the paragraphs are merged into a single ul
    # with data-sources tracking the original blocks
    html = """<ul data-sources="0,1">
    <li data-sources="0">First item from block 0</li>
    <li data-sources="0">Second item from block 0</li>
    <li data-sources="0">Third item from block 0</li>
    <li data-sources="1">Fourth item from block 1</li>
    <li data-sources="1">Fifth item from block 1</li>
</ul>"""
    
    # Expected output: StructuredNodes representing the list structure
    expected_nodes = [
        StructuredNode(
            tag=TagName.UL,
            children=[
                StructuredNode(
                    tag=TagName.LI,
                    text="First item from block 0",
                    positional_data=[content_blocks[0].positional_data]
                ),
                StructuredNode(
                    tag=TagName.LI,
                    text="Second item from block 0",
                    positional_data=[content_blocks[0].positional_data]
                ),
                StructuredNode(
                    tag=TagName.LI,
                    text="Third item from block 0",
                    positional_data=[content_blocks[0].positional_data]
                ),
                StructuredNode(
                    tag=TagName.LI,
                    text="Fourth item from block 1",
                    positional_data=[content_blocks[1].positional_data]
                ),
                StructuredNode(
                    tag=TagName.LI,
                    text="Fifth item from block 1",
                    positional_data=[content_blocks[1].positional_data]
                )
            ],
            positional_data=[
                content_blocks[0].positional_data,
                content_blocks[1].positional_data
            ]
        )
    ]
    
    # Test the function
    print("Running create_nodes_from_html...")
    result = create_nodes_from_html(html, content_blocks)
    print(f"Got {len(result)} nodes")

    # Assertions comparing result to expected structure
    assert len(result) == len(expected_nodes), f"Expected {len(expected_nodes)} nodes, got {len(result)}"
    
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
    
    print("Test case specification complete - ready for implementation!")


if __name__ == "__main__":
    try:
        test_create_nodes_from_html_list_merging()
        print("✅ Test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()