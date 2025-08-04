# TODO: Aggregate same-page bounding boxes for the same output node

from transform.models import ContentBlock, StructuredNode, BlockType
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
            text_content = text_content.strip() if text_content else None
        else:
            # Process children normally (no inline styling)
            for child in element.children:
                if isinstance(child, Tag):
                    children.append(_convert_element_to_node(child))
                elif isinstance(child, NavigableString) and child.strip():
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
    from utils.schema import PositionalData, EmbeddingSource, TagName, BoundingBox
    from transform.models import BlockType
    
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
                content_blocks[0].positional_data,
                content_blocks[1].positional_data
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


if __name__ == "__main__":
    try:
        test_create_nodes_from_html_list_merging()
        print("✅ Test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)