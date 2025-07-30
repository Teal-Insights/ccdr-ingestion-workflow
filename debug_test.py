from utils.html import create_nodes_from_html
from utils.schema import PositionalData, EmbeddingSource, TagName
from transform.models import BlockType, ContentBlock

# Simple test
content_blocks = [
    ContentBlock(
        positional_data=PositionalData(
            page_pdf=1,
            page_logical=1,
            bbox={"x1": 100, "y1": 200, "x2": 500, "y2": 250}
        ),
        block_type=BlockType.TEXT,
        embedding_source=EmbeddingSource.TEXT_CONTENT,
        text_content="Test"
    )
]

html = '<li data-sources="0">First item from <i>block 0</i></li>'
result = create_nodes_from_html(html, content_blocks)

print(f"Result: {len(result)} nodes")
if result:
    print(f"First node text: '{result[0].text}'")
    print(f"Expected: 'First item from <i>block 0</i>'")
    print(f"Match: {result[0].text == 'First item from <i>block 0</i>'}")