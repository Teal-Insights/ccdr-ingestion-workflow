from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from typing import Optional
from utils.schema import PositionalData, EmbeddingSource

class BlockType(str, Enum):
    """
    Represents the type of a block extracted from a PDF page.
    """
    TEXT = "Text"
    TITLE = "Title"
    SECTION_HEADER = "Section header"
    PICTURE = "Picture"
    TABLE = "Table"
    LIST_ITEM = "List item"
    FORMULA = "Formula"
    FOOTNOTE = "Footnote"
    PAGE_HEADER = "Page header"
    PAGE_FOOTER = "Page footer"
    CAPTION = "Caption"

class ExtractedLayoutBlock(BaseModel):
    """
    Represents a raw layout element extracted from a PDF page by the Huridocs
    PDF document layout extractor.
    """
    # Silently discard `id` field if present
    model_config = ConfigDict(extra='ignore')

    left: float = Field(description="X-coordinate of the left edge of the block")
    top: float = Field(description="Y-coordinate of the top edge of the block")
    width: float = Field(description="Width of the block")
    height: float = Field(description="Height of the block")
    page_number: int = Field(
        ge=1, description="Page number where this block appears (1-indexed)"
    )
    page_width: float = Field(description="Width of the page in points")
    page_height: float = Field(description="Height of the page in points")
    text: str = Field(description="Text content of the layout block")
    type: BlockType = Field(description="Type/category of the layout block (e.g., 'Page header', 'Paragraph', etc.)")


class LayoutBlock(ExtractedLayoutBlock):
    """
    Represents an extracted layout block enriched with logical page numbers.
    """
    logical_page_number: Optional[str] = Field(
        description="Logical page number where this block appears (integer, roman, or letter)"
    )


class ContentBlockBase(BaseModel):
    """
    Conforms the layout block's positional data to the schema used by
    content nodes in our database. Also fixes up the block type using
    rules-based heuristics, and infers the embedding source from the
    block type.
    """
    positional_data: PositionalData = Field(description="Positional data for the block")
    block_type: BlockType = Field(description="Type/category of the layout block (e.g., 'Page header', 'Paragraph', etc.)")
    embedding_source: Optional[EmbeddingSource] = Field(default=None, description="Whether to embed text, description, or caption")
    text_content: Optional[str] = Field(default=None, description="Text content (if a text block)")


class ContentBlock(ContentBlockBase):
    """
    Adds image description and storage location to the base content
    block. Contains all the information needed to create a database
    content node.
    """
    storage_url: Optional[str] = Field(default=None, description="URL of the image or other content (if a non-text block)")
    description: Optional[str] = Field(default=None, description="Description of the block (if a non-text block)")
    caption: Optional[str] = Field(default=None, description="Caption associated with the block (if available)")

    def to_html(self, bboxes: bool = False, block_id: int = None) -> str:
        """
        Convert the content block to an HTML string.
        """
        if self.block_type == BlockType.PICTURE:
            return create_image_block_html(self, bboxes=bboxes, block_id=block_id)
        else:
            return create_text_block_html(self, bboxes=bboxes, block_id=block_id)


def create_image_block_html(block: ContentBlock, bboxes: bool = False, block_id: int = None) -> str:
    """
    Create an img element for a picture block.
    
    Args:
        block: ContentBlock of type PICTURE
        bboxes: If True, include bbox coordinates as data attributes
        block_id: Optional block ID for the element
        
    Returns:
        HTML img element as string
    """
    img_attrs = []
    
    # Add id if provided
    if block_id is not None:
        img_attrs.append(f'id="{block_id}"')
    
    # Add page data attribute
    img_attrs.append(f'data-page="{block.positional_data.page_pdf}"')
    
    # Add src attribute if storage_url is available
    if block.storage_url:
        img_attrs.append(f'src="{block.storage_url}"')
    
    # Add alt text from description or text_content
    alt_text = block.description or block.text_content or "Image"
    img_attrs.append(f'alt="{alt_text}"')
    
    # Add bbox data attribute if requested
    if bboxes and block.positional_data.bbox:
        bbox_values = [
            block.positional_data.bbox["x1"],
            block.positional_data.bbox["y1"], 
            block.positional_data.bbox["x2"],
            block.positional_data.bbox["y2"]
        ]
        bbox_str = ",".join(str(int(coord)) for coord in bbox_values)
        img_attrs.append(f'data-bbox="{bbox_str}"')
    
    attrs_str = " ".join(img_attrs)
    return f"<img {attrs_str} />"


def create_text_block_html(block: ContentBlock, bboxes: bool = False, block_id: int = None) -> str:
    """
    Create a p element for a text block.
    
    Args:
        block: ContentBlock with text content
        bboxes: If True, include bbox coordinates as data attributes
        block_id: Optional block ID for the element
        
    Returns:
        HTML p element as string
    """
    p_attrs = []
    
    # Add id if provided
    if block_id is not None:
        p_attrs.append(f'id="{block_id}"')
    
    # Add page data attribute
    p_attrs.append(f'data-page="{block.positional_data.page_pdf}"')
    
    # Add bbox data attribute if requested
    if bboxes and block.positional_data.bbox:
        bbox_values = [
            block.positional_data.bbox["x1"],
            block.positional_data.bbox["y1"],
            block.positional_data.bbox["x2"], 
            block.positional_data.bbox["y2"]
        ]
        bbox_str = ",".join(str(int(coord)) for coord in bbox_values)
        p_attrs.append(f'data-bbox="{bbox_str}"')
    
    attrs_str = " ".join(p_attrs)
    attrs_part = f" {attrs_str}" if attrs_str else ""
    
    # Use text_content or fallback to empty string
    text_content = block.text_content or ""
    
    return f"<p{attrs_part}>{text_content}</p>"