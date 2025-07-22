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


class ContentBlock(ContentBlockBase):
    """
    Adds text content or image description and storage location to the
    base content block. Contains all the information needed to create a
    database content node.
    """
    text_content: Optional[str] = Field(default=None, description="Text content (if a text block)")
    storage_url: Optional[str] = Field(default=None, description="URL of the image or other content (if a non-text block)")
    description: Optional[str] = Field(default=None, description="Description of the block (if a non-text block)")
    caption: Optional[str] = Field(default=None, description="Caption associated with the block (if available)")
    