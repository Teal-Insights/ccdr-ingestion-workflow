from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Union, Any
from abc import ABC


class BaseBlock(BaseModel, ABC):
    """
    Abstract base class for all block types extracted from a PDF page.
    """
    block_type: str = Field(
        description="Type of the block (text, image, svg, etc.)"
    )
    page_number: int = Field(
        ge=1, 
        description="Page number where this block appears (1-indexed)"
    )
    bbox: List[float] = Field(
        min_length=4,
        max_length=4,
        description="Bounding box coordinates [x0, y0, x1, y1] where (x0,y0) is top-left and (x1,y1) is bottom-right"
    )


class TextBlock(BaseBlock):
    """
    Represents a single text block extracted from a PDF page.
    """
    block_type: Literal["text"] = Field(
        default="text",
        description="Type of the block, always 'text' for text blocks"
    )
    text: str = Field(
        description="Reconstructed HTML with styling and semantic grouping"
    )
    plain_text: str = Field(
        description="Clean plain text content without HTML formatting"
    )


class ImageBlock(BaseBlock):
    """
    Represents a single image block extracted from a PDF page.
    """
    block_type: Literal["image"] = Field(
        default="image",
        description="Type of the block, always 'image' for image blocks"
    )
    storage_url: str = Field(
        description="Path or URL to the stored image file"
    )
    description: str = Field(
        description="AI-generated description of the image content"
    )


class SvgBlock(BaseBlock):
    """
    Represents a single SVG block extracted from a PDF page.
    """
    block_type: Literal["svg"] = Field(
        default="svg",
        description="Type of the block, always 'svg' for SVG blocks"
    )
    storage_url: str = Field(
        description="Path or URL to the stored SVG file"
    )
    description: str = Field(
        description="AI-generated description of the SVG content"
    )


# Union type for all possible block types
Block = Union[TextBlock, ImageBlock, SvgBlock]


class BlocksDocument(BaseModel):
    """
    Represents the complete output from block extraction for a PDF document.
    This is a type-agnostic container that can hold any combination of block types.
    """
    pdf_path: str = Field(
        description="Path to the original PDF file that was processed"
    )
    total_pages: int = Field(
        ge=1,
        description="Total number of pages in the PDF document"
    )
    total_blocks: int = Field(
        ge=0,
        description="Total number of blocks extracted from all pages"
    )
    blocks: List[Block] = Field(
        description="List of all blocks extracted from the document"
    )

    @model_validator(mode='after')
    def validate_total_blocks(self):
        """Validate that total_blocks matches the actual length of blocks list"""
        if self.total_blocks != len(self.blocks):
            raise ValueError(
                f"total_blocks ({self.total_blocks}) does not match "
                f"the actual number of blocks ({len(self.blocks)})"
            )
        return self