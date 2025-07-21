import pymupdf
from transform.models import LayoutBlock, BlockType, ContentBlock
from utils.schema import EmbeddingSource, PositionalData
from transform.extract_drawings_by_bbox import is_completely_contained, extract_geometries_in_bbox
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
import asyncio
import base64
import io
import os
import dotenv
from litellm import acompletion

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_classification(input: str) -> BlockType:
    if "text" in input.lower() and not any(
        "picture" in input.lower(), "table" in input.lower()
    ):
        return BlockType.TEXT
    elif "picture" in input.lower() and not any(
        "text" in input.lower(), "table" in input.lower()
    ):
        return BlockType.PICTURE
    elif "table" in input.lower() and not any(
        "text" in input.lower(), "picture" in input.lower()
    ):
        return BlockType.TABLE
    else:
        print(f"Warning: LLM returned invalid classification ({input}); falling back to 'text'")
        return BlockType.TEXT


def get_embedding_source(block_type: BlockType) -> EmbeddingSource:
    if block_type == BlockType.TEXT:
        return EmbeddingSource.TEXT_CONTENT
    elif block_type == BlockType.PICTURE:
        return EmbeddingSource.DESCRIPTION


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def reclassify_images_with_gemini(
    images: list[Image.Image], api_key: str, semaphore: asyncio.Semaphore
) -> list[BlockType]:
    """Use Gemini to reclassify the image with async retry logic."""
    # Set environment variable for LiteLLM
    os.environ["GEMINI_API_KEY"] = api_key

    prompt = """Here is a layout block, extracted from a PDF. Classify it as a text
    block, a picture/figure block, or a table block. If it's a picture/figure block,
    return "picture". If it's a text block, return "text". If it's a table block,
    return "table". Charts should be classified as "picture", even if the legends or
    axes contain text. If the block contains more than one chart, return "picture".
    If it contains a combination of text and decorative elements (such as logos or
    icons), return "text".
    """

    messages = []
    for image in images:
        # Convert PIL Image to base64
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        # Create message content with text and image
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ],
            }
        ]
        messages.append(message)

    # Use semaphore to limit concurrent API calls
    tasks = []
    async with semaphore:
        for message in messages:
            tasks.append(acompletion(
                    model="gemini/gemini-2.5-flash",
                    messages=messages,
                    temperature=0.0
                )
            )

    responses = await asyncio.gather(*tasks)
    return [get_classification(response.message.content) for response in responses]


async def reclassify_block_types(blocks: list[LayoutBlock], pdf_path: str) -> list[ContentBlock]:
    """Reclassify block types based on the content of the bbox"""
    pdf = pymupdf.open(pdf_path)

    content_blocks: list[ContentBlock] = []
    indices_to_reclassify: list[int] = []
    for i, block in enumerate(blocks):
        # Coerce positional data to our own schema
        bbox = (block.left, block.top, block.left + block.width, block.top + block.height)
        content_block_base = ContentBlock(
            positional_data=PositionalData(
                bbox=bbox,
                page_pdf=block.page_number,
                page_logical=block.logical_page_number,
            )
        )

        # If a text block, check for images or drawings in the bbox
        if block.type == BlockType.TEXT:
            page = pdf[block.page_number - 1]

            # If image in bbox, reclassify as a picture block
            image_list = page.get_images(full=True)
            for xref, *_rest in image_list:
                rect_list = page.get_image_rects(xref)
                for rect in rect_list:
                    if is_completely_contained(rect, bbox):
                        content_block_base.block_type = BlockType.PICTURE
                        break

            # If a drawing in bbox, add to indices to reclassify
            page_svg_image = page.get_svg_image()
            drawings = extract_geometries_in_bbox(page_svg_image, bbox)
            if drawings:
                indices_to_reclassify.append(i)

        content_blocks.append(content_block_base)

    if indices_to_reclassify:
        semaphore = asyncio.Semaphore(2)
        classifications = await reclassify_images_with_gemini(
            [content_blocks[i].image for i in indices_to_reclassify], GEMINI_API_KEY, semaphore
        )
        for i, classification in enumerate(classifications):
            content_blocks[indices_to_reclassify[i]].block_type = classification

    for content_block in content_blocks:
        content_block.embedding_source = get_embedding_source(content_block.block_type)

    return content_blocks
