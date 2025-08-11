import asyncio
import base64
import io
import os
from PIL import Image
import pydantic
from litellm import Choices, Router
from litellm.files.main import ModelResponse
from sqlmodel import Session, select

from transform.models import ContentBlock, BlockType
from utils.schema import Publication, Document
from utils.db import engine
from utils.litellm_router import create_router


class ImageDescription(pydantic.BaseModel):
    label: str
    description: str


async def describe_images_with_vlm(
    content_blocks_with_images: list[ContentBlock],
    temp_dir: str,
    document_id: int,
    gemini_api_key: str,
    openai_api_key: str,
    deepseek_api_key: str,
    openrouter_api_key: str,
) -> list[ContentBlock]:
    """Use Gemini to describe the image with async retry logic."""

    # Get publication title and document description to include in the prompt
    with Session(engine) as session:
        # Query publication title through the document's publication_id
        publication_title: str = session.exec(
            select(Publication.title)
            .join(Document, Publication.id == Document.publication_id)
            .where(Document.id == document_id)
        ).one()
        
        # Query document description directly
        document_description: str = session.exec(
            select(Document.description).where(Document.id == document_id)
        ).one()

    prompt: str = (
        "The following image comes from a World Bank publication titled: "
        f"{publication_title}. The document description is: {document_description}. "
        "Describe the image in detail. Return JSON with the following fields:\n\n"
        "- A label from: 'chart', 'graph', 'diagram', 'map', 'photo', 'table', or 'text_box'\n"
        "- A description of what the image shows/communicates\n\n"
        "You may be provided with the text that surrounds the image, which you can "
        "use to inform your response, but you should focus on primarily the image itself."
    )

    # Create tasks for async processing
    router = create_router(gemini_api_key, openai_api_key, deepseek_api_key, openrouter_api_key)
    tasks = []
    
    for i, content_block in enumerate(content_blocks_with_images):
        # Skip non-image blocks
        if content_block.block_type != BlockType.PICTURE:
            continue

        # Use enumeration index for image path, matching extract_images.py format
        image_path = os.path.join(temp_dir, "images", f"doc_{document_id}_{i}.webp")

        # Check if image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)

        # Convert PIL Image to base64
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="WEBP")
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        # Get text from previous two and next two content blocks if of a text-like type and on same page
        current_page: int = content_blocks_with_images[i].positional_data.page_pdf
        text_surrounding_image: str = "Text surrounding the image (if any):\n"
        for j in range(i-2, i+3):
            if (
                j >= 0
                and j < len(content_blocks_with_images)
                and content_blocks_with_images[j].positional_data.page_pdf == current_page
                and (content_blocks_with_images[j].block_type in [
                    BlockType.TEXT, BlockType.TITLE, BlockType.SECTION_HEADER,
                    BlockType.CAPTION, BlockType.LIST_ITEM, BlockType.FORMULA,
                    BlockType.TABLE
                ] or j == i) # Include the current block if it has text_content
                and content_blocks_with_images[j].text_content  # Check text_content exists
            ):
                text_content = content_blocks_with_images[j].text_content
                if text_content:  # Additional check for non-empty content
                    text_surrounding_image += text_content
                    text_surrounding_image += "\n"

        # Create message content with text and image
        message = [
            {
                "role": "system",
                "content": f"{prompt}"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_surrounding_image},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/webp;base64,{img_base64}"},
                    },
                ],
            }
        ]

        # Create task and include the content block index for pairing
        task = asyncio.create_task(
            _process_single_image(
                message, i, router
            )
        )
        tasks.append(task)

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Process results and update corresponding content blocks
    for result in results:
        if result is not None:
            block_index, description = result
            content_blocks_with_images[block_index].description = description
            
    return content_blocks_with_images


async def _process_single_image(
    messages: list, block_index: int, router: Router
) -> tuple[int, str] | None:
    """Process a single image."""
    try:
        # litellm router exposes `acompletion` for async completion
        response = await router.acompletion(
            model="image-classifier",
            messages=messages,  # type: ignore
            temperature=0.0,
            response_format=ImageDescription
        )

        if (
            response
            and isinstance(response, ModelResponse)
            and isinstance(response.choices[0], Choices)
            and response.choices[0].message.content
        ):
            image_description = ImageDescription.model_validate_json(response.choices[0].message.content)
            return block_index, f"{image_description.label}: {image_description.description}"
        else:
            print(f"Warning: No valid response from Gemini for block {block_index}")
            return None
    except Exception as e:
        print(f"Error processing block {block_index}: {e}")
        return None


if __name__ == "__main__":
    import json
    import dotenv

    dotenv.load_dotenv(override=True)

    gemini_api_key, openai_api_key, deepseek_api_key, openrouter_api_key = "", "", "", os.getenv("OPENROUTER_API_KEY", "")
    temp_dir: str = "./artifacts"
    document_id: int = 601

    with open(os.path.join("artifacts", "doc_601_content_blocks_with_images.json"), "r") as fr:
        content_blocks_with_images = json.load(fr)
        content_blocks_with_images = [ContentBlock.model_validate(block) for block in content_blocks_with_images]
    content_blocks_with_descriptions = asyncio.run(describe_images_with_vlm(
        content_blocks_with_images, temp_dir, document_id, gemini_api_key, openai_api_key, deepseek_api_key, openrouter_api_key
    ))
    print("Images described successfully!")
    with open(os.path.join("artifacts", "doc_601_content_blocks_with_descriptions.json"), "w") as fw:
        json.dump([block.model_dump() for block in content_blocks_with_descriptions], fw, indent=2)