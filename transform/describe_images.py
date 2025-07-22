# TODO: Should be an easy matter to include a little text context from
# neighboring text blocks with the image to improve the description

import asyncio
import base64
import io
import os
from PIL import Image
import pydantic
from litellm import completion, Choices
from litellm.files.main import ModelResponse
from tenacity import retry, stop_after_attempt, wait_exponential

from transform.models import ContentBlock


class ImageDescription(pydantic.BaseModel):
    label: str
    description: str


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def describe_images_with_vlm(
    content_blocks_with_images: list[ContentBlock], api_key: str, temp_dir: str, document_id: int
) -> list[ContentBlock]:
    """Use Gemini to describe the image with async retry logic."""
    
    prompt = """Describe the image in detail. Include the following:
    - A label from: "chart", "graph", "diagram", "map", "photo", "table", or "text_box"
    - A description of what the image shows/communicates"""

    # Use semaphore to limit concurrent API calls
    tasks = []
    async with asyncio.Semaphore(2):
        for content_block in content_blocks_with_images:
            image_path = os.path.join(temp_dir, "images", f"doc_{document_id}_{content_block.id}.png")
            image = Image.open(image_path)

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

            tasks.append(
                asyncio.create_task(
                    completion(
                        model="gemini/gemini-2.5-flash",
                        messages=message,
                        temperature=0.0,
                        response_format={
                            "type": "json_object",
                            "response_schema": ImageDescription.model_json_schema(),
                        },
                    )
                )
            )

    # Parse JSON response into Pydantic model
    responses = await asyncio.gather(*tasks)
    for response in responses:
        if (
            response
            and isinstance(response, ModelResponse)
            and isinstance(response.choices[0], Choices)
            and response.choices[0].message.content
        ):
            description = ImageDescription.model_validate_json(
                response.choices[0].message.content
            )
            content_block.description = f"{description.label}: {description.description}"
        else:
            raise Exception("No valid response from Gemini")
