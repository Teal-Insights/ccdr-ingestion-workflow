from typing import Literal
from litellm import Router
from pydantic import BaseModel, Field
from utils.html import ALLOWED_TAGS


class Feedback(BaseModel):
    message: str = Field(description="A human-readable description of the issue")
    affected_ids: list[int] = Field(description="Element IDs from the input file that are affected by the issue")
    severity: Literal["critical", "minor"] = Field(description="The severity of the issue")


PROMPT: str = f"""# Task

An AI agent was asked to convert an HTML partial with a flat structure comprised of only `p` and `img` tags to a better structured HTML representation of the content, with semantic tags and logical hierarchy. You are tasked to provide feedback to the agent.

## Original requirements

- Only these tags are allowed: {", ".join(f"`{tag}`" for tag in ALLOWED_TAGS)}.
- Only `header` (front matter), `main` (body content), and `footer` (back matter) are allowed at the top level; all other tags should be nested within one of these.
- Whitespace and encoding issues should be cleaned up; otherwise maintain the identical wording/spelling of the text content and of image descriptions and source URLs.
- Assign elements a `data-sources` attribute with a comma-separated list of ids of the source elements (for attribute mapping and content validation). This can include ranges, e.g., `data-sources="0-5,7,9-12"`. Inline style tags `b`, `i`, `u`, `s`, `sup`, `sub`, and `br` do not need `data-sources`.
- The document should be unstyled.

## Output instructions

Issues to flag:

- Very badly formed HTML (for instance, table of contents items massed with line breaks into a single `p` tag rather than represented as list items)
- Missing content (e.g., use of truncated text or placeholder text in output rather than the full original text)
- Incorrect content (e.g., text that is not in the original document or has been significantly altered from the original)

## Output format

Please provide your feedback as a JSON array of objects with the following schema:

{Feedback.model_json_schema()}

If there are no issues, please return an empty array."""

async def provide_feedback(
    input_html: str,
    output_html: str,
    router: Router,
) -> str:
    input_lines: list[int] = [str(i + 1) + ": " + line for i, line in enumerate(input_html.split("\n"))]
    output_lines: list[int] = [str(i + 1) + ": " + line for i, line in enumerate(output_html.split("\n"))]

    # Create message content with text and image
    messages = [
        {
            "role": "system",
            "content": f"<prompt>\n{PROMPT}\n</prompt>"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<original_html>\n{'\n'.join(input_lines)}\n</original_html>\n<modified_html>\n{'\n'.join(output_lines)}\n</modified_html>",
                },
            ],
        }
    ]

    response = await router.acompletion(
        model="gemini/gemini-2.5-flash",
        messages=messages,
        response_format=Feedback,
    )

    return response.choices[0].message.content