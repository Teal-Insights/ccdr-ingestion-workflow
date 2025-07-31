import logging
import asyncio
import json
import os
import re
from typing import Literal, Optional
from litellm import Router
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices
from pydantic import BaseModel, Field, ValidationError, field_validator

from transform.detect_top_level_structure import parse_range_string
from transform.models import ContentBlock, StructuredNode
from utils.schema import TagName, PositionalData
from utils.html import create_nodes_from_html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SplitGroup(BaseModel):
    range: str = Field(description="Comma-separated inclusive id range of elements to be included in the group (e.g., '0-51,53')")
    tag: Optional[Literal["section", "aside", "nav"]] = Field(default=None, description="Tag in which to enclose the entire group, if any")
    
    @field_validator('range')
    @classmethod
    def validate_range_string(cls, v: str) -> str:
        """Validate that range string only contains numbers, spaces, dashes, or commas."""
        if not re.match(r'^[0-9\s,\-]+$', v):
            raise ValueError("Range string can only contain numbers, spaces, dashes, or commas")
        return v

    def get_html(self, html: str) -> str:
        current_chunk = []
        for id in parse_range_string(self.range):
            pattern = re.compile(rf"^.+ id=\"{id}\".+$", re.MULTILINE)
            match = pattern.search(html)
            if match:
                current_chunk.append(match.group(0))
            else:
                raise ValueError(f"Tag with id {id} not found in source HTML")
        return "\n".join(current_chunk)

class SplitResult(BaseModel):
    groups: list[SplitGroup] = Field(default_factory=list, description="Ranges of ids that will be grouped together")

    def split_html(self, html: str) -> list[str]:
        """Assuming one tag per line and a numerical id for each tag, split the HTML into chunks by id list"""
        return [group.get_html(html) for group in self.groups]
    
    def validate_comprehensive_coverage(self, num_content_blocks: int) -> bool:
        """
        Validate that group ranges comprehensively cover all content blocks from 0 to num_content_blocks-1.
        """
        if num_content_blocks <= 0:
            if not self.groups:
                return True
            raise ValueError("Cannot have groups when num_content_blocks is 0 or negative")
        
        group_ids = [id for group in self.groups for id in parse_range_string(group.range)]

        covered_ids = set(group_ids)
        expected_ids = set(range(num_content_blocks))
        if not covered_ids == expected_ids:
            raise ValueError(f"Group ranges must cover all content blocks exactly once. Difference: {covered_ids.difference(expected_ids)}")

        if len(group_ids) > len(covered_ids):
            duplicates = [id for id in covered_ids if group_ids.count(id) > 1]
            raise ValueError(f"Group ranges contained duplicate ID coverage: {sorted(duplicates)}")
        return True


class SplitSection(BaseModel):
    """
    Represents a section created by splitting HTML content, containing
    a tag type, content blocks, and the HTML string representation.
    """
    tag: Optional[Literal["section", "aside", "nav"]] = Field(
        default=None, 
        description="Tag in which to enclose the entire section, if any"
    )
    content_blocks: list[ContentBlock] = Field(
        description="List of content blocks in this section"
    )
    html_str: str = Field(
        description="HTML string representation of this section"
    )

    @classmethod
    def from_split_group(
        cls,
        split_group: SplitGroup,
        html: str,
        content_blocks: list[ContentBlock],
    ) -> "SplitSection":
        html_str = split_group.get_html(html)
        return cls(
            tag=split_group.tag,
            content_blocks=content_blocks,
            html_str=html_str
        )


class HTMLResult(BaseModel):
    html: str = Field(description="Revised HTML content with semantic tags, logical hierarchy, and data-sources attributes")


SPLIT_PROMPT = """You will be given a very long HTML document with a flat structure and only `p` and `img` tags.
Subagents will process the content to propose a better structured HTML representation of the content, with semantic tags and logical hierarchy.
However, the input is too long for a single subagent to process.
It must be split into at least {num_chunks} sections, each no longer than {max_chunk_size} characters.
Chunk boundaries should ideally never interrupt a logical section of the document, or else subagents will be unable to group related content in their output HTML.
If an entire chunk should be wrapped in a structural container, specify the tag in the `tag` field.
Typically a range will be a single group of contiguous blocks, but there may be edge cases where the blocks are slightly out of logical order, in which case you can tack on an extra block or range with a comma (e.g., `0-51,53`).

# Response format

Return your response in the following JSON format:

```json
{response_schema}
```

# Task

Below is the HTML content you are to split.

Content:

```html
{html_representation}
```
"""


# TODO: Add sectionType classifications
HTML_PROMPT = """You are an expert in HTML and document structure.
You are given an HTML partial with a flat structure and only `p` and `img` tags.
Your task is to propose a better structured HTML representation of the content, with semantic tags and logical hierarchy.

# Requirements

- You may only use the following tags: {allowed_tags}.
- You may split, merge, or replace structural containers as necessary, but you should make an effort to:
    - Clean up any whitespace, encoding, redundant style tags, or other formatting issues
    - Otherwise maintain the identical wording/spelling of the text content and of image descriptions and source URLs
    - Add a `data-sources` attribute to any content-bearing elements with a comma-separated list of ids of the source elements (for attribute mapping and content validation). This can include ranges, e.g., `data-sources="0-5,7,9-12"`

# Response format

Return the full revised HTML content in the following JSON format:

```json
{response_schema}
```

# Task

Below is the HTML content you are to transform.
The parent node(s) of this content is/are: {parent_tags}.

Content:

```html
{html_representation}
```
"""

ALLOWED_TAGS: str = ", ".join(
    tag.value for tag in TagName
    if tag not in [TagName.HEADER, TagName.MAIN, TagName.FOOTER]
) + ", b, i, u, s, sup, sub, br"


MODEL_TOKEN_LIMIT: int = 128000
# assume 4 chars per token, allow ~1/3 margin of error
MAX_CHUNK_CHAR_SIZE: int = ((MODEL_TOKEN_LIMIT * 4) * 2) // 3


def de_fence(text: str, type: Literal["json", "html"] = "json") -> str:
    """If response is fenced code block, remove fence"""
    stripped_text = text.strip()
    if stripped_text.startswith((f"```{type}\n", "```\n", f"``` {type}\n")):
        # Find the first newline after the opening fence
        first_newline = stripped_text.find('\n')
        if first_newline != -1:
            # Remove the opening fence
            stripped_text = stripped_text[first_newline + 1:]
        
        # Remove the closing fence if it exists
        if stripped_text.endswith("\n```"):
            stripped_text = stripped_text[:-4].rstrip()
    
    return stripped_text


# TODO: Experiment with Claude, which can do 128k tokens output
def create_router(
    gemini_api_key: str, 
    openai_api_key: str, 
    deepseek_api_key: str,
    openrouter_api_key: str,
) -> Router:
    """Create a LiteLLM Router with advanced load balancing and fallback configuration."""
    model_list = [
        # {
        #     "model_name": "html-parser",
        #     "litellm_params": {
        #         "model": "openrouter/x-ai/grok-3-mini", # 16k tokens output
        #         "api_key": openrouter_api_key,
        #         "max_parallel_requests": 10,
        #         "weight": 1,
        #     }
        # },
        # {
        #     "model_name": "html-parser",
        #     "litellm_params": {
        #         "model": "openai/gpt-4o-mini", # 16k tokens output
        #         "api_key": openai_api_key,
        #         "max_parallel_requests": 10,
        #         "weight": 1,
        #     }
        # },
        # {
        #     "model_name": "html-parser",
        #     "litellm_params": {
        #         "model": "openai/gpt-4.1-mini", # 32k tokens output
        #         "api_key": openai_api_key,
        #         "max_parallel_requests": 10,
        #         "weight": 1,
        #     }
        # }
        # {
        #     "model_name": "html-parser",
        #     "litellm_params": {
        #         "model": "openrouter/openrouter/horizon-alpha", # 128k tokens output
        #         "api_key": openrouter_api_key,
        #         "max_parallel_requests": 3,
        #         "weight": 1
        #     }
        # },
        {
            "model_name": "html-parser",
            "litellm_params": {
                "model": "openrouter/anthropic/claude-sonnet-4", # 128k tokens output
                "api_key": openrouter_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        }
    ]

    # Router configuration
    return Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",  # Weighted random selection
        fallbacks=[{"html-parser": ["html-parser"]}],  # Falls back within the same group
        num_retries=2,
        allowed_fails=5,
        cooldown_time=30,
        enable_pre_call_checks=True,  # Enable context window and rate limit checks
        default_max_parallel_requests=50,  # Global default
        set_verbose=False,  # Set to True for debugging
    )


def _positional_data_from_blocks(content_blocks: list[ContentBlock], page_dimensions: dict[int, dict[str, int]]) -> list[PositionalData]:
    """
    Extract unique positional data per page from a list of content blocks.
    
    Args:
        content_blocks: List of content blocks that may span multiple pages
        page_dimensions: Dict mapping page numbers to full-page bbox dimensions
        
    Returns:
        List of PositionalData with one entry per unique page, using full-page bbox
    """
    unique_pages = set()
    positional_data = []
    
    for block in content_blocks:
        page_pdf = block.positional_data.page_pdf
        if page_pdf not in unique_pages:
            unique_pages.add(page_pdf)
            positional_data.append(
                PositionalData(
                    page_pdf=page_pdf,
                    bbox=page_dimensions[page_pdf - 1],
                )
            )
    
    return positional_data


async def _split_html(
    content_blocks: list[ContentBlock],
    router: Router,
    messages: Optional[list[dict[str, str]]] = None,
    max_validation_attempts: int = 3,
    attempt: int = 0,
) -> list[SplitSection]:
    """Process a single input using LiteLLM Router with built-in concurrency control and fallbacks."""
    html = "\n".join(block.to_html(block_id=i) for i, block in enumerate(content_blocks))
    if len(html) > MAX_CHUNK_CHAR_SIZE:
        messages = messages or [
            {
                "role": "user",
                "content": SPLIT_PROMPT.format(
                    html_representation=html,
                    num_chunks=len(html) // MAX_CHUNK_CHAR_SIZE + 1,
                    max_chunk_size=MAX_CHUNK_CHAR_SIZE,
                    response_schema=SplitResult.model_json_schema()
                )
            }
        ]

        response = await router.acompletion(
            model="html-parser",
            messages=messages, # type: ignore
            temperature=0.0,
            response_format=SplitResult
        )

        try:
            if (
                response
                and isinstance(response, ModelResponse)
                and isinstance(response.choices[0], Choices)
                and response.choices[0].message.content
            ):
                split_result = SplitResult.model_validate_json(de_fence(response.choices[0].message.content))
                split_result.validate_comprehensive_coverage(len(content_blocks))
                split_sections = [
                    SplitSection.from_split_group(group, html, content_blocks)
                    for group in split_result.groups
                ]

                # Debug: Dump input and response to file
                with open(os.path.join("artifacts", "split_html_input.html"), "w") as fw:
                    fw.write(html)
                with open(os.path.join("artifacts", "split_html_output.json"), "w") as fw:
                    fw.write(split_result.model_dump_json())

                for i, split_section in enumerate(split_sections):
                    if len(split_section.html_str) > MODEL_TOKEN_LIMIT * 3: # Slightly more permissive than the prompt calls for
                        raise ValueError(f"Split section {i} is too long: {len(split_section.html_str)} characters")
   
                return split_sections
            else:
                raise ValueError("No valid response from LLM")
        except (ValidationError, ValueError) as e:
            if attempt < max_validation_attempts - 1:
                logger.warning(f"Validation error (attempt {attempt+1}/{max_validation_attempts}): {e}")
                # Append error message and retry
                if (
                    response
                    and isinstance(response, ModelResponse)
                    and isinstance(response.choices[0], Choices)
                    and response.choices[0].message.content
                ):
                    messages.append(response.choices[0].message.model_dump())
                messages.append(
                {
                    "role": "user",
                    "content": f"Your previous response had a validation error: {str(e)}. "
                                "Please correct your response to match the required schema and constraints."
                })
                return await _split_html(content_blocks, router, messages, max_validation_attempts, attempt + 1)
            else:
                raise ValueError(f"Validation error on final attempt: {e}")
    else:
        return [SplitSection(tag=None, content_blocks=content_blocks, html_str=html)]


async def _restructure_html(html_str: str, parents: str, router: Router) -> str:
    """
    Use LLM to restructure HTML content with better semantic structure.
    
    Args:
        html_str: HTML string to restructure (from section.html_str)
        parents: Parent tag context for the LLM
        router: LiteLLM Router for LLM calls
        
    Returns:
        Restructured HTML string
    """    
    # Prepare the prompt
    messages = [
        {
            "role": "user",
            "content": HTML_PROMPT.format(
                html_representation=html_str,
                parent_tags=parents,
                allowed_tags=ALLOWED_TAGS,
                response_schema=HTMLResult.model_json_schema()
            )
        }
    ]
    
    # Make the LLM call
    try:
        response = await router.acompletion(
            model="html-parser",
            messages=messages, # type: ignore
            temperature=0.0
        )

        # Debug: Dump input and response to file
        with open(os.path.join("artifacts", "restructure_html_input.html"), "w") as fw:
            fw.write(html_str)
        with open(os.path.join("artifacts", "restructure_html_response.json"), "w") as fw:
            fw.write(response.choices[0].message.content)

        if (
            response
            and isinstance(response, ModelResponse)
            and isinstance(response.choices[0], Choices)
            and response.choices[0].message.content
        ):
            # Remove any code fencing and return the restructured HTML
            html_result = HTMLResult.model_validate_json(de_fence(response.choices[0].message.content, type="json"))
            if not html_result.html.strip():
                raise ValueError("No valid response from LLM for HTML restructuring, returning original HTML")
            
            # Validate that all data-sources attributes are valid range strings
            data_sources = re.findall(r'data-sources="([^"]+)"', html_result.html)
            for data_source in data_sources:
                if not re.match(r'^[0-9\s,\-]+$', data_source):
                    raise ValueError(f"Invalid data-sources attribute: {data_source}. Must be a comma-separated list with only numbers, spaces, or dashes.")
            
            return html_result.html
        else:
            logger.warning("No valid response from LLM for HTML restructuring, returning original HTML")
            return html_str
            
    except Exception as e:
        logger.error(f"Error during HTML restructuring: {e}, returning original HTML")
        return html_str


if __name__ == "__main__":
    import time
    import dotenv
    import pymupdf

    dotenv.load_dotenv()

    async def _test_main() -> None:
        """Ad-hoc harness for manual testing; not used by production code."""

        # Get API keys for all providers
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        assert gemini_api_key, "GEMINI_API_KEY is not set"
        assert openai_api_key, "OPENAI_API_KEY is not set"
        assert deepseek_api_key, "DEEPSEEK_API_KEY is not set"
        assert openrouter_api_key, "OPENROUTER_API_KEY is not set"

        # Create router with built-in load balancing and concurrency control
        router = create_router(
            gemini_api_key, 
            openai_api_key, 
            deepseek_api_key,
            openrouter_api_key,
        )

        doc = pymupdf.open(os.path.join("artifacts", "wkdir", "doc_601.pdf"))
        page_dimensions = {
            page.number: {"x1": 0, "y1": 0, "x2": page.rect.width, "y2": page.rect.height}
            for page in doc
        }

        with open(os.path.join("artifacts", "doc_601_top_level_structure.json"), "r") as fr:
            top_level_structure: list[tuple[TagName, list[ContentBlock]]] = [
                (TagName(tag_name), [ContentBlock.model_validate(block) for block in blocks])  # type: ignore[arg-type]
                for tag_name, blocks in json.load(fr)
            ]

        parent_nodes = [
            StructuredNode(
                tag=tag_name,
                children=[],
                positional_data=_positional_data_from_blocks(blocks, page_dimensions)
            )
            for tag_name, blocks in top_level_structure
        ]

        split_section_tasks = [
            _split_html(blocks, router)
            for _, blocks in top_level_structure
        ]
        split_sections: list[list[SplitSection]] = await asyncio.gather(*split_section_tasks)

        for parent_node, split in zip(parent_nodes, split_sections):
            for section in split:
                restructured_html = await _restructure_html(
                    section.html_str,
                    parent_node.tag + ", " + section.tag if section.tag else parent_node.tag,
                    router
                )
                child_nodes = create_nodes_from_html(restructured_html, section.content_blocks)
                if section.tag:
                    parent_node.children.append(
                        StructuredNode(
                            tag=TagName(section.tag),
                            children=child_nodes,
                            positional_data=_positional_data_from_blocks(
                                section.content_blocks, page_dimensions
                            )
                        )
                    )
                else:
                    parent_node.children.extend(child_nodes)

        with open(os.path.join("artifacts", "doc_601_nested_structure.json"), "w") as fw:
            json.dump([node.model_dump() for node in parent_nodes], fw, indent=2)

    start_time = time.time()
    asyncio.run(_test_main())
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time} seconds")