# TODOs:
#  1. Validate that the children and sources lists are disjoint
#  2. Validate that all original ids are present in the children and sources lists
#  3. Validate that list items have a list container parent
#  4. Validate that table components have a table container parent
#  5. Validate that the LLM doesn't propose a single wrapper node that has all the indices as its children
#  6. validate that any proposed img tag node has exactly one source, and the source is an img block
#  7. Accumulate cost of all calls to the LLM and log the total cost at the end
#  8. Add a "smart" model group to the router for fallback on validation errors
#  9. Add some examples to the prompt, maybe even use cosine similarity to find the most similar example
#  10. Remove attributes from the StructuredNode, since we're not using it
#  11. Use a helper for Markdown code block extraction in case the LLM wraps the output JSON
#  12. For images, get the text content of next sibling caption or figcaption block, if any, and assign to caption field

import logging
import asyncio
import json
import os
from datetime import datetime
from litellm import Router, completion_cost
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices
from typing import Optional
from pydantic import BaseModel, Field, ValidationError

from transform.detect_top_level_structure import parse_range_string
from transform.models import ContentBlock, StructuredNode, BlockType
from utils.schema import TagName, PositionalData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create specialized file logger for validated responses
def create_response_logger():
    """Create a file logger specifically for dumping validated LLM responses."""
    response_logger = logging.getLogger(f"{__name__}.responses")
    response_logger.setLevel(logging.INFO)

    # Prevent propagation to avoid duplicate logs in main logger
    response_logger.propagate = False

    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)

    # Create file handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"artifacts/validated_responses_{timestamp}.jsonl"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Use a simple formatter that just outputs the message (since we'll log JSON)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    response_logger.addHandler(file_handler)
    
    return response_logger

# Create the response logger instance
response_logger = create_response_logger()


class Context(BaseModel):
    parent_tags: list[TagName] = Field(default_factory=list, description="Parent tags of the current node")
    parent_heading: Optional[TagName] = Field(default=None, description="Parent heading of the current node")


class ProposedNode(BaseModel):
    tag: TagName = Field(description="HTML tag name")
    children: Optional[str] = Field(default=None, description="Comma-separated inclusive ranges for ids of elements that will become children of this node (e.g., '0-3,5,7-9')")
    sources: Optional[str] = Field(default=None, description="Comma-separated inclusive ranges for ids of elements being merged into or replaced by this node, or from which this node is being split (e.g., '4,6,10-12')")
    text: Optional[str] = Field(default=None, description="Text content (should only be present for leaf nodes, may contain `b`, `i`, `u`, `s`, `sup`, `sub` tags)")


class HTMLPartial(BaseModel):
    proposed_nodes: list[ProposedNode] = Field(description="Proposed nodes for the HTML partial")


class ParsedProposedNode(BaseModel):
    tag: TagName = Field(description="HTML tag name")
    children: list[int] = Field(default_factory=list, description="Ids of nodes that will become children of this node")
    sources: list[int] = Field(default_factory=list, description="Ids of nodes being merged into or replaced by this node, or from which this node is being split")
    text: Optional[str] = Field(default=None, description="Text content (should only be present for leaf nodes, may contain `b`, `i`, `u`, `s`, `sup`, `sub` tags)")


class ParsedHTMLPartial(BaseModel):
    proposed_nodes: list[ParsedProposedNode] = Field(description="Proposed nodes for the HTML partial")


PROMPT = """You are an expert in HTML and document structure.
You are given an HTML document with a flat structure and only `p` and `img` tags.
Your task is to propose a better structured HTML representation of the content, with semantic tags and logical hierarchy.

# Response format

Return your response in the following JSON format:

```json
{response_schema}
```

# Requirements

- You may only use the following tags: {allowed_tags}.
- Remember to use *inclusive* ranges. E.g., 0-2 encompasses 0, 1, and 2. The last index is included in the range!
- Every node should be mapped by id to exactly one `children` or `sources` field.
- `children` and `sources` must be disjoint.
    - That is, you may wrap elements in a structural container or mutate them by splitting/merging/replacing; don't do both to the same element.
    - Elements you wrap in a structural container will be further processed by a subagent, so they cease to be your concern.
- Leaves with `sources` are leaf nodes and should usually have `text` content (except for images).
- For purposes of this exercise, in-line styling tags like `b`, `i`, `u`, `s`, `sup`, and `sub` are considered text content.
- Clean up whitespace problems and mis-encoded character entities like `&amp;`, but otherwise remain faithful to the original text.

# Task

Below is the HTML content you are to transform.
The parent nodes of this content are: {parent_tags}.
The parent heading level, if any, is: {parent_heading}.

Content:

```html
{html_representation}
```
"""

ALLOWED_TAGS: str = ", ".join(
    tag.value for tag in TagName
    if tag not in [TagName.HEADER, TagName.MAIN, TagName.FOOTER]
)


def de_fence(text: str) -> str:
    """If response is fenced code block, remove fence"""
    stripped_text = text.strip()
    if stripped_text.startswith(("```json\n", "```\n", "``` json\n")):
        # Find the first newline after the opening fence
        first_newline = stripped_text.find('\n')
        if first_newline != -1:
            # Remove the opening fence
            stripped_text = stripped_text[first_newline + 1:]
        
        # Remove the closing fence if it exists
        if stripped_text.endswith("\n```"):
            stripped_text = stripped_text[:-4].rstrip()
    
    return stripped_text


def create_router(
    gemini_api_key: str, 
    openai_api_key: str, 
    deepseek_api_key: str,
    openrouter_api_key: str,
) -> Router:
    """Create a LiteLLM Router with advanced load balancing and fallback configuration."""
    model_list = [
        {
            "model_name": "html-parser",  # Using a common model_name for load balancing
            "litellm_params": {
                "model": "openrouter/google/gemini-2.5-flash",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 5,
                "weight": 3,
            }
        },
        {
            "model_name": "html-parser",
            "litellm_params": {
                "model": "openrouter/qwen/qwen3-coder:free",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 1,
                "weight": 3,
            }
        },
        {
            "model_name": "html-parser",
            "litellm_params": {
                "model": "openrouter/x-ai/grok-3-mini",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 5,
                "weight": 3,
            }
        },
        {
            "model_name": "html-parser",
            "litellm_params": {
                "model": "deepseek/deepseek-chat", 
                "api_key": deepseek_api_key,
                "max_parallel_requests": 15,
                "weight": 2,    # Secondary preference
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


def _create_nodes_from_blocks(blocks: list[ContentBlock]) -> list[StructuredNode]:
    """Fallback helper to create leaf StructuredNodes from a list of ContentBlocks
    when the LLM fails to propose a valid response or maximum depth is reached."""
    leaf_nodes: list[StructuredNode] = []
    for block in blocks:
        tag = TagName.IMG if block.block_type == BlockType.PICTURE else TagName.P
        leaf_nodes.append(
            StructuredNode(
                tag=tag,
                children=[],
                text=block.text_content,
                positional_data=[block.positional_data],
                description=block.description,
                storage_url=block.storage_url
            )
        )
    return leaf_nodes


async def _process_single_input(
    blocks: list[ContentBlock], 
    context: Context, 
    router: Router,
    depth: int = 0
) -> ParsedHTMLPartial | None:
    """Process a single input using LiteLLM Router with built-in concurrency control and fallbacks."""
    html_representation = "".join([block.to_html(block_id=i) for i, block in enumerate(blocks)])
    messages = [
        {
            "role": "user",
            "content": PROMPT.format(
                html_representation=html_representation,
                parent_tags=" -> ".join([f"{tag.value}" for tag in context.parent_tags]),
                parent_heading=context.parent_heading.value if context.parent_heading else "None",
                allowed_tags=ALLOWED_TAGS,
                response_schema=HTMLPartial.model_json_schema()
            )
        }
    ]
    
    # Router handles concurrency, retries, and fallbacks automatically
    # Only retry on validation errors, not API failures
    max_validation_attempts = 3
    for attempt in range(max_validation_attempts):
        try:
            # Use the router - it handles retries, fallbacks, and concurrency automatically
            response = await router.acompletion(
                model="html-parser",  # Use the common model name for load balancing
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                response_format={
                    "type": "json_object",
                    "response_schema": HTMLPartial.model_json_schema(),
                }
            )

            if (
                response
                and isinstance(response, ModelResponse)
                and isinstance(response.choices[0], Choices)
                and response.choices[0].message.content
            ):
                cost = completion_cost(completion_response=response)
                # Get actual model used from response
                actual_model = getattr(response, 'model', 'unknown')
                logger.debug(f"${float(cost):.10f} (using {actual_model})")
                
                html_partial = HTMLPartial.model_validate_json(
                    de_fence(response.choices[0].message.content)
                )
                parsed_html_partial = ParsedHTMLPartial(
                    proposed_nodes=[
                        ParsedProposedNode(
                            tag=proposed.tag,
                            children=parse_range_string(proposed.children) if proposed.children else [],
                            sources=parse_range_string(proposed.sources) if proposed.sources else [],
                            text=proposed.text,
                        ) for proposed in html_partial.proposed_nodes
                    ]
                )
                # Validate that the children and sources lists are valid to prevent breaking index errors
                invalid_child_id = next(
                    (id_num 
                    for node in parsed_html_partial.proposed_nodes 
                    for id_num in node.children
                    if id_num not in range(len(blocks))
                ), -1)
                if invalid_child_id != -1:
                    raise ValueError(f"Your response included an out of bounds child index: {invalid_child_id} (valid range: 0-{len(blocks)-1})")
                invalid_source_id = next(
                    (id_num 
                    for node in parsed_html_partial.proposed_nodes 
                    for id_num in node.sources
                    if id_num not in range(len(blocks))
                ), -1)
                if invalid_source_id != -1:
                    hint = " Remember to use *inclusive* ranges. E.g., 0-2 encompasses 0, 1, and 2. The last index is included in the range!"
                    raise ValueError(f"Your response included an out of bounds source index: {invalid_source_id} (valid range: 0-{len(blocks)-1}){hint if invalid_source_id + 1 == len(blocks) else ''}")
                
                # Log the validated response for review and fine-tuning
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "model": actual_model,
                    "cost": float(cost),
                    "depth": depth,
                    "context": context.model_dump(),
                    "input_blocks_count": len(blocks),
                    "messages": messages,
                    "raw_response": response.choices[0].message.content,
                    "parsed_response": parsed_html_partial.model_dump(),
                    "validation_attempt": attempt + 1
                }
                response_logger.info(json.dumps(log_entry, ensure_ascii=False))
                
                return parsed_html_partial
            else:
                logger.warning("No valid response from LLM")
                return None
        except (ValidationError, ValueError) as e:
            if attempt < max_validation_attempts - 1:
                logger.warning(f"Validation error (attempt {attempt+1}/{max_validation_attempts}): {e}")
                # Append error message and retry
                messages.append({
                    "role": "user",
                    "content": f"Your previous response had a validation error: {str(e)}. "
                               "Please correct your response to match the required schema and constraints."
                })
            else:
                logger.warning(f"Validation error on final attempt: {e}")
                return None
        except Exception as e:
            # LiteLLM Router already handled retries and fallbacks for API errors
            logger.error(f"Error processing input after all retries and fallbacks: {e}")
            return None
    return None


async def detect_nested_structure(
    blocks: list[ContentBlock],
    context: Context,
    router: Router,
    depth: int = 0,
    max_depth: int = 5,
) -> list[StructuredNode]:
    """Given a list of `ContentBlock` objects that represent the content of an HTML node,
    convert them to naive HTML, and call an LLM to propose a better structured HTML
    representation. Continue this process recursively until leaf nodes are reached.

    Args:
        blocks: List of `ContentBlock` objects that represent the content of an HTML node.
        context: Context for the LLM, including the parent headings and tags.
        router: LiteLLM Router instance with fallback configuration and built-in concurrency control.
        depth: Current depth of recursion.
        max_depth: Maximum depth to recurse.

    Returns:
        List of `StructuredNode` objects that represent the structured HTML representation of the node.
    """
    # Stop recursion if depth limit is reached
    if depth >= max_depth:
        # Fall back to creating leaf nodes from the ContentBlocks
        return _create_nodes_from_blocks(blocks)

    # Process the input with the LLM
    response: ParsedHTMLPartial | None = await _process_single_input(blocks, context, router, depth)
    nodes = []
    # Stop recursion if LLM response is invalid
    if response is None:
        # Fall back to creating leaf nodes from the ContentBlocks
        return _create_nodes_from_blocks(blocks)
    else:
        html_response: ParsedHTMLPartial = response

        # 1. Launch recursive tasks for each child-bearing proposed node
        child_tasks: dict[int, asyncio.Task[list[StructuredNode]]] = {}
        for i, proposed in enumerate(html_response.proposed_nodes):
            if proposed.children:               
                # Determine heading context for this branch
                parent_heading: TagName | None = next(
                    (
                        n.tag for n in html_response.proposed_nodes[:i]
                        if n.tag in {TagName.H1, TagName.H2, TagName.H3, TagName.H4, TagName.H5, TagName.H6}
                    ),
                    None
                )

                child_context = Context(
                    parent_tags=context.parent_tags + [proposed.tag],
                    parent_heading=parent_heading or context.parent_heading,
                )

                child_tasks[i] = asyncio.create_task(
                    detect_nested_structure(
                        [blocks[j] for j in proposed.children],
                        context=child_context,
                        router=router,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                )

        # 2. Await all child tasks in parallel (Router handles concurrency control automatically)
        if child_tasks:
            resolved_lists = await asyncio.gather(*child_tasks.values())
            children_results = {
                idx: child_list for idx, child_list in zip(child_tasks.keys(), resolved_lists)
            }
        else:
            children_results = {}

        # 3. Assemble StructuredNodes in original order, inserting resolved children
        for idx, proposed in enumerate(html_response.proposed_nodes):
            children = children_results.get(idx, [])
            nodes.append(
                StructuredNode(
                    tag=proposed.tag,
                    children=children,
                    text=proposed.text,
                    # TODO: introduce a helper to merge same-page bboxes; also use children's positional data
                    positional_data=[blocks[j].positional_data for j in proposed.sources],
                    description=blocks[proposed.sources[0]].description if proposed.sources else None,
                    storage_url=blocks[proposed.sources[0]].storage_url if proposed.sources else None,
                )
            )
    # Base case: all nodes are leaves or all children have been processed
    return nodes


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

        tasks = [
            detect_nested_structure(
                blocks,
                context=Context(parent_tags=[tag_name], parent_heading=None),
                router=router,
                max_depth=7,
            )
            for tag_name, blocks in top_level_structure
        ]

        children_lists = await asyncio.gather(*tasks)

        nested_structure: list[StructuredNode] = []
        for (tag_name, blocks), children_list in zip(top_level_structure, children_lists):
            nested_structure.append(
                StructuredNode(
                    tag=tag_name,
                    children=children_list,
                    positional_data=[
                        PositionalData(
                            page_pdf=page_number,
                            bbox=page_dimensions[page_number - 1],
                        )
                        for page_number in {block.positional_data.page_pdf for block in blocks}
                    ],
                )
            )

        with open(os.path.join("artifacts", "doc_601_nested_structure.json"), "w") as fw:
            json.dump([node.model_dump() for node in nested_structure], fw, indent=2)

    start_time = time.time()
    asyncio.run(_test_main())
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time} seconds")