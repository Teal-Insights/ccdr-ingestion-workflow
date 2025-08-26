import logging
import asyncio
import os
from litellm import Router, completion_cost
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices
from typing import Optional, Iterable
from pydantic import BaseModel, Field, ValidationError
from bs4 import BeautifulSoup, Tag

from utils.range_parser import parse_range_string
from utils.html import ALLOWED_TAGS as ALLOWED_TAGS_LIST
from utils.html import validate_html_tags
from utils.html import validate_sources_and_children
from utils.schema import TagName
from utils.json import de_fence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Context(BaseModel):
    parent_tags: list[TagName] = Field(default_factory=list, description="Parent tags of the current node")
    parent_heading: Optional[TagName] = Field(default=None, description="Parent heading of the current node")


class HTMLStringResponse(BaseModel):
    html_str: str = Field(description="HTML payload")

TOP_LEVEL_TAGS_LIST: list[str] = ["header", "main", "footer"]
TOP_LEVEL_TAGS: str = ", ".join(TOP_LEVEL_TAGS_LIST)

# Below this length, prefer a no-deferral response to fully resolve content in one pass
NO_DEFERRAL_CHAR_THRESHOLD = 15000

STANDARD_PROMPT = """# Task

You are restructuring a flat HTML partial (with `id` attributes) into a semantic hierarchy.
Due to the length of the input HTML partial, you may mark nodes with `data-children` to defer work to a later pass.

# Requirements:

- Use ONLY these tags: {allowed_tags}.
- Leaves MUST have `data-sources` with comma-separated inclusive ranges of ids from the given input.
- Structural containers that defer work MUST be empty and have `data-children` with comma-separated inclusive ranges of ids from the given input.
- `data-children` and `data-sources` must be disjoint, comprehensively covering all ids from the given input.
- `data-sources` may be reused (e.g., if a single node is split into multiple nodes, the same data-source may be used for each node)
- Preserve exact text and image src/alt; you may fix whitespace and encoding.
- For purposes of this exercise, in-line styling tags like `b`, `i`, `u`, `s`, `sup`, and `sub` are considered text content, not nodes of their own.
- You may fix up messy, malformed, or duplicate styling tags or remove them in cases where they don't make sense.

# Response format

- Return a single JSON object, no prose, matching this schema:
```json
{response_schema}
```
- Return your semantically restructured HTML partial as the value of the `html_str` field.

# Example

Suppose you receive an HTML partial with context main -> section and ids 300-321, like this:

```html
<!-- Assume 18 p tags for Section 4.4 with ids 300-317 here -->
<p id="318">4.5  Macroeconomic Impact of Critical Short-Term Investments</p>
<p id="319">In the short-term (2024—26) and under any scenario, Lebanon urgently needs investment in key service and growth-providing sectors, with a financing envelope of US$770 million (Table 6) . Although Lebanon could embark on either of the two scenarios, the reality could be somewhere in between. Consequently, the Lebanon CCDR assessed the impact of an urgent financing envelope that responds to key (partial) needs in the four sectors covered in the CCDR (energy, water, transport, solid waste) (Table 6) . In view of the severe lack of service provision in the energy and water sectors, these two sectors will receive most of the financing (~39 percent and 34 percent of the total investment envelope, respectively). In the energy sector, a priority would be to increase the low-cost supply (mix adequacy) and the readiness of the grid and EDL for high penetration of variable renewable electricity (grid reliability, operational and commercial efficiency). The transport sector would receive about 15.5 percent of total financing and the water sector approximately 11.6 percent. It is assumed that 80 percent of the financing will be nonconcessional and 80 percent would consist of capital expenditures.</p>
<p id="320">Table 6: Short-Term Priority Investment Envelope and <b>Financing</b> Assumptions</p>
<p id="321"><b>Sector</b> <b>2024</b> <b>2025</b> <b>2026</b> <b><b>Total</b></b> <b>Financing</b> <b>Imports</b> <b><b>Capital</b> stock</b> <b>Noncon-</b> <b>cessional</b> <b><b>Financing</b> <b><b>(%)</b></b></b> <b>Con<b>cessional</b></b> <b><b>Financing</b> and</b> <b>grants <b><b>(%)</b></b></b> <b><b>(%)</b></b> <b>Domestic</b> <b>spending</b> <b><b>(%)</b></b> <b>Capital</b> <b><b>expendi-</b></b> <b><b>tures <b><b>(%)</b></b></b></b> <b>Operating</b> <b><b>expendi-</b></b> <b><b>tures <b><b>(%)</b></b></b></b> <b>Energy</b> 100 100 100 <b>300</b> 80 20 80 20 80 20 <b>Water</b> 80 80 100 <b>260</b> 80 20 50 50 95 5 <b>Transport</b> 40 40 40 <b>120</b> 80 20 20 80 <b>90</b> 10 <b>Solid waste</b> 30 30 30 <b>90</b> 80 20 80 20 80 20 <b><b>Total</b></b> <b>770</b></p>
```

To avoid a long output, you might opt to defer processiong of Section 4.4 to a later pass. You would create an empty section with `data-children` set to the inclusive range of that section's ids, and focus your efforts on populating Section 4.5 for now. You would then return this HTML partial in the "html_str" field of your JSON response:

```html
<section data-children="300-317">
</section>
<section data-sources="318-321">
    <h2 data-sources="318">4.5 Macroeconomic Impact of Critical Short-Term Investments</h2>
    <p data-sources="319">In the short-term (2024—26) and under any scenario, Lebanon urgently needs investment in key service and growth-providing sectors, with a financing envelope of US$770 million.</p>
    <table data-sources="321">
        <caption data-sources="320">Table 6: Short-Term Priority Investment Envelope and Financing Assumptions</caption>
        <thead data-sources="321">
            <tr data-sources="321">
                <th data-sources="321">Sector</th>
                <th data-sources="321">2024</th>
                <th data-sources="321">2025</th>
                <th data-sources="321">2026</th>
                <th data-sources="321">Total</th>
                <th data-sources="321">Financing (%)</th>
                <th data-sources="321">Capital expenditures (%)</th>
                <th data-sources="321">Operating expenditures (%)</th>
            </tr>
        </thead>
        <tbody data-sources="321">
            <tr data-sources="321">
                <td data-sources="321">Energy</td>
                <td data-sources="321">100</td>
                <td data-sources="321">100</td>
                <td data-sources="321">100</td>
                <td data-sources="321">300</td>
                <td data-sources="321">80</td>
                <td data-sources="321">80</td>
                <td data-sources="321">20</td>
            </tr>
            <tr data-sources="321">
                <td data-sources="321">Water</td>
                <td data-sources="321">80</td>
                <td data-sources="321">80</td>
                <td data-sources="321">100</td>
                <td data-sources="321">260</td>
                <td data-sources="321">80</td>
                <td data-sources="321">95</td>
                <td data-sources="321">5</td>
            </tr>
            <tr data-sources="321">
                <td data-sources="321">Transport</td>
                <td data-sources="321">40</td>
                <td data-sources="321">40</td>
                <td data-sources="321">120</td>
                <td data-sources="321">80</td>
                <td data-sources="321">90</td>
                <td data-sources="321">10</td>
            </tr>
            <tr data-sources="321">
                <td data-sources="321">Solid waste</td>
                <td data-sources="321">30</td>
                <td data-sources="321">30</td>
                <td data-sources="321">90</td>
                <td data-sources="321">80</td>
                <td data-sources="321">80</td>
                <td data-sources="321">20</td>
            </tr>
            <tr data-sources="321">
                <td data-sources="321"><b>Total</b></td>
                <td data-sources="321" colspan="3"></td>
                <td data-sources="321"><b>770</b></td>
                <td data-sources="321" colspan="3"></td>
            </tr>
        </tbody>
    </table>
</section>
```

# Input

Input ids information:

- Allowed ids: {all_ids}
- Min id: {min_id}
- Max id: {max_id}

This partial is part of a larger document. Here is some information about the context within which it is nested:

- Parent tags: {parent_tags}
- Parent heading: {parent_heading}

Select heading levels and structural tags with this context in mind.

Input HTML partial:

```html
{html_representation}
```
"""

TOP_LEVEL_PROMPT = """# Task

You are preparing the top-level structure for a document.

# Requirements:

- Output MUST contain only these top-level tags: `header`, `main`, and/or `footer` (one or more).
    - `header`: Front matter (title pages, table of contents, preface, etc.)
    - `main`: Body content (main chapters, sections, core content)
    - `footer`: Back matter (appendices, bibliography, index, etc.)
- Each top-level tag MUST be empty (no children) and MUST have a data-children attribute. (The tags will be populated by another worker in a subsequent pass.)
- The `data-children` attribute MUST have a value that is a comma-separated list of inclusive ranges of ids from the given input.
- There might be a few tags at the seams where we're unsure which section they belong to. Perhaps a decorative image precedes the main content. We make a judgment call as to which section it belongs to.
- Generally the input elements will already be in the correct reading order and `data-children` will be a single range, but occasionally we might encounter an edge case where it is appropriate to use a comma-separated range to resequence a few elements at the seams. For instance, `main` might have `data-children="39-86,88"`, and `footer` might have `data-children="87,89-100"`.

# Response format

- Return a single JSON object matching this schema:
```json
{response_schema}
```

Place your HTML inside the `html_str` field as a string.

# Example

Suppose you receive a flat HTML partial with ids 0-292. The ids 0-26 contain a title page, table of contents, and list of abbreviations. The ids 27-251 contain several chapters of text, with end notes. The ids 252-292 contain a bibliography.

You should return:

```html
<header data-children="0-26"></header>
<main data-children="27-251"></main>
<footer data-children="252-292"></footer>
```

# Input

Input ids information:

- Allowed ids: {all_ids}
- Min id: {min_id}
- Max id: {max_id}

Input HTML:

```html
{html_representation}
```
"""

NO_DEFERRAL_PROMPT = """You are an expert in HTML and document structure.
You are given an HTML partial with a flat structure and only `p` and `img` tags.
Your task is to propose a better structured HTML representation of the content, with semantic tags and logical hierarchy.

# Requirements

- You may only use the following tags: {allowed_tags}.
- Styling is not in your purview, and styles in your output will be ignored.
- You may split, merge, or replace structural containers as necessary, but you should make an effort to:
    - Clean up any whitespace, encoding, redundant style tags, or other formatting issues
    - Otherwise maintain the identical wording/spelling of the text content and of image descriptions and source URLs
- All leaf elements MUST have a `data-sources` attribute with a comma-separated list of ids of the source elements. This can include inclusive ranges, e.g., `data-sources="0-5,7,9-12"`
    - Note: inline style tags `b`, `i`, `u`, `s`, `sup`, `sub`, and `br` do not need `data-sources`; we treat them as flat text content and exclude them from DOM parsing for the purposes of this exercise
    - When using ranges, both endpoints MUST exist in the input. If necessary, split into multiple valid ranges (e.g., `0-3,5,7-9`).
    - `data-sources` must comprehensively cover all ids from the given input to pass validation. (If an input element is empty or garbage, you may attach its id as a data source to a neighboring element in the output.)
- Preserve exact text and image src/alt; you may fix whitespace and encoding.
- You may fix up messy, malformed, or duplicate styling tags or remove them in cases where they don't make sense.

# Example

Suppose you receive this HTML partial with context main -> section -> section and ids 318-321:

```html
<p id="318">4.5  Macroeconomic Impact of Critical Short-Term Investments</p>
<p id="319">In the short-term (2024—26) and under any scenario, Lebanon urgently needs investment in key service and growth-providing sectors, with a financing envelope of US$770 million (Table 6) . Although Lebanon could embark on either of the two scenarios, the reality could be somewhere in between. Consequently, the Lebanon CCDR assessed the impact of an urgent financing envelope that responds to key (partial) needs in the four sectors covered in the CCDR (energy, water, transport, solid waste) (Table 6) . In view of the severe lack of service provision in the energy and water sectors, these two sectors will receive most of the financing (~39 percent and 34 percent of the total investment envelope, respectively). In the energy sector, a priority would be to increase the low-cost supply (mix adequacy) and the readiness of the grid and EDL for high penetration of variable renewable electricity (grid reliability, operational and commercial efficiency). The transport sector would receive about 15.5 percent of total financing and the water sector approximately 11.6 percent. It is assumed that 80 percent of the financing will be nonconcessional and 80 percent would consist of capital expenditures.</p>
<p id="320">Table 6: Short-Term Priority Investment Envelope and <b>Financing</b> Assumptions</p>
<p id="321"><b>Sector</b> <b>2024</b> <b>2025</b> <b>2026</b> <b><b>Total</b></b> <b>Financing</b> <b>Imports</b> <b><b>Capital</b> stock</b> <b>Noncon-</b> <b>cessional</b> <b><b>Financing</b> <b><b>(%)</b></b></b> <b>Con<b>cessional</b></b> <b><b>Financing</b> and</b> <b>grants <b><b>(%)</b></b></b> <b><b>(%)</b></b> <b>Domestic</b> <b>spending</b> <b><b>(%)</b></b> <b>Capital</b> <b><b>expendi-</b></b> <b><b>tures <b><b>(%)</b></b></b></b> <b>Operating</b> <b><b>expendi-</b></b> <b><b>tures <b><b>(%)</b></b></b></b> <b>Energy</b> 100 100 100 <b>300</b> 80 20 80 20 80 20 <b>Water</b> 80 80 100 <b>260</b> 80 20 50 50 95 5 <b>Transport</b> 40 40 40 <b>120</b> 80 20 20 80 <b>90</b> 10 <b>Solid waste</b> 30 30 30 <b>90</b> 80 20 80 20 80 20 <b><b>Total</b></b> <b>770</b></p>
```

You might return this HTML partial in the "html_str" field:

```html
<h2 data-sources="318">4.5 Macroeconomic Impact of Critical Short-Term Investments</h2>
<p data-sources="319">In the short-term (2024—26) and under any scenario, Lebanon urgently needs investment in key service and growth-providing sectors, with a financing envelope of US$770 million.</p>
<table data-sources="321">
    <caption data-sources="320">Table 6: Short-Term Priority Investment Envelope and Financing Assumptions</caption>
    <thead data-sources="321">
        <tr data-sources="321">
            <th data-sources="321">Sector</th>
            <th data-sources="321">2024</th>
            <th data-sources="321">2025</th>
            <th data-sources="321">2026</th>
            <th data-sources="321">Total</th>
            <th data-sources="321">Financing (%)</th>
            <th data-sources="321">Capital expenditures (%)</th>
            <th data-sources="321">Operating expenditures (%)</th>
        </tr>
    </thead>
    <tbody data-sources="321">
        <tr data-sources="321">
            <td data-sources="321">Energy</td>
            <td data-sources="321">100</td>
            <td data-sources="321">100</td>
            <td data-sources="321">100</td>
            <td data-sources="321">300</td>
            <td data-sources="321">80</td>
            <td data-sources="321">80</td>
            <td data-sources="321">20</td>
        </tr>
        <tr data-sources="321">
            <td data-sources="321">Water</td>
            <td data-sources="321">80</td>
            <td data-sources="321">80</td>
            <td data-sources="321">100</td>
            <td data-sources="321">260</td>
            <td data-sources="321">80</td>
            <td data-sources="321">95</td>
            <td data-sources="321">5</td>
        </tr>
        <tr data-sources="321">
            <td data-sources="321">Transport</td>
            <td data-sources="321">40</td>
            <td data-sources="321">40</td>
            <td data-sources="321">120</td>
            <td data-sources="321">80</td>
            <td data-sources="321">90</td>
            <td data-sources="321">10</td>
        </tr>
        <tr data-sources="321">
            <td data-sources="321">Solid waste</td>
            <td data-sources="321">30</td>
            <td data-sources="321">30</td>
            <td data-sources="321">90</td>
            <td data-sources="321">80</td>
            <td data-sources="321">80</td>
            <td data-sources="321">20</td>
        </tr>
        <tr data-sources="321">
            <td data-sources="321"><b>Total</b></td>
            <td data-sources="321" colspan="3"></td>
            <td data-sources="321"><b>770</b></td>
            <td data-sources="321" colspan="3"></td>
        </tr>
    </tbody>
</table>
```

# Response format

- Return a single JSON object, no prose, matching this schema:
```json
{response_schema}
```
- Return your semantically restructured HTML partial as the value of the `html_str` field.

# Input

Input ids information:

- Allowed ids: {all_ids}
- Min id: {min_id}
- Max id: {max_id}

This partial is part of a larger document. Here is some information about the context within which it is nested:

- Parent tags: {parent_tags}
- Parent heading: {parent_heading}

Select heading levels and structural tags with the context in mind.

Input HTML partial:

```html
{html_representation}
```
"""


# Template for fixup prompts
FIXUP_TEMPLATE = """
You previously produced HTML that failed validation. Please repair your previous HTML to comply with constraints.

Validation messages:
{messages}

Here is the original prompt you were responding to (context only; do not change its instructions):

````markdown
{original_prompt}
````

Here is your previous HTML output to repair:

```html
{previous_output}
```
"""


def subset_flat_html_by_ids(flat_html: str, ids: Iterable[int]) -> str:
    """Extract a flat subset HTML containing only elements with the given ids, in the given order.

    Assumes the flat HTML contains block-level elements with id attributes that are integers.
    """
    soup = BeautifulSoup(flat_html, "html.parser")
    subset_parts: list[str] = []
    for id_num in ids:
        el = soup.find(attrs={"id": str(id_num)})
        if isinstance(el, Tag):
            subset_parts.append(str(el))
    return "".join(subset_parts)


def create_router(
    gemini_api_key: str, 
    openai_api_key: str, 
    deepseek_api_key: str,
    openrouter_api_key: str,
) -> Router:
    """Create a LiteLLM Router with advanced load balancing and fallback configuration."""
    model_list = [
        {
            "model_name": "html-parser-fallback",
            "litellm_params": {
                "model": "gemini/gemini-2.5-flash",
                "api_key": gemini_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        {
            "model_name": "html-parser-fallback",
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-chat",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        {
            "model_name": "html-parser-fallback",
            "litellm_params": {
                "model": "openrouter/anthropic/claude-sonnet-4",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 10,
                "weight": 3,
            }
        },
        {
            "model_name": "html-parser",
            "litellm_params": {
                "model": "openrouter/openai/gpt-5-mini",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        {
            "model_name": "html-parser",
            "litellm_params": {
                "model": "gemini/gemini-2.5-pro",
                "api_key": gemini_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
    ]
    
    # Router configuration
    return Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",  # Weighted random selection
        fallbacks=[{"html-parser": ["html-parser-fallback"]}],
        num_retries=2,
        allowed_fails=5,
        cooldown_time=30,
        enable_pre_call_checks=True,  # Enable context window and rate limit checks
        default_max_parallel_requests=50,  # Global default
        set_verbose=False,  # Set to True for debugging
    )


async def _call_llm_for_html(
    content: str,
    context: Context,
    router: Router,
) -> str | None:
    """Call the LLM to get a single HTML payload obeying data-children/data-sources contract."""

    messages = [{"role": "user", "content": content}]

    max_validation_attempts = 3
    for attempt in range(max_validation_attempts):
        try:
            response = await router.acompletion(
                model="html-parser",
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                response_format=HTMLStringResponse,
            )

            if (
                response
                and isinstance(response, ModelResponse)
                and isinstance(response.choices[0], Choices)
                and response.choices[0].message.content
            ):
                cost = completion_cost(completion_response=response)
                actual_model = getattr(response, 'model', 'unknown')
                logger.debug(f"${float(cost):.10f} (using {actual_model})")

                payload = de_fence(response.choices[0].message.content, type="json")
                html_resp = HTMLStringResponse.model_validate_json(payload)
                return html_resp.html_str
            else:
                logger.warning("No valid response from LLM")
                return None
        except (ValidationError, ValueError) as e:
            if attempt < max_validation_attempts - 1:
                logger.warning(f"Validation error (attempt {attempt+1}/{max_validation_attempts}): {e}")
                messages.append({
                    "role": "user",
                    "content": f"Your previous response had a validation error: {str(e)}. Please return ONLY one HTML payload per the contract."
                })
            else:
                logger.warning(f"Validation error on final attempt: {e}")
                return None
        except Exception as e:
            logger.error(f"Error processing input after all retries and fallbacks: {e}")
            return None
    return None


def _top_level_data_children_nodes(soup: BeautifulSoup) -> list[Tag]:
    """Return nodes with data-children that do not have an ancestor with data-children.

    This prevents double-processing nested containers in the same pass.
    """
    candidates = soup.find_all(attrs={"data-children": True})
    top_level: list[Tag] = []
    for node in candidates:
        has_ancestor = any(parent.has_attr("data-children") for parent in node.parents if isinstance(parent, Tag))
        if not has_ancestor:
            top_level.append(node)  # type: ignore[arg-type]
    return top_level


async def detect_nested_structure(
    flat_html: str,
    context: Context,
    router: Router,
    depth: int = 0,
    max_depth: int = 5,
) -> str:
    """Recursively restructure flat HTML using an LLM.

    At each level, the model returns a mix of:
    - leaf nodes populated with data-sources
    - empty structural nodes carrying data-children to defer expansion
    """
    if depth >= max_depth:
        # Pretty-print only at the top level to avoid altering intermediate recursion
        if depth == 0:
            try:
                return BeautifulSoup(flat_html, "html.parser").prettify(formatter="minimal")
            except Exception:
                return flat_html
        return flat_html

    is_top_level = (not context.parent_tags) and (context.parent_heading is None)
    if is_top_level:
        # Extract all ids for disjoint coverage requirement
        soup = BeautifulSoup(flat_html, "html.parser")
        all_ids: list[int] = []
        for el in soup.find_all(attrs={"id": True}):
            try:
                all_ids.append(int(el.get("id", "")))
            except Exception:
                continue
        content = TOP_LEVEL_PROMPT.format(
            html_representation=flat_html,
            all_ids=", ".join(str(x) for x in all_ids),
            min_id=min(all_ids) if all_ids else 0,
            max_id=max(all_ids) if all_ids else -1,
            response_schema=HTMLStringResponse.model_json_schema(),
        )
    else:
        # Size-based routing: choose no-deferral if small, otherwise standard
        # Extract all ids for disjoint coverage requirement in this subset
        subset_soup = BeautifulSoup(flat_html, "html.parser")
        subset_all_ids: list[int] = []
        for el in subset_soup.find_all(attrs={"id": True}):
            try:
                subset_all_ids.append(int(el.get("id", "")))
            except Exception:
                continue

        # Compute allowed tags string for non-top-level passes (exclude top-level tags)
        allowed_non_top_level = ", ".join([t for t in ALLOWED_TAGS_LIST if t not in TOP_LEVEL_TAGS_LIST])

        if len(flat_html) < 15000:
            content = NO_DEFERRAL_PROMPT.format(
                html_representation=flat_html,
                parent_tags=" -> ".join([f"{tag.value}" for tag in context.parent_tags]) or "None",
                parent_heading=context.parent_heading.value if context.parent_heading else "None",
                allowed_tags=allowed_non_top_level,
                response_schema=HTMLStringResponse.model_json_schema(),
                all_ids=", ".join(str(x) for x in subset_all_ids),
                min_id=min(subset_all_ids) if subset_all_ids else 0,
                max_id=max(subset_all_ids) if subset_all_ids else -1,
            )
        else:
            content = STANDARD_PROMPT.format(
                html_representation=flat_html,
                parent_tags=" -> ".join([f"{tag.value}" for tag in context.parent_tags]) or "None",
                parent_heading=context.parent_heading.value if context.parent_heading else "None",
                allowed_tags=allowed_non_top_level,
                response_schema=HTMLStringResponse.model_json_schema(),
                all_ids=", ".join(str(x) for x in subset_all_ids),
                min_id=min(subset_all_ids) if subset_all_ids else 0,
                max_id=max(subset_all_ids) if subset_all_ids else -1,
            )

    proposed_html = await _call_llm_for_html(content, context, router)
    if not isinstance(proposed_html, str) or not proposed_html.strip():
        return flat_html

    # TODO: Add data-sources and data-children validation
    # TODO: Loop until valid, splitting at ````markdown and ```` to extract and populate template with the original prompt on each pass
    # Validate tags and perform a single fixup retry if needed
    is_valid, invalid_tags = (
        validate_html_tags(proposed_html, exclude=TOP_LEVEL_TAGS_LIST)
        if not is_top_level
        else validate_html_tags(proposed_html)
    )

    allowed_tags_str = (
        TOP_LEVEL_TAGS if is_top_level else ", ".join([t for t in ALLOWED_TAGS_LIST if t not in TOP_LEVEL_TAGS_LIST])
    )
    messages_payload: str = ""
    if not is_valid:
        messages_payload += "\n- HTML would not parse. Please ensure that the HTML is valid, with matched opening and closing tags."
    if invalid_tags:
        messages_payload += f"\n- Disallowed tags: {', '.join(sorted(invalid_tags))}. Remove these tags. Allowed tags at this level: {allowed_tags_str}."

    # Structural validation for data-sources/data-children (non-top-level only)
    struct_issues_count_before = 0
    struct_issues_messages = ""
    if not is_top_level:
        struct_ok, struct_issues, _struct_meta = validate_sources_and_children(flat_html, proposed_html)
        struct_issues_count_before = len(struct_issues)
        if not struct_ok and struct_issues:
            for issue in struct_issues:
                # Each issue adds a concise bullet line
                struct_issues_messages += f"\n- {issue.message}"
            messages_payload += struct_issues_messages

    if (not is_valid) or invalid_tags or (not is_top_level and struct_issues_count_before > 0):
        fixup_content = FIXUP_TEMPLATE.format(
            messages=messages_payload,
            original_prompt=content,
            previous_output=proposed_html,
        )
        fixed_html = await _call_llm_for_html(fixup_content, context, router)
        if isinstance(fixed_html, str) and fixed_html.strip():
            # Re-validate once; if still invalid, proceed with whatever we have
            is_valid_after, invalid_after = (
                validate_html_tags(fixed_html, exclude=TOP_LEVEL_TAGS_LIST)
                if not is_top_level
                else validate_html_tags(fixed_html)
            )
            struct_improved = False
            if not is_top_level:
                struct_ok_after, struct_issues_after, _ = validate_sources_and_children(flat_html, fixed_html)
                # Consider improved if violations decreased or now valid
                if struct_ok_after or len(struct_issues_after) < struct_issues_count_before:
                    struct_improved = True
            # Accept fixed output if tag validation improved or structural validation improved
            if (
                is_valid_after and not is_valid
            ) or (
                len(invalid_after) < len(invalid_tags)
            ) or struct_improved:
                proposed_html = fixed_html

    soup = BeautifulSoup(proposed_html, "html.parser")

    # Gather top-level containers to expand
    targets = _top_level_data_children_nodes(soup)
    if not targets:
        # Pretty-print only at the top level
        if depth == 0:
            try:
                return soup.prettify(formatter="minimal")
            except Exception:
                return str(soup)
        return str(soup)

    tasks: list[asyncio.Task[str]] = []
    for node in targets:
        ids = parse_range_string(node.get("data-children", ""))
        subset = subset_flat_html_by_ids(flat_html, ids)

        # Derive child context
        try:
            node_tag = TagName(node.name.lower())
        except Exception:
            node_tag = None  # type: ignore[assignment]

        child_context = Context(
            parent_tags=context.parent_tags + ([node_tag] if node_tag else []),
            parent_heading=context.parent_heading,
        )

        tasks.append(asyncio.create_task(
            detect_nested_structure(
                subset,
                context=child_context,
                router=router,
                depth=depth + 1,
                max_depth=max_depth,
            )
        ))

    # Resolve all children in parallel
    results = await asyncio.gather(*tasks)
    for node, resolved_html in zip(targets, results):
        # Preserve original data-children to convert into data-sources after population
        original_data_children = node.get("data-children", None)
        node.clear()
        # Safely append children from the resolved HTML (avoid inserting a BeautifulSoup doc directly)
        # Fallback to the original subset if the resolved HTML is empty
        fallback_subset = subset_flat_html_by_ids(flat_html, parse_range_string(original_data_children or ""))
        parsed = BeautifulSoup(resolved_html.strip() or fallback_subset, "html.parser")
        for child in list(parsed.contents):
            node.append(child)
        # Replace data-children with identically valued data-sources for traceability
        if original_data_children:
            node["data-sources"] = original_data_children
        if node.has_attr("data-children"):
            del node["data-children"]

    # Pretty-print only at the top level
    if depth == 0:
        try:
            return soup.prettify(formatter="minimal")
        except Exception:
            return str(soup)
    return str(soup)


if __name__ == "__main__":
    import time
    import dotenv

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
        router: Router = create_router(
            gemini_api_key, 
            openai_api_key, 
            deepseek_api_key,
            openrouter_api_key,
        )

        with open(os.path.join("artifacts", "doc_601", "input.html"), "r") as fr:
            html_content: str = fr.read()

        nested_structure: str = await detect_nested_structure(
            html_content,
            context=Context(),
            router=router,
            max_depth=7,
        )

        from utils.html import pretty_print_html
        with open(os.path.join("artifacts", "doc_601_nested_structure_recursed.html"), "w") as fw:
            fw.write(pretty_print_html(nested_structure))

    start_time = time.time()
    asyncio.run(_test_main())
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time} seconds")