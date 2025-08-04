# TODO: For images, get the text content of sibling caption or figcaption block, if any, and assign to caption field

import logging
import asyncio
import json
import os
import re
from datetime import datetime
from typing import Literal, Optional
from litellm import Router
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices
from pydantic import BaseModel, Field, ValidationError, field_validator

from transform.detect_top_level_structure import parse_range_string
from transform.models import ContentBlock, StructuredNode
from utils.schema import TagName, PositionalData, BoundingBox
from utils.html import create_nodes_from_html
from utils.json import de_fence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ALLOWED_TAGS: str = ", ".join(
    tag.value for tag in TagName
    if tag not in [TagName.HEADER, TagName.MAIN, TagName.FOOTER]
) + ", b, i, u, s, sup, sub, br"

MODEL_TOKEN_LIMIT: int = 64000
# assume 4 chars per token, allow ~1/3 margin of error
MAX_CHUNK_CHAR_SIZE: int = ((MODEL_TOKEN_LIMIT * 4) * 2) // 3


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
    - Assign elements a `data-sources` attribute with a comma-separated list of ids of the source elements (for attribute mapping and content validation). This can include ranges, e.g., `data-sources="0-5,7,9-12"`
        - Note: inline style tags `b`, `i`, `u`, `s`, `sup`, `sub`, and `br` do not need `data-sources`, but all other tags should have this attribute

# Response format

Return the full revised HTML content in the following JSON format:

```json
{response_schema}
```

# Example

Here is an information-dense example illustrating some key transformations:

1.  **Nesting and Logical Headings:** Creating a hierarchical structure with `<section>`, `<h2>`, and `<h3>` tags based on the input's numbering and context.
2.  **Aggregation:** Combining multiple separate `<p>` tags from the input into a single, semantically correct `<table>` element in the output. The container `<table>` gets an aggregated `data-sources` attribute.
3.  **Splitting:** Deconstructing a single, complex `<p>` tag (which contains multiple headings, lists, and tabular data mashed together) into multiple structured elements in the output (`<h3>`, `<ul>`, `<table>`). Each of the new elements correctly inherits the `data-sources` from the original `<p>` tag.
4.  **List and Table Creation:** Identifying implicit lists and tables within the input text and formatting them correctly in the output.
5.  **Cleaning:** Removing redundant tags, whitespace, and other formatting issues while maintaining the original text content.

## Example input

```html
<p id="141">The persistence of weak governance, compounded by the economic crisis, undermines the capacity of the water sector to respond to the impacts of climate change. The water sector faces significant governance challenges that impede effective resource management and delivery of sustainable water services. Despite adoption of integrated water resource management principles, progress is considerably slower than the regional average. Complex legal frameworks and regulations have resulted in a convoluted institutional structure, diminishing accountability and transparency. Responsibilities for water management are fragmented across ministries and administrative levels, leading to overlap of responsibilities and lack of accountability. Coordination among stakeholders is deficient, and there is a notable absence of reliable data for informed planning, strategy development, policy formulation, and investment prioritization. Prolongation</p>
<p id="142">of the crisis will exacerbate the sector’s limitations by preventing new investments in basic service delivery and the storage capacity needed to build resilience to projected climate change in the long term. The renewable water resource diminished from more than 2,000 cubic meters per capita in the 1960s to less than 660 cubic meters per capita in 2020. The “Muddling through” scenario in the water sector will have serious consequences that are highlighted in Table 3 .</p>
<p id="143">Table 3</p>
<p id="144"><b>Characteristics</b></p>
<p id="145"><b>Impacts</b></p>
<p id="146">•  Limited infrastructure to store water and bridge the seasonal gap in the availability of water.</p>
<p id="147">•  The operations of most of the country’s about 75 wastewater treatment continue to be hindered by the high costs of energy and limited sewerage networks connectivity.</p>
<p id="148">•  The institutional capacity for building resilience to climate change is limited at the level of the water utility and the ministry.</p>
<p id="149">•  Tariffs are too low to cover operation and maintenance costs for existing infrastructure, address the high share of non-revenue water, the brain drain of human capital and lack of technical capabilities.</p>
<p id="150">•  The continuous brain drains of the human capital within the Water Establishments.</p>
<p id="151">•  More than 70 percent of the population face critical water shortages and an increasing cost of supply.</p>
<p id="152">•  Increases in the cost of water for the extremely income vulnerable groups.</p>
<p id="153">•  Poor quality of services undermines the willingness of the population to pay, while the increase in the cost of basic goods and services means that many are unable to pay.</p>
<p id="154">• Increased prevalence of, and inability to control, water borne diseases, as seen with the outbreak of cholera in October 2022</p>
<p id="195">3.5  Recommendations and Investment Needs</p>
<p id="196">Amid a deepening economic crisis, Lebanon has almost no fiscal space, limited institutional capacity, and numerous development challenges; therefore, it is critical to prioritize and sequence recommended measures  and  interventions—reflecting  their  urgency,  synergies,  and  trade-offs— in responding to development and climate needs. The recommended measures of this CCDR, responding to the above- <b>HIGH</b>lighted development and climate needs, are shown in Figure 12 . The recommended measures of this CCDR, responding to the above-<b>HIGH</b>lighted development and climate needs, are shown in Figure 12 .</p>
<p id="197">Figure 12: Country Climate and Development Report Recommended Policies and Interventions</p>
<p id="198"><b><b>HIGH</b> IMPACT AND RECOVERY-DRIVING INVESTMENTS</b> Urgent interventions with <b>LOW</b> tradeoffs Expand capacity of cleaner and affordable renewable energy sources and switch from liquid fuel to natural gas Increase water sector resilience by rehabilitating/expand water supply network, reducing non-revenue water, water supply reservoir, treatment plants, sewerage networks, rehabilitation of irrigation canals Promote solid waste sector sustainability by rehabilitate existing treatment facilities (sorting and composting plants), building new ones in case they do not exist, and investing in building and equipping sanitary landfills Enhance access and sustainability of transport by focusing on public transport service delivery and electrification, and ensuring climate resilience of roads and ports <b>Cost (US$ million)</b> Energy 4000 Water 1800 Solid waste 200 Transport 1580 Long-term interventions to keep in mind Less urgent, but <b>HIGH</b>ly beneﬁcial interventions Adopt climate policies to further reduce emissions intensity on track for net zero trajectory, as the tradeoff between development and climate would arise by mid 2030s Adoption of a nation-wide circular economy approach Implement Landfill Gas recovery systems to further reduce GHG emissions in designated SW sites Scope innovative technologies to improve wastewater treatment while promoting energy efficiency Promote climate action in health and education services and foster innovation Implement the Integrated Hydrological Information System Meteorological and hydrometric network expansion and improvement <b>LONG TERM MEASURES (BEYOND 2030)</b> Promote climate-smart and sustainable ecoturism Establish a funding mechanism with a dedicated revenue stream to support road network maintenance works Plan for Just Transition in collaboration with non-government stakeholders Adopt green procurement principies Formalize and consolidate public transport providers Adoption of a national composting strategy to reduce the amount of organic waste sent to landfills Develop a sustainable financing model for disaster monitoring and management <b>MEDIUM-TERM MEASURES</b> DE VEL OPMENT URGENC Y <b>HIGH</b> <b>LOW</b></p>
```

## Example output

```json
{
  "html": "<section data-sources=\"141-154\">\n    <p data-sources=\"141-142\">The persistence of weak governance, compounded by the economic crisis, undermines the capacity of the water sector to respond to the impacts of climate change. The water sector faces significant governance challenges that impede effective resource management and delivery of sustainable water services. Despite adoption of integrated water resource management principles, progress is considerably slower than the regional average. Complex legal frameworks and regulations have resulted in a convoluted institutional structure, diminishing accountability and transparency. Responsibilities for water management are fragmented across ministries and administrative levels, leading to overlap of responsibilities and lack of accountability. Coordination among stakeholders is deficient, and there is a notable absence of reliable data for informed planning, strategy development, policy formulation, and investment prioritization. Prolongation of the crisis will exacerbate the sector's limitations by preventing new investments in basic service delivery and the storage capacity needed to build resilience to projected climate change in the long term. The renewable water resource diminished from more than 2,000 cubic meters per capita in the 1960s to less than 660 cubic meters per capita in 2020. The \"Muddling through\" scenario in the water sector will have serious consequences that are highlighted in Table 3.</p>\n    <table data-sources=\"143-154\">\n      <caption data-sources=\"143\">Table 3</caption>\n      <thead data-sources=\"144-145\">\n        <tr data-sources=\"144-145\">\n          <th data-sources=\"144\">Characteristics</th>\n          <th data-sources=\"145\">Impacts</th>\n        </tr>\n      </thead>\n      <tbody data-sources=\"146-154\">\n        <tr data-sources=\"146,151\">\n          <td data-sources=\"146\">• Limited infrastructure to store water and bridge the seasonal gap in the availability of water.</td>\n          <td data-sources=\"151\">• More than 70 percent of the population face critical water shortages and an increasing cost of supply.</td>\n        </tr>\n        <tr data-sources=\"147,152\">\n          <td data-sources=\"147\">• The operations of most of the country's about 75 wastewater treatment continue to be hindered by the high costs of energy and limited sewerage networks connectivity.</td>\n          <td data-sources=\"152\">• Increases in the cost of water for the extremely income vulnerable groups.</td>\n        </tr>\n        <tr data-sources=\"148,153\">\n          <td data-sources=\"148\">• The institutional capacity for building resilience to climate change is limited at the level of the water utility and the ministry.</td>\n          <td data-sources=\"153\">• Poor quality of services undermines the willingness of the population to pay, while the increase in the cost of basic goods and services means that many are unable to pay.</td>\n        </tr>\n        <tr data-sources=\"149,154\">\n          <td data-sources=\"149\">• Tariffs are too low to cover operation and maintenance costs for existing infrastructure, address the high share of non-revenue water, the brain drain of human capital and lack of technical capabilities.</td>\n          <td data-sources=\"154\">• Increased prevalence of, and inability to control, water borne diseases, as seen with the outbreak of cholera in October 2022</td>\n        </tr>\n        <tr data-sources=\"150\">\n          <td data-sources=\"150\">• The continuous brain drains of the human capital within the Water Establishments.</td>\n          <td data-sources=\"150\"></td>\n        </tr>\n      </tbody>\n    </table>\n  </section>\n  <section data-sources=\"195-198\">\n    <h2 data-sources=\"195\">3.5 Recommendations and Investment Needs</h2>\n    <p data-sources=\"196\">Amid a deepening economic crisis, Lebanon has almost no fiscal space, limited institutional capacity, and numerous development challenges; therefore, it is critical to prioritize and sequence recommended measures and interventions—reflecting their urgency, synergies, and trade-offs— in responding to development and climate needs. The recommended measures of this CCDR, responding to the above-highlighted development and climate needs, are shown in Figure 12.</p>\n    <figure data-sources=\"197-198\">\n      <figcaption data-sources=\"197\">Figure 12: Country Climate and Development Report Recommended Policies and Interventions</figcaption>\n      <h3 data-sources=\"198\">High Impact and Recovery-Driving Investments</h3>\n      <p data-sources=\"198\">Urgent interventions with low tradeoffs:</p>\n      <ul data-sources=\"198\">\n        <li data-sources=\"198\">Expand capacity of cleaner and affordable renewable energy sources and switch from liquid fuel to natural gas.</li>\n        <li data-sources=\"198\">Increase water sector resilience by rehabilitating/expanding the water supply network, reducing non-revenue water, water supply reservoirs, treatment plants, sewerage networks, and rehabilitating irrigation canals.</li>\n        <li data-sources=\"198\">Promote solid waste sector sustainability by rehabilitating existing treatment facilities (sorting and composting plants), building new ones if they do not exist, and investing in building and equipping sanitary landfills.</li>\n        <li data-sources=\"198\">Enhance access and sustainability of transport by focusing on public transport service delivery and electrification, and ensuring climate resilience of roads and ports.</li>\n      </ul>\n      <table data-sources=\"198\">\n        <caption data-sources=\"198\">Cost (US$ million)</caption>\n        <tbody data-sources=\"198\">\n          <tr data-sources=\"198\"><th data-sources=\"198\">Energy</th><td data-sources=\"198\">4000</td></tr>\n          <tr data-sources=\"198\"><th data-sources=\"198\">Water</th><td data-sources=\"198\">1800</td></tr>\n          <tr data-sources=\"198\"><th data-sources=\"198\">Solid waste</th><td data-sources=\"198\">200</td></tr>\n          <tr data-sources=\"198\"><th data-sources=\"198\">Transport</th><td data-sources=\"198\">1580</td></tr>\n        </tbody>\n      </table>\n      <h3 data-sources=\"198\">Medium-Term Measures</h3>\n      <p data-sources=\"198\">Less urgent, but highly beneficial interventions:</p>\n      <ul data-sources=\"198\">\n        <li data-sources=\"198\">Promote climate-smart and sustainable ecotourism.</li>\n        <li data-sources=\"198\">Establish a funding mechanism with a dedicated revenue stream to support road network maintenance works.</li>\n        <li data-sources=\"198\">Plan for a Just Transition in collaboration with non-government stakeholders.</li>\n        <li data-sources=\"198\">Adopt green procurement principles.</li>\n        <li data-sources=\"198\">Formalize and consolidate public transport providers.</li>\n        <li data-sources=\"198\">Adopt a national composting strategy to reduce the amount of organic waste sent to landfills.</li>\n        <li data-sources=\"198\">Develop a sustainable financing model for disaster monitoring and management.</li>\n      </ul>\n      <h3 data-sources=\"198\">Long-Term Measures (Beyond 2030)</h3>\n      <ul data-sources=\"198\">\n        <li data-sources=\"198\">Adopt climate policies to further reduce emissions intensity on a track for a net-zero trajectory, as the tradeoff between development and climate would arise by the mid-2030s.</li>\n        <li data-sources=\"198\">Adopt a nation-wide circular economy approach.</li>\n        <li data-sources=\"198\">Implement Landfill Gas recovery systems to further reduce GHG emissions in designated SW sites.</li>\n        <li data-sources=\"198\">Scope innovative technologies to improve wastewater treatment while promoting energy efficiency.</li>\n        <li data-sources=\"198\">Promote climate action in health and education services and foster innovation.</li>\n        <li data-sources=\"198\">Implement the Integrated Hydrological Information System.</li>\n        <li data-sources=\"198\">Expand and improve the meteorological and hydrometric network.</li>\n      </ul>\n    </figure>\n  </section>"
}


# Task

Below is the HTML content you are to transform.
The parent node(s) of this content is/are: {parent_tags}.

Content:

```html
{html_representation}
```
"""


# TODO: Experiment with Claude, which can do 128k tokens output
def create_router(
    gemini_api_key: str, 
    openai_api_key: str, 
    deepseek_api_key: str,
    openrouter_api_key: str,
) -> Router:
    """Create a LiteLLM Router with advanced load balancing and fallback configuration."""
    model_list = [
        {
            "model_name": "html-splitter",
            "litellm_params": {
                "model": "openrouter/x-ai/grok-3-mini", # 16k tokens output
                "api_key": openrouter_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        {
            "model_name": "html-splitter",
            "litellm_params": {
                "model": "openai/gpt-4o-mini", # 16k tokens output
                "api_key": openai_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        {
            "model_name": "html-splitter",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini", # 32k tokens output
                "api_key": openai_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        # {
        #     "model_name": "html-parser",
        #     "litellm_params": {
        #         "model": "openrouter/z-ai/glm-4.5", # 96k tokens output
        #         "api_key": openrouter_api_key,
        #         "max_parallel_requests": 3,
        #         "weight": 1
        #     }
        # },
        # {
        #     "model_name": "html-parser",
        #     "litellm_params": {
        #         "model": "openrouter/z-ai/glm-4.5-air", # 96k tokens output
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
                "max_parallel_requests": 3,
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


def _positional_data_from_blocks(content_blocks: list[ContentBlock], page_dimensions: dict[int, BoundingBox]) -> list[PositionalData]:
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
                    page_logical=block.positional_data.page_logical,
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
            model="html-splitter",
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
                split_result = SplitResult.model_validate_json(de_fence(response.choices[0].message.content, type="json"))
                split_result.validate_comprehensive_coverage(len(content_blocks))
                split_sections = [
                    SplitSection.from_split_group(group, html, content_blocks)
                    for group in split_result.groups
                ]

                # Debug: Dump input and response to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(os.path.join("artifacts", "splits", f"input_{timestamp}.html"), "w") as fw:
                    fw.write(html)
                response_model = response.model or "unknown"
                with open(os.path.join("artifacts", "splits", f"output_{timestamp}_{response_model.split('/')[-1]}.json"), "w") as fw:
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


async def _restructure_html(
    html_str: str, 
    parents: str, 
    router: Router,
    messages: Optional[list[dict[str, str]]] = None,
    max_validation_attempts: int = 3,
    attempt: int = 0,
) -> str:
    """
    Use LLM to restructure HTML content with better semantic structure.
    
    Args:
        html_str: HTML string to restructure (from section.html_str)
        parents: Parent tag context for the LLM
        router: LiteLLM Router for LLM calls
        messages: Optional list of messages for retry attempts
        max_validation_attempts: Maximum number of validation attempts
        attempt: Current attempt number
        
    Returns:
        Restructured HTML string
    """    
    # Prepare the prompt
    messages = messages or [
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

        if (
            response
            and isinstance(response, ModelResponse)
            and isinstance(response.choices[0], Choices)
            and response.choices[0].message.content
        ):            
            try:
                # Remove any code fencing and return the restructured HTML
                html_result = HTMLResult.model_validate_json(de_fence(response.choices[0].message.content, type="json"))
                if not html_result.html.strip():
                    raise ValueError("Empty HTML content in response")

                # Debug: Dump input and response to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(os.path.join("artifacts", "revisions", f"input_{timestamp}.html"), "w") as fw:
                    fw.write(html_str)
                response_model = response.model or "unknown"
                with open(os.path.join("artifacts", "revisions", f"output_{timestamp}_{response_model.split('/')[-1]}.html"), "w") as fw:
                    fw.write(html_result.html)

                # Validate that all data-sources attributes are valid range strings
                data_sources = re.findall(r'data-sources="([^"]+)"', html_result.html)
                for data_source in data_sources:
                    if not re.match(r'^[0-9\s,\-]+$', data_source):
                        raise ValueError(f"Invalid data-sources attribute: {data_source}. Must be a comma-separated list with only numbers, spaces, or dashes.")
                
                return html_result.html
            except (ValidationError, ValueError) as e:
                if attempt < max_validation_attempts - 1:
                    logger.warning(f"Validation error (attempt {attempt+1}/{max_validation_attempts}): {e}")
                    # Append error message and retry
                    messages.append(response.choices[0].message.model_dump())
                    messages.append({
                        "role": "user",
                        "content": f"Your previous response had a validation error: {str(e)}. "
                                    "Please correct your response to match the required schema and constraints."
                    })
                    return await _restructure_html(html_str, parents, router, messages, max_validation_attempts, attempt + 1)
                else:
                    logger.error(f"Validation error on final attempt: {e}, returning original HTML")
                    return html_str
        else:
            raise ValueError("No valid response from LLM")
            
    except Exception as e:
        if attempt < max_validation_attempts - 1:
            logger.warning(f"Error during HTML restructuring (attempt {attempt+1}/{max_validation_attempts}): {e}")
            return await _restructure_html(html_str, parents, router, messages, max_validation_attempts, attempt + 1)
        else:
            logger.error(f"Error during HTML restructuring on final attempt: {e}, returning original HTML")
            return html_str


async def process_top_level_structure(
    top_level_structure: list[tuple[TagName, list[ContentBlock]]],
    pdf_path: str,
    gemini_api_key: str,
    openai_api_key: str,
    deepseek_api_key: str,
    openrouter_api_key: str,
) -> list[StructuredNode]:
    """Process the top-level structure of the document."""
    # Create router with built-in load balancing and concurrency control
    router = create_router(
        gemini_api_key, 
        openai_api_key, 
        deepseek_api_key,
        openrouter_api_key,
    )

    doc = pymupdf.open(pdf_path)
    page_dimensions = {
        page.number: BoundingBox(x1=0, y1=0, x2=page.rect.width, y2=page.rect.height)
        for page in doc
    }

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
    return parent_nodes


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

        pdf_path = os.path.join("artifacts", "wkdir", "doc_601.pdf")

        with open(os.path.join("artifacts", "doc_601_top_level_structure.json"), "r") as fr:
            top_level_structure: list[tuple[TagName, list[ContentBlock]]] = [
                (TagName(tag_name), [ContentBlock.model_validate(block) for block in blocks])  # type: ignore[arg-type]
                for tag_name, blocks in json.load(fr)
            ]

        parent_nodes = await process_top_level_structure(
            top_level_structure,
            pdf_path,
            gemini_api_key,
            openai_api_key,
            deepseek_api_key,
            openrouter_api_key,
        )

        with open(os.path.join("artifacts", "doc_601_nested_structure.json"), "w") as fw:
            json.dump([node.model_dump() for node in parent_nodes], fw, indent=2)

    start_time = time.time()
    asyncio.run(_test_main())
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time} seconds")