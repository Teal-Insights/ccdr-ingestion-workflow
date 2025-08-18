import logging
import json
import time
from pathlib import Path

from utils.models import ContentBlock, StructuredNode
from utils.html import create_nodes_from_html, validate_data_sources, validate_html_tags, ALLOWED_TAGS
from utils.claude_code_client import ClaudeCodeClient, FileInput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HTML_PROMPT = f"""You are an expert in HTML and document structure.
You are given an HTML partial with a flat structure and only `p` and `img` tags.
Your task is to propose a better structured HTML representation of the content, with semantic tags and logical hierarchy.
Only `header`, `main`, and/or `footer` are allowed at the top level; all other tags should be nested within one of these.

# Requirements

- You may only use the following tags: {", ".join(ALLOWED_TAGS)}.
    - `header`: Front matter (title pages, table of contents, preface, etc.)
    - `main`: Body content (main chapters, sections, core content)
    - `footer`: Back matter (appendices, bibliography, index, etc.)
- Styling is not in your purview, and styles in your output will be ignored.
- You may split, merge, or replace structural containers as necessary, but you should make an effort to:
    - Clean up any whitespace, encoding, redundant style tags, or other formatting issues
    - Otherwise maintain the identical wording/spelling of the text content and of image descriptions and source URLs
    - Assign elements a `data-sources` attribute with a comma-separated list of ids of the source elements (for attribute mapping and content validation). This can include ranges, e.g., `data-sources="0-5,7,9-12"`
        - Note: inline style tags `b`, `i`, `u`, `s`, `sup`, `sub`, and `br` do not need `data-sources`, but all other tags should have this attribute
    - The `data-sources` MUST reference only IDs that actually exist in the input HTML. Do not invent IDs or include IDs outside the input's set
    - When using ranges, both endpoints MUST exist in the input, and the range MUST NOT span missing IDs. If necessary, split into multiple valid ranges (e.g., `0-3,5,7-9`)
    - Never extrapolate beyond the minimum/maximum ID present in the input; do not create ranges like `0-10` if the input only contains `0-4`
- You may find it helpful to make small-to-medium-sized edits rather than very large ones.
    - If, for example, `main` spans many tens of thousands of characters, you might create the top level container first and then incrementally populate it over the course of several edits.
    - Outputting more than 32,000 tokens at a time may cause your session to crash.

# File Output Instructions

You MUST restructure the input HTML and save the complete restructured result to the specified output file path.

Please read the input HTML file carefully, analyze its structure, and create a well-structured HTML output that follows all the requirements above. Save the complete restructured HTML to the output file.

Remember:
- Every element (except inline style tags b, i, u, s, sup, sub, br) MUST have a data-sources attribute
- The data-sources values must comprehensively cover all IDs from the input HTML
- Use semantic HTML structure with proper nesting
- Clean up formatting issues while preserving exact text content
"""


def restructure_with_claude_code(
    content_blocks: list[ContentBlock],
    output_file: str = "output.html",
    timeout_seconds: int = 3600  # 60 minutes default for document restructuring
) -> list[StructuredNode]:
    """
    Restructure HTML using Claude Code API service.
    
    Args:
        content_blocks: List of content blocks to restructure
        output_file: Name of output file
        timeout_seconds: Timeout for the Claude Code job (default 30 minutes for complex documents)
        
    Returns:
        List of structured nodes parsed from output
    """
    
    # Generate input HTML
    input_html = "\n".join([block.to_html(block_id=i) for i, block in enumerate(content_blocks)])
    
    # Create the prompt that references the input file
    file_prompt = f"""You were asked to restructure the HTML content from the input file: input.html.

You were given the following prompt:

{HTML_PROMPT}"""

    # Load the settings and validate_html.py files
    with open("claude_config/.claude/settings.json", "r") as fr:
        settings_content = fr.read()
    with open("claude_config/.claude/hooks/validate_html.py", "r") as fr:
        validate_html_content = fr.read()

    # Prepare configuration files
    config_files = [
        FileInput(path=".claude/settings.json", content=settings_content),
        FileInput(path=".claude/hooks/validate_html.py", content=validate_html_content)
    ]

    # Execute the restructuring job using the client
    with ClaudeCodeClient() as client:
        restructured_html = client.execute_restructuring_job(
            input_html=input_html,
            prompt=file_prompt,
            output_file=output_file,
            config_files=config_files,
            timeout_s=timeout_seconds
        )

    # Create fixup prompt to address validation issues
    fixup_prompt = """

Please fix the data-sources validation issues in current_output.html.

The validation found these problems:"""

    # Validate data sources
    missing_ids, extra_ids = validate_data_sources(input_html, restructured_html)
    is_valid, invalid_tags = validate_html_tags(restructured_html)

    if (len(missing_ids) + len(extra_ids)) > 3:
        fixup_prompt += f"""

- Missing IDs: {missing_ids} (these IDs from original_input.html are not referenced in any data-sources)
- Extra IDs: {extra_ids} (these IDs don't exist in original_input.html but are referenced in data-sources)

Please review current_output.html and fix the data-sources attributes to ensure:
1. All IDs from original_input.html (0 to max ID) are referenced exactly once across all data-sources
2. No non-existent IDs are referenced in data-sources
3. Ranges in data-sources are valid (both endpoints exist, no gaps in ranges)"""
        logger.warning(f"Data sources validation issues - Missing: {missing_ids}, Extra: {extra_ids}")

    if not is_valid:
        fixup_prompt += """

- Invalid HTML

HTML would not parse. Please fix the HTML to ensure it is valid."""
        logger.warning("Invalid HTML")

    if invalid_tags:
        fixup_prompt += f"""

- Invalid tags: {invalid_tags} (these tags are not allowed)

Please review current_output.html and fix the tags to ensure the HTML contains only the following tags:
{", ".join(ALLOWED_TAGS)}"""
        logger.warning(f"Invalid tags: {invalid_tags}")

        logger.info("Running fixup job to address validation issues...")
        
        with ClaudeCodeClient() as fixup_client:
            restructured_html = fixup_client.execute_fixup_job(
                original_html=input_html,
                current_output=restructured_html,
                fixup_prompt=fixup_prompt,
                output_file=output_file,
                config_files=config_files,
                timeout_s=600  # Shorter timeout for fixup
            )
        
        # Re-validate after fixup
        missing_ids, extra_ids = validate_data_sources(input_html, restructured_html)
        is_valid, invalid_tags = validate_html_tags(restructured_html)
        if (len(missing_ids) + len(extra_ids)) > 0:
            logger.warning(f"Validation issues remain after fixup - Missing: {missing_ids}, Extra: {extra_ids}")
        else:
            logger.info("Validation issues resolved after fixup")

    if not is_valid:
        raise ValueError("Invalid HTML")
    if invalid_tags:
        raise ValueError(f"Invalid tags: {invalid_tags}")

    # Extract body contents if present
    if "<body>" in restructured_html and "</body>" in restructured_html:
        restructured_html = restructured_html.split("<body>")[1].split("</body>")[0]
    
    # Parse into structured nodes
    nodes = create_nodes_from_html(restructured_html, content_blocks)
    
    logger.info(f"Successfully restructured HTML with {len(nodes)} top-level nodes")
    
    return nodes


if __name__ == "__main__":
    import json
    import dotenv
    import time
    
    dotenv.load_dotenv()
    
    # Load content blocks
    input_file = Path("artifacts") / "doc_601_content_blocks_with_styles.json"
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        exit(1)
    
    with open(input_file, "r") as fr:
        content_blocks_data = json.load(fr)
        content_blocks = [ContentBlock.model_validate(block) for block in content_blocks_data]
    
    logger.info(f"Loaded {len(content_blocks)} content blocks")
    
    # Run restructuring
    start_time = time.time()
    try:
        structured_nodes = restructure_with_claude_code(
            content_blocks,
            output_file="output.html",
            timeout_seconds=3600  # 60 minutes for complex document restructuring
        )
        end_time = time.time()
        
        # Save structured nodes to JSON
        output_json = Path("artifacts") / "doc_601_nested_structure.json"
        with open(output_json, "w") as fw:
            json.dump([node.model_dump() for node in structured_nodes], fw, indent=2)
        
        logger.info(f"Process completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Structured data saved to {output_json}")
        
    except Exception as e:
        logger.error(f"Restructuring failed: {e}")
        exit(1)