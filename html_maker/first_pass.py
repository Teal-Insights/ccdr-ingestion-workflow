import logging
from pathlib import Path

from utils.html import ALLOWED_TAGS
from utils.claude_code_client import ClaudeCodeClient, FileInput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


HTML_PROMPT: str = f"""You are an expert in HTML and document structure.
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


async def run_first_pass(
    input_html: str,
    output_file: str,
    timeout_seconds: int = 3600,
) -> str:
    """Run the initial restructuring pass via the Claude Code service.

    Returns the restructured HTML string. Also writes to `output_file` via the service.
    """
    file_prompt: str = HTML_PROMPT + "\n\nThe input HTML is in input.html. Write the output to {output_file}."

    # Load configuration files for the Claude Code service
    with open("claude_config/.claude/settings.json", "r") as fr:
        settings_content = fr.read()
    with open("claude_config/.claude/hooks/validate_html.py", "r") as fr:
        validate_html_content = fr.read()

    config_files: list[FileInput] = [
        FileInput(path="input.html", content=input_html),
        FileInput(path=".claude/settings.json", content=settings_content),
        FileInput(path=".claude/hooks/validate_html.py", content=validate_html_content),
    ]

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Execute restructuring job
    with ClaudeCodeClient() as client:
        restructured_html: str = client.execute_restructuring_job(
            input_html=input_html,
            prompt=file_prompt,
            output_file=output_file,
            config_files=config_files,
            timeout_s=timeout_seconds,
        )

    return restructured_html


