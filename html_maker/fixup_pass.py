import logging
from typing import Iterable

from utils.html import ALLOWED_TAGS
from utils.claude_code_client import ClaudeCodeClient, FileInput
from html_maker.restructure_with_CC_service import HTML_PROMPT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _build_fixup_prompt(
    missing_ids: Iterable[int] | None,
    extra_ids: Iterable[int] | None,
    html_invalid: bool,
    invalid_tags: list[str] | None,
    gemini_feedback: str | None = None,
) -> str:
    prompt = "You were given the following prompt:\n\n" + HTML_PROMPT + "\n\nYou wrote your output to output.html, but it did not pass validation checks. Please fix the remaining issues in output.html.\n\nThe validation found these problems:"

    if missing_ids:
        prompt += f"\n\n- Missing IDs: {",".join(str(i) for i in missing_ids)} (these IDs from original_input.html are not referenced in any data-sources; "
        "either the input elements not been added to the output or the output elements have not been correctly annotated with data-sources)"

    if extra_ids:
        prompt += f"\n\n- Extra IDs: {",".join(str(i) for i in extra_ids)} (these IDs don't exist in original_input.html but are referenced in data-sources)"

    if missing_ids or extra_ids:
        prompt += (
            "\n\nPlease review output.html and fix the data-sources attributes to ensure:\n"
            "1. All content from input.html (0 to max ID) has been added to the output\n"
            "2. All content in output.html has been correctly annotated with data-sources\n"
            "3. No non-existent IDs are referenced in data-sources\n"
        )

    if html_invalid:
        prompt += "\n\n- HTML would not parse. Please fix the HTML to ensure it is valid."

    if invalid_tags:
        prompt += (
            f"\n\n- Invalid tags: {invalid_tags} (these tags are not allowed)\n\n"
            "Please review current_output.html and fix the tags to ensure the HTML contains only the following tags:\n"
            f"{', '.join(ALLOWED_TAGS)}"
        )

    if gemini_feedback:
        prompt += (
            "\n\n- Gemini feedback (critical issues to address):\n"
            f"{gemini_feedback}"
        )

    return prompt


def run_fixup_pass(
    original_html: str,
    current_output: str,
    output_file: str,
    missing_ids: Iterable[int] | None = None,
    extra_ids: Iterable[int] | None = None,
    html_invalid: bool = False,
    invalid_tags: list[str] | None = None,
    gemini_feedback: str | None = None,
    timeout_seconds: int = 600,
    doc_id: int | None = None,
) -> str:
    """Execute a single fixup pass using the Claude Code service and return updated HTML."""

    # Load configuration files for the Claude Code service
    with open("claude_config/.claude/settings.json", "r") as fr:
        settings_content = fr.read()
    with open("claude_config/.claude/hooks/validate_html.py", "r") as fr:
        validate_html_content = fr.read()

    config_files: list[FileInput] = [
        FileInput(path=".claude/settings.json", content=settings_content),
        FileInput(path=".claude/hooks/validate_html.py", content=validate_html_content),
    ]

    fixup_prompt = _build_fixup_prompt(
        missing_ids=missing_ids or [],
        extra_ids=extra_ids or [],
        html_invalid=html_invalid,
        invalid_tags=invalid_tags or [],
        gemini_feedback=gemini_feedback,
    )

    with ClaudeCodeClient() as client:
        updated_html: str = client.execute_fixup_job(
            original_html=original_html,
            current_output=current_output,
            fixup_prompt=fixup_prompt,
            output_file=output_file,
            config_files=config_files,
            timeout_s=timeout_seconds,
            doc_id=doc_id,
        )

    return updated_html
