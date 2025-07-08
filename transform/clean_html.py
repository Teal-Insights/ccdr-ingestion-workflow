import asyncio
import os
import argparse
import sys
from typing import Literal
from tenacity import retry, stop_after_attempt, wait_exponential
from litellm import acompletion, Choices
from litellm.files.main import ModelResponse
import dotenv

# TODO: use the `data-block-number` map to populate src attributes for `img` tags

PROMPT = """
Transform the following HTML document to match our clean HTML specification. Follow these requirements exactly:

## Transformation Rules

1. **Text Formatting**: Use only the following simple semantic HTML tags:
   - `<b>` for bold/higher font weight
   - `<i>` for italic
   - `<u>` for underline
   - `<s>` for strikethrough
   - `<sup>` for superscript
   - `<sub>` for subscript
   - `<h1>` through `<h6>` for headings

2. **Document Structure**: Use only the following structural tags:
   - `figure`, `figcaption` for figures
   - `img` for images and SVGs
   - `table`, `th`, `tr`, `td`, `caption` for tables
   - `ul`,`ol`,`li` for lists
   - `title` for the document title on title pages
   - `section` and `subsection` for logical sections of the document
   - `nav` to wrap table of contents and similar navigational `section`s (more than one `nav` element may be used if these sections are not adjacent)
   - `aside` for footnotes and sidebars (immediately after the referencing element if there is one; else between elements wherever seems most natural)
   - `p` for paragraphs
   - `math`, `code`, `blockquote` for math, code blocks, and quotations
   - `cite` for citations
   - `a` for internal and external links, including between ToC items and sections, and between reference markers and notes

3. **Attributes**:
   - Use semantic `id`s like "chapter-1", "footnote-1", "unnumbered-section-title" to support internal anchor linking
   - Add `data-section-type` attribute to sections using one of: 'abstract', 'acknowledgements', 'appendix', 'bibliography', 'chapter', 'conclusion', 'copyright_page', 'dedication', 'epilogue', 'executive_summary', 'footer', 'foreword', 'header', 'index', 'introduction', 'list_of_boxes', 'list_of_figures', 'list_of_tables', 'notes_section', 'part', 'preface', 'prologue', 'section', 'stanza', 'subsection', 'table_of_contents', 'text_box', 'title_page'
   - Map original block IDs to structural tags using `data-block-number` attribute with a comma-separated list of block numbers (e.g., `data-block-number="264,265"`)
   - `img` tags' `src` attribute may be left empty; we will use the `data-block-number` map later to populate this programmatically

4. **Content Consolidation**:
   - Consolidate multiple `<span>`, `<p>`, and `<div>` tags that belong to the same paragraph into a single `<p>` tag
   - Remove all inline styles and CSS classes (preserve only semantic formatting through allowed HTML tags)
   - Clean up encoding quirks (e.g., replace U+00A0 with space U+0020)
   - Remove unwanted line breaks and normalize whitespace
   - Preserve the original document's exact wording, including spelling, punctuation, and capitalization
   - Ensure proper reading order

5. **Output**:
   - Enclose the output in triple backticks (```html```) to indicate a code block

Input HTML:
{input_html}
"""


def extract_html_from_markdown(content: str) -> str:
    """Extract HTML content from markdown code fence if present."""
    if "```html" in content:
        # Take whatever is between ```html ... ```
        return content.split("```html")[1].split("```")[0].strip()
    return content.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def clean_html_with_llm(
    input_html: str, api_key: str, semaphore: asyncio.Semaphore
) -> str:
    """Clean HTML using LLM with semaphore to limit concurrent calls"""
    async with semaphore:
        response = await acompletion(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "user", "content": PROMPT.format(input_html=input_html)}
            ],
            api_key=api_key,
        )
        if (
            response
            and isinstance(response, ModelResponse)
            and isinstance(response.choices[0], Choices)
            and response.choices[0].message.content
        ):
            return extract_html_from_markdown(response.choices[0].message.content)
        else:
            raise Exception("No valid response from LLM")


async def process_html_file(
    file_path: str, api_key: str, semaphore: asyncio.Semaphore
) -> str:
    """Process a single HTML file and return cleaned HTML"""
    with open(file_path, "r", encoding="utf-8") as f:
        input_html = f.read()
    return await clean_html_with_llm(input_html, api_key, semaphore)


async def process_html_inputs_concurrently(
    html_input_paths: list[tuple[Literal["header", "main", "footer"], str]],
    output_path: str,
    api_key: str,
    max_concurrent_calls: int = 5,
) -> str:
    """
    Process multiple HTML input files concurrently and assemble into a single clean HTML document.

    Args:
        html_input_paths: List of tuples where each tuple contains (section_type, file_path)
        output_path: Path where the assembled HTML file will be written
        api_key: API key for the LLM service
        max_concurrent_calls: Maximum number of concurrent LLM API calls

    Returns:
        Path to the output HTML file
    """
    semaphore = asyncio.Semaphore(max_concurrent_calls)

    print(
        f"🚀 Processing {len(html_input_paths)} HTML files with max {max_concurrent_calls} concurrent calls..."
    )

    # Create tasks for all input files
    tasks = []
    for section_type, file_path in html_input_paths:
        task = process_html_file(file_path, api_key, semaphore)
        tasks.append((section_type, file_path, task))

    # Execute all tasks concurrently
    results = []
    for section_type, file_path, task in tasks:
        try:
            cleaned_html = await task
            results.append((section_type, cleaned_html))
            print(
                f"  ✅ Processed {section_type} section from {os.path.basename(file_path)}"
            )
        except Exception as e:
            print(
                f"  ❌ Failed to process {section_type} section from {os.path.basename(file_path)}: {e}"
            )
            results.append(
                (section_type, f"<!-- Error processing {section_type}: {e} -->")
            )

    # Assemble the final HTML document
    html_sections = {"header": [], "main": [], "footer": []}

    for section_type, content in results:
        html_sections[section_type].append(content)

    # Build the final HTML structure
    html_parts = ["<html>", "<body>"]

    # Add header sections
    if html_sections["header"]:
        html_parts.append("<header>")
        html_parts.extend(html_sections["header"])
        html_parts.append("</header>")

    # Add main sections
    if html_sections["main"]:
        html_parts.append("<main>")
        html_parts.extend(html_sections["main"])
        html_parts.append("</main>")

    # Add footer sections
    if html_sections["footer"]:
        html_parts.append("<footer>")
        html_parts.extend(html_sections["footer"])
        html_parts.append("</footer>")

    html_parts.extend(["</body>", "</html>"])

    # Write the assembled HTML to file
    final_html = "\n".join(html_parts)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"✅ Assembled HTML document written to: {output_path}")
    return output_path


def parse_input_spec(
    input_spec: str,
) -> tuple[Literal["header", "main", "footer"], str]:
    """
    Parse input specification in format 'section_type:file_path'

    Args:
        input_spec: String in format 'header:path/to/file.html' or 'main:path/to/file.html' or 'footer:path/to/file.html'

    Returns:
        Tuple of (section_type, file_path)
    """
    if ":" not in input_spec:
        raise ValueError(
            f"Invalid input spec '{input_spec}'. Expected format: 'section_type:file_path'"
        )

    section_type, file_path = input_spec.split(":", 1)

    if section_type not in ["header", "main", "footer"]:
        raise ValueError(
            f"Invalid section type '{section_type}'. Must be one of: header, main, footer"
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Type cast to satisfy the type checker since we've validated the value
    return section_type, file_path  # type: ignore


async def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Clean HTML files using LLM and assemble into a structured document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python -m transform.clean_html -i main:input.html -o output.html -k YOUR_API_KEY
  
  # Multiple files
  python -m transform.clean_html \\
    -i header:header.html \\
    -i main:content1.html \\
    -i main:content2.html \\
    -i footer:footer.html \\
    -o assembled_document.html \\
    -k YOUR_API_KEY
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        action="append",
        required=True,
        help="Input HTML files in format 'section_type:file_path' where section_type is header|main|footer. Can be specified multiple times.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path for the assembled HTML document",
    )

    parser.add_argument(
        "-k",
        "--api-key",
        help="API key for DeepSeek LLM service (if not provided, will try to load DEEPSEEK_API_KEY from environment)",
    )

    parser.add_argument(
        "-c",
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum number of concurrent LLM API calls (default: 3)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Handle API key - try command line first, then environment
    api_key = args.api_key
    if not api_key:
        dotenv.load_dotenv(override=True)
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print(
                "Error: No API key provided. Either use --api-key or set DEEPSEEK_API_KEY environment variable.",
                file=sys.stderr,
            )
            sys.exit(1)
        elif args.verbose:
            print("Using DEEPSEEK_API_KEY from environment")

    # Parse input specifications
    try:
        html_input_paths = []
        for input_spec in args.input:
            section_type, file_path = parse_input_spec(input_spec)
            html_input_paths.append((section_type, file_path))
            if args.verbose:
                print(f"Added {section_type} section: {file_path}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Process the HTML files
    try:
        result_path = await process_html_inputs_concurrently(
            html_input_paths, args.output, api_key, args.max_concurrent
        )

        print(f"\n🎉 Success! Cleaned HTML document saved to: {result_path}")

    except Exception as e:
        print(f"Error processing HTML files: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
