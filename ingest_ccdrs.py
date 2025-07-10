"""
OPTIMAL SOLUTION:
Columns: ['page.get_svg_image', 'page.get_text("html")', 'page.get_text("blocks")', 'page.get_images']

Coverage details:
  Full image bounding box: page.get_images = Y
  Usable PNG/JPG images: page.get_images = Y
  Usable SVG images: page.get_svg_image = Y
  Path invisibility detection: page.get_svg_image + element.is_visible (playwright) = Y
  Readable text: page.get_text("html") = Y
  Semantic grouping: page.get_text("html") = Y
  Granular CSS styles: page.get_text("html") = Y
  Full text bounding box: page.get_text("blocks") = Y
  Text invisibility detection: page.get_text("html") + element.is_visible (playwright) = Y
"""


import dotenv
import asyncio
import tempfile
import os
from typing import Literal
from transform.extract_text_blocks import extract_text_blocks_with_styling
from transform.extract_images import extract_images_from_pdf
from transform.extract_svgs import extract_svgs_from_pdf
from transform.combine_blocks import combine_blocks
from transform.convert_to_html import convert_blocks_to_html
from transform.detect_structure import detect_structure
from transform.clean_html import process_html_inputs_concurrently
from transform.describe_images import describe_images_in_json

dotenv.load_dotenv(override=True)

# Fail fast if the API keys are not set
gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
assert gemini_api_key, "GEMINI_API_KEY is not set"
deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
assert deepseek_api_key, "DEEPSEEK_API_KEY is not set"

# Create a temporary directory for the entire pipeline
temp_dir: str = tempfile.mkdtemp(prefix="pdf_parsing_pipeline_")
print(f"Using temporary directory: {temp_dir}")

# TODO: Make this a command line argument and support use of folder paths
pdf_path: str = "input.pdf"
print(f"Processing PDF: {pdf_path}")

# 1. Extract ImageBlocks and write BlocksDocument to <temp_dir>/images.json
extracted_image_blocks_path, images_dir = extract_images_from_pdf(
    pdf_path, os.path.join(temp_dir, "images.json"), images_dir=os.path.join(temp_dir, "images")
)
print("Image blocks extracted successfully!")

# 2. Extract the SvgBlocks from the PDF and write BlocksDocument to <temp_dir>/svgs.json
svgs_dir: str = os.path.join(temp_dir, "svgs")
extracted_svg_blocks_path: str = asyncio.run(extract_svgs_from_pdf(
    pdf_path, os.path.join(temp_dir, "svgs.json"), svgs_dir=svgs_dir
))
print("SVG blocks extracted successfully!")

# 3. Extract TextBlocks from the PDF and write BlocksDocument to <temp_dir>/text_blocks.json
extracted_text_blocks_path: str = extract_text_blocks_with_styling(
    pdf_path, os.path.join(temp_dir, "text_blocks.json"), temp_dir
)
print(f"Text blocks extracted successfully to {extracted_text_blocks_path}!")

# 4. Combine the blocks into a single JSON file
combined_blocks_output_file_path: str = os.path.join(temp_dir, "combined_blocks.json")
combined_blocks_path: str = combine_blocks(
    [
        extracted_text_blocks_path,
        extracted_image_blocks_path,
        extracted_svg_blocks_path,
    ],
    combined_blocks_output_file_path,
)
print("Blocks combined successfully!")

# 5. Describe the images and SVGs with a VLM
# TODO: Pass text context with image to Gemini to improve description
# TODO: Validate that all ImageBlocks and SvgBlocks have a description after this step
descriptions_output_file_path: str = os.path.join(temp_dir, "described_blocks.json")
described_blocks_path: str = asyncio.run(describe_images_in_json(
    json_file_path=combined_blocks_path,
    images_dir=images_dir,
    api_key=gemini_api_key,
    output_file_path=descriptions_output_file_path,
    svg_as_text=True,
))
print("Images and SVGs described successfully!")

# TODO: Blocks' text field contains HTML with in-line styles that still need cleaning
# Extract unique styles and have an LLM write a transformation rule for each one

# 6. Convert the blocks to HTML with ids, plaintext, and no bboxes
# (This is purely about context length management and id tagging for the next step)
html_output_file_path: str = os.path.join(temp_dir, "html.html")
html_path: str = convert_blocks_to_html(
    combined_blocks_path,
    html_output_file_path,
    rich_text=False,
    bboxes=False,
    include_ids=True,
)
print("HTML created successfully!")

# 7. Detect the structure of the document and return paths to JSON blocksdocs for each section
# (This is purely about content length management for the next step, since outputs can be max 8k tokens)
structure_output_dir: str = os.path.join(temp_dir, "structure")
structure_paths: list[tuple[Literal["header", "main", "footer"], str]] = asyncio.run(
    detect_structure(
        html_path, combined_blocks_path, structure_output_dir, api_key=gemini_api_key
    )
)
print("Structure detected successfully!")

# 8. Convert the blocks to HTML with rich text and bboxes
# (reuse the function from step 5 with different parameters)
rich_html_output_paths: list[tuple[Literal["header", "main", "footer"], str]] = []
for section_name, section_path in structure_paths:
    rich_html_output_file_path: str = os.path.join(temp_dir, f"{section_name}.html")
    rich_html_output_paths.append(
        (
            section_name,
            convert_blocks_to_html(
                section_path,
                rich_html_output_file_path,
                rich_text=True,
                bboxes=True,
                include_ids=True,
            ),
        )
    )
    print(f"Rich HTML created successfully for {section_name}!")

# 9. Have an LLM clean and conform the HTML to our spec
cleaned_html_path: str = asyncio.run(
    process_html_inputs_concurrently(
        rich_html_output_paths,
        os.path.join(temp_dir, "cleaned_document.html"),
        api_key=deepseek_api_key,
        max_concurrent_calls=3,
    )
)
print("HTML cleaned and assembled successfully!")

# 10. Transform the cleaned HTML document into a graph matching our schema and ingest it into our database


# 11. Enrich the database records by generating relations from anchor tags

print(f"Pipeline completed! All outputs in: {temp_dir}")
