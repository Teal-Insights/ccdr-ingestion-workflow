import dotenv
import asyncio
import tempfile
import os
from pdf_parsing_eval.extract_text_blocks import extract_text_blocks_with_styling
from pdf_parsing_eval.extract_images import extract_images_from_pdf
from pdf_parsing_eval.extract_svgs import extract_svgs_from_pdf
from pdf_parsing_eval.combine_blocks import combine_blocks
from pdf_parsing_eval.convert_to_html import convert_blocks_to_html

dotenv.load_dotenv(override=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
assert gemini_api_key, "GEMINI_API_KEY is not set"
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
assert deepseek_api_key, "DEEPSEEK_API_KEY is not set"

# Create a temporary directory for the entire pipeline
temp_dir = tempfile.mkdtemp(prefix="pdf_parsing_pipeline_")
print(f"Using temporary directory: {temp_dir}")

pdf_path = "input.pdf"
print(f"Processing PDF: {pdf_path}")

# 1. Extract the text blocks from the PDF
blocks_output_file_path = os.path.join(temp_dir, "text_blocks.json")
extracted_text_blocks_path = extract_text_blocks_with_styling(pdf_path, blocks_output_file_path, temp_dir)
print(f"Text blocks extracted successfully!")

# 2. Extract the images from the PDF as blocks
images_output_file_path = os.path.join(temp_dir, "images.json")
images_dir = os.path.join(temp_dir, "images")
extracted_image_blocks_path = asyncio.run(extract_images_from_pdf(
    pdf_path, 
    images_output_file_path, 
    api_key=os.getenv("GEMINI_API_KEY"),
    images_dir=images_dir
))
print(f"Image blocks extracted successfully!")

# 3. Extract the SVGs from the PDF and add them as blocks
svg_output_file_path = os.path.join(temp_dir, "svgs.json")
svgs_dir = os.path.join(temp_dir, "svgs")
extracted_svg_blocks_path = asyncio.run(extract_svgs_from_pdf(
    pdf_path, 
    svg_output_file_path, 
    api_key=deepseek_api_key,
    svgs_dir=svgs_dir
))
print(f"SVG blocks extracted successfully!")

# 4. Combine the blocks into a single JSON file
combined_blocks_output_file_path = os.path.join(temp_dir, "combined_blocks.json")
combined_blocks_path = combine_blocks([extracted_text_blocks_path, extracted_image_blocks_path, extracted_svg_blocks_path], combined_blocks_output_file_path)
print(f"Blocks combined successfully!")

# 5. Convert the blocks to HTML with plaintext and no bboxes
html_output_file_path = os.path.join(temp_dir, "html.html")
html_path = convert_blocks_to_html(combined_blocks_path, html_output_file_path, rich_text=False, bboxes=False, include_ids=True)
print(f"HTML converted successfully!")

# 6. Detect the structure of the document


# 7. Convert the blocks to pseudo-html with rich text and bboxes


# 8. Have an LLM transform the pseudo-html into a real HTML document matching our spec


# 9. Transform the HTML document into a graph


# 10. Create the relations from anchor tags

print(f"Pipeline completed! All outputs in: {temp_dir}")

