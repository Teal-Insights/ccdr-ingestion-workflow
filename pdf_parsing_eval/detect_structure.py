# Take a look at extract_images.py to see how to call Gemini with LiteLLM (we won't need concurrency this time)
# We want a function that takes the path to an HTML file with numbered divs (corresponding to PDF blocks)
# Return JSON object with inclusive block number ranges for header, main, footer, corresponding to front matter, body matter, and back matter of the document
# Example:
# {
#   "header": "1-3,5,7-9",
#   "main": "4,6,10-12",
#   "footer": "13-15"
# }
# Support comma-separated ranges and hyphenated ranges
# Convert number ranges to lists or some other iterable containing all the block numbers (ints) for each section
# Allow any of the sections to be empty (no pages corresponding to that section)
# Enforce that all blocks are included in one of the sections
# Generate a JSON BlocksDocument file for each section and return a list of section_name, file_path pairs
# Planned for future: check token length of HTML corresponding to each section to see if it exceeds LLM output limit, in which case we need to get next level of section structure
