import os
import json
import requests
import logging

logger = logging.getLogger(__name__)


def extract_layout(pdf_path: str, output_path: str) -> str:
    """Extract layout from a PDF file and save it to a JSON file"""
    with open(pdf_path, 'rb') as pdf_file:
        response = requests.post(
            os.getenv("LAYOUT_EXTRACTOR_API_URL"),
            files={'file': pdf_file},
            headers={'X-API-Key': os.getenv("LAYOUT_EXTRACTOR_API_KEY")},
            timeout=300  # 5 minutes timeout
        )
    response.raise_for_status()

    # Handle JSON response
    json_data = response.json()
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"Extracted layout for {pdf_path}")
    return output_path


if __name__ == "__main__":
    pdf_path = "artifacts/wkdir/doc_601.pdf"
    output_path = "artifacts/wkdir/doc_601.json"
    extract_layout(pdf_path, output_path)