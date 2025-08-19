import os
import json
import requests
import logging

logger = logging.getLogger(__name__)


def extract_layout(pdf_path: str, output_path: str, api_url: str, api_key: str) -> str:
    """Extract layout from a PDF file and save it to a JSON file"""
    with open(pdf_path, 'rb') as pdf_file:
        response = requests.post(
            api_url,
            files={'file': pdf_file},
            headers={'X-API-Key': api_key},
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
    import dotenv

    dotenv.load_dotenv()
    LAYOUT_EXTRACTOR_API_URL = os.getenv("LAYOUT_EXTRACTOR_API_URL")
    LAYOUT_EXTRACTOR_API_KEY = os.getenv("LAYOUT_EXTRACTOR_API_KEY")
    assert LAYOUT_EXTRACTOR_API_URL, "LAYOUT_EXTRACTOR_API_URL is not set"
    assert LAYOUT_EXTRACTOR_API_KEY, "LAYOUT_EXTRACTOR_API_KEY is not set"

    pdf_path = "artifacts/wkdir/doc_601.pdf"
    output_path = "artifacts/wkdir/doc_601.json"
    extract_layout(pdf_path, output_path, LAYOUT_EXTRACTOR_API_URL, LAYOUT_EXTRACTOR_API_KEY)