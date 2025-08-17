import logging
import os
import time
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import dotenv
from pydantic import BaseModel, Field

from utils.models import ContentBlock, StructuredNode, TagName
from utils.html import create_nodes_from_html, validate_data_sources

# Load environment variables
dotenv.load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# API configuration
API_KEY = os.getenv("CLAUDE_CODE_API_KEY")
BASE_URL = os.getenv("CLAUDE_CODE_BASE_URL", "").rstrip("/")

if not API_KEY or not BASE_URL:
    raise ValueError("CLAUDE_CODE_API_KEY and CLAUDE_CODE_BASE_URL must be set in .env file")

# Default constants
DEFAULT_SESSION_TTL_S = 3600
DEFAULT_JOB_TIMEOUT_S = 600
MAX_JOB_TIMEOUT_S = 1800

# --- Pydantic Models for API ---

class FileInput(BaseModel):
    path: str = Field(..., description="Relative path of the file in the workspace.")
    content: str = Field(..., description="UTF-8 encoded content of the file.")

class SessionCreateRequest(BaseModel):
    files: List[FileInput]
    ttl_seconds: int = Field(DEFAULT_SESSION_TTL_S, gt=0)

class FileInfo(BaseModel):
    path: str
    size_bytes: int

class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: datetime
    expires_at: datetime
    input_files: List[FileInfo]

class JobRunRequest(BaseModel):
    prompt: str
    timeout_s: int = Field(DEFAULT_JOB_TIMEOUT_S, gt=0, le=MAX_JOB_TIMEOUT_S)

class JobRunResponse(BaseModel):
    job_id: str
    session_id: str
    status: str = "queued"
    created_at: datetime

class OutputFileResult(BaseModel):
    path: str
    size_bytes: int
    sha256: str
    download_url: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    session_id: str
    status: str # "queued", "running", "succeeded", "failed", "timed_out"
    error: Optional[str] = None
    prompt: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    output_files: Optional[List[OutputFileResult]] = None

class SessionResultsResponse(BaseModel):
    session_id: str
    latest_job_id: Optional[str] = None
    output_files: List[OutputFileResult]
    result_json: Optional[Dict[str, Any]] = None

ALLOWED_TAGS: str = ", ".join(
    tag.value for tag in TagName
) + ", b, i, u, s, sup, sub, br"

HTML_PROMPT = f"""You are an expert in HTML and document structure.
You are given an HTML partial with a flat structure and only `p` and `img` tags.
Your task is to propose a better structured HTML representation of the content, with semantic tags and logical hierarchy.
Only `header`, `main`, and/or `footer` are allowed at the top level; all other tags should be nested within one of these.

# Requirements

- You may only use the following tags: {ALLOWED_TAGS}.
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
    working_dir: Optional[Path] = None,  # pylint: disable=unused-argument
    timeout_seconds: int = 1800  # 30 minutes default for document restructuring
) -> list[StructuredNode]:
    """
    Restructure HTML using Claude Code API service.
    
    Args:
        content_blocks: List of content blocks to restructure
        output_file: Name of output file
        working_dir: Optional working directory (not used by API, kept for compatibility)
        timeout_seconds: Timeout for the Claude Code job (default 30 minutes for complex documents)
        
    Returns:
        List of structured nodes parsed from output
    """
    
    # Generate input HTML
    input_html = "\n".join([block.to_html(block_id=i) for i, block in enumerate(content_blocks)])
    
    # Create the prompt that references the input file
    file_prompt = f"""Please restructure the HTML content from the input file: input.html

Save the complete restructured HTML to: {output_file}

{HTML_PROMPT}"""

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    logger.info("Creating session on Claude Code API service...")

    # Load the settings and validate_html.py files
    with open("claude_config/.claude/settings.json", "r") as fr:
        settings_content = fr.read()
    with open("claude_config/.claude/hooks/validate_html.py", "r") as fr:
        validate_html_content = fr.read()

    # Step 1: Create a session with the input HTML file
    session_request = SessionCreateRequest(
        files=[
            FileInput(path="input.html", content=input_html),
            FileInput(path=".claude/settings.json", content=settings_content),
            FileInput(path=".claude/hooks/validate_html.py", content=validate_html_content)
        ],
        ttl_seconds=max(timeout_seconds * 2, 3600)  # Give extra time for session
    )
    
    session_response = requests.post(
        f"{BASE_URL}/session",
        headers=headers,
        json=session_request.model_dump()
    )
    
    if session_response.status_code != 200:
        raise RuntimeError(f"Failed to create session: {session_response.status_code} - {session_response.text}")
    
    session_create_response = SessionCreateResponse.model_validate(session_response.json())
    session_id = session_create_response.session_id
    logger.info(f"Created session: {session_id}")
    
    try:
        # Step 2: Start a job to restructure the HTML
        job_request = JobRunRequest(
            prompt=file_prompt,
            timeout_s=timeout_seconds
        )
        
        job_response = requests.post(
            f"{BASE_URL}/job",
            headers=headers,
            params={"session_id": session_id},
            json=job_request.model_dump()
        )
        
        if job_response.status_code != 200:
            raise RuntimeError(f"Failed to start job: {job_response.status_code} - {job_response.text}")
        
        job_run_response = JobRunResponse.model_validate(job_response.json())
        job_id = job_run_response.job_id
        logger.info(f"Started job: {job_id}")
        
        # Step 3: Poll for job completion
        start_time = time.time()
        while True:
            status_response = requests.get(
                f"{BASE_URL}/job/status",
                headers=headers,
                params={"job_id": job_id}
            )
            
            if status_response.status_code != 200:
                raise RuntimeError(f"Failed to get job status: {status_response.status_code} - {status_response.text}")
            
            status_data = status_response.json()
            job_status = JobStatusResponse.model_validate(status_data)
            status = job_status.status
            
            if status == "succeeded":
                logger.info("Job completed successfully")
                break
            elif status == "failed":
                error_msg = job_status.error or "Unknown error"
                raise RuntimeError(f"Job failed: {error_msg}")
            elif status == "timed_out":
                raise RuntimeError("Job timed out on server")
            elif status in ["queued", "running"]:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    raise RuntimeError(f"Job timed out after {elapsed:.1f} seconds")
                logger.info(f"Job status: {status} (elapsed: {elapsed:.1f}s)")
                time.sleep(5)  # Poll every 5 seconds
            else:
                raise RuntimeError(f"Unknown job status: {status}")
        
        # Step 4: Get the results
        results_response = requests.get(
            f"{BASE_URL}/session/results",
            headers=headers,
            params={"session_id": session_id}
        )
        
        if results_response.status_code != 200:
            raise RuntimeError(f"Failed to get results: {results_response.status_code} - {results_response.text}")
        
        session_results = SessionResultsResponse.model_validate(results_response.json())
        if session_results.result_json:
            logger.info(f"Response from Claude Code API service: {session_results.result_json}")
        output_files = session_results.output_files
        
        # Find the output HTML file
        output_html_file = None
        for file_info in output_files:
            if file_info.path == output_file:
                output_html_file = file_info
                break
        
        if not output_html_file:
            raise RuntimeError(f"Output file '{output_file}' not found in results. Available files: {[f.path for f in output_files]}")
        
        # Step 5: Download the output HTML
        download_response = requests.get(
            f"{BASE_URL}/download",
            headers=headers,
            params={
                "session_id": session_id,
                "file_path": output_file
            }
        )
        
        if download_response.status_code != 200:
            raise RuntimeError(f"Failed to download output: {download_response.status_code} - {download_response.text}")
        
        restructured_html = download_response.text
        
        # Validate data sources
        missing_ids, extra_ids = validate_data_sources(input_html, restructured_html)
        if (len(missing_ids) + len(extra_ids)) > 3:
            logger.warning(f"Data sources validation issues - Missing: {missing_ids}, Extra: {extra_ids}")
        
        # Extract body contents if present
        if "<body>" in restructured_html and "</body>" in restructured_html:
            restructured_html = restructured_html.split("<body>")[1].split("</body>")[0]
        
        # Parse into structured nodes
        nodes = create_nodes_from_html(restructured_html, content_blocks)
        
        logger.info(f"Successfully restructured HTML with {len(nodes)} top-level nodes")
        
        return nodes
        
    finally:
        # Clean up: Delete the session
        try:
            delete_response = requests.delete(
                f"{BASE_URL}/session",
                headers=headers,
                params={"session_id": session_id},
                timeout=30  # Add timeout to avoid hanging on cleanup
            )
            if delete_response.status_code in [200, 204]:
                logger.info(f"Deleted session: {session_id}")
            elif delete_response.status_code == 404:
                logger.info(f"Session {session_id} already deleted or expired")
            else:
                logger.warning(f"Failed to delete session: {delete_response.status_code} - {delete_response.text}")
        except requests.exceptions.RequestException as e:
            # Handle network/connection errors gracefully
            logger.warning(f"Network error during session cleanup (session may auto-expire): {e}")
        except Exception as e:
            # Handle any other unexpected errors
            logger.warning(f"Unexpected error cleaning up session: {e}")


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
            timeout_seconds=1800  # 30 minutes for complex document restructuring
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