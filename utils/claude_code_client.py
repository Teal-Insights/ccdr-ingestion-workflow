"""
Claude Code API client for managing sessions, jobs, and file operations.

This module provides a reusable client for interacting with the Claude Code API service,
handling session lifecycle, job execution, and result retrieval.
"""

import logging
import os
import time
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime
import dotenv
from pydantic import BaseModel, Field

# Load environment variables
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# --- Pydantic Models for API ---

class FileInput(BaseModel):
    path: str = Field(..., description="Relative path of the file in the workspace.")
    content: str = Field(..., description="UTF-8 encoded content of the file.")

class SessionCreateRequest(BaseModel):
    files: List[FileInput]
    ttl_seconds: int = Field(3600, gt=0)

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
    timeout_s: int = Field(600, gt=0, le=3600)

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


class ClaudeCodeClient:
    """
    Client for interacting with the Claude Code API service.
    
    Handles session management, job execution, polling, and cleanup.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, cleanup_on_exit: bool = False):
        """
        Initialize the Claude Code API client.
        
        Args:
            api_key: API key for authentication. Defaults to CLAUDE_CODE_API_KEY env var.
            base_url: Base URL for the API. Defaults to CLAUDE_CODE_BASE_URL env var.
        """
        api_key_value = api_key or os.getenv("CLAUDE_CODE_API_KEY")
        base_url_value = base_url or os.getenv("CLAUDE_CODE_BASE_URL")

        if not api_key_value or not base_url_value:
            raise ValueError("CLAUDE_CODE_API_KEY and CLAUDE_CODE_BASE_URL must be set")

        self.api_key: str = api_key_value
        self.base_url: str = base_url_value.rstrip("/")

        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        self.cleanup_on_exit = cleanup_on_exit
        self.session_id: Optional[str] = None
    
    def create_session(self, files: List[FileInput], ttl_seconds: int = 3600) -> str:
        """
        Create a new session with the provided files.
        
        Args:
            files: List of files to upload to the session
            ttl_seconds: Time-to-live for the session in seconds
            
        Returns:
            Session ID
            
        Raises:
            RuntimeError: If session creation fails
        """
        logger.info("Creating session on Claude Code API service...")
        
        session_request = SessionCreateRequest(
            files=files,
            ttl_seconds=ttl_seconds
        )
        
        response = requests.post(
            f"{self.base_url}/session",
            headers=self.headers,
            json=session_request.model_dump()
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to create session: {response.status_code} - {response.text}")
        
        session_create_response = SessionCreateResponse.model_validate(response.json())
        self.session_id = session_create_response.session_id
        logger.info(f"Created session: {self.session_id}")
        
        return self.session_id
    
    def run_job(self, prompt: str, timeout_s: int = 600) -> str:
        """
        Run a job in the current session.
        
        Args:
            prompt: The prompt to execute
            timeout_s: Timeout for the job in seconds
            
        Returns:
            Job ID
            
        Raises:
            RuntimeError: If job creation fails or no session exists
        """
        if not self.session_id:
            raise RuntimeError("No active session. Call create_session() first.")
        
        job_request = JobRunRequest(
            prompt=prompt,
            timeout_s=timeout_s
        )
        
        response = requests.post(
            f"{self.base_url}/job",
            headers=self.headers,
            params={"session_id": self.session_id},
            json=job_request.model_dump()
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to start job: {response.status_code} - {response.text}")
        
        job_run_response = JobRunResponse.model_validate(response.json())
        job_id = job_run_response.job_id
        logger.info(f"Started job: {job_id}")
        
        return job_id
    
    def wait_for_job_completion(self, job_id: str, timeout_s: int = 600, poll_interval: int = 5) -> JobStatusResponse:
        """
        Poll for job completion.
        
        Args:
            job_id: ID of the job to monitor
            timeout_s: Maximum time to wait for completion
            poll_interval: Time between status checks in seconds
            
        Returns:
            Final job status response
            
        Raises:
            RuntimeError: If job fails, times out, or polling exceeds timeout
        """
        start_time = time.time()
        
        while True:
            response = requests.get(
                f"{self.base_url}/job/status",
                headers=self.headers,
                params={"job_id": job_id}
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to get job status: {response.status_code} - {response.text}")
            
            job_status = JobStatusResponse.model_validate(response.json())
            status = job_status.status
            
            if status == "succeeded":
                logger.info("Job completed successfully")
                return job_status
            elif status == "failed":
                error_msg = job_status.error or "Unknown error"
                raise RuntimeError(f"Job failed: {error_msg}")
            elif status == "timed_out":
                raise RuntimeError("Job timed out on server")
            elif status in ["queued", "running"]:
                elapsed = time.time() - start_time
                if elapsed > timeout_s:
                    raise RuntimeError(f"Job timed out after {elapsed:.1f} seconds")
                logger.info(f"Job status: {status} (elapsed: {elapsed:.1f}s)")
                time.sleep(poll_interval)
            else:
                raise RuntimeError(f"Unknown job status: {status}")
    
    def get_session_results(self) -> SessionResultsResponse:
        """
        Get results from the current session.
        
        Returns:
            Session results including output files
            
        Raises:
            RuntimeError: If no session exists or request fails
        """
        if not self.session_id:
            raise RuntimeError("No active session")
        
        response = requests.get(
            f"{self.base_url}/session/results",
            headers=self.headers,
            params={"session_id": self.session_id}
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get results: {response.status_code} - {response.text}")
        
        session_results = SessionResultsResponse.model_validate(response.json())
        if session_results.result_json:
            logger.info(f"Response from Claude Code API service: {session_results.result_json}")
        
        return session_results
    
    def download_file(self, file_path: str) -> str:
        """
        Download a file from the current session.
        
        Args:
            file_path: Path of the file to download
            
        Returns:
            File contents as string
            
        Raises:
            RuntimeError: If no session exists or download fails
        """
        if not self.session_id:
            raise RuntimeError("No active session")

        logger.info(f"Downloading file {file_path} from session {self.session_id}")
        response = requests.get(
            f"{self.base_url}/download",
            headers=self.headers,
            params={
                "session_id": self.session_id,
                "file_path": file_path
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download output: {response.status_code} - {response.text}")
        
        return response.text
    
    def delete_session(self) -> None:
        """
        Delete the current session.
        
        Handles cleanup gracefully with proper error logging.
        """
        if not self.session_id or not self.cleanup_on_exit:
            return
        
        try:
            response = requests.delete(
                f"{self.base_url}/session",
                headers=self.headers,
                params={"session_id": self.session_id},
                timeout=30
            )
            
            if response.status_code in [200, 204]:
                logger.info(f"Deleted session: {self.session_id}")
            elif response.status_code == 404:
                logger.info(f"Session {self.session_id} already deleted or expired")
            else:
                logger.warning(f"Failed to delete session: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error during session cleanup (session may auto-expire): {e}")
        except Exception as e:
            logger.warning(f"Unexpected error cleaning up session: {e}")
        finally:
            self.session_id = None
    
    def execute_restructuring_job(
        self, 
        input_html: str, 
        prompt: str,
        output_file: str = "output.html",
        config_files: Optional[List[FileInput]] = None,
        timeout_s: int = 3600
    ) -> str:
        """
        Execute a complete restructuring job with session management.
        
        This is a high-level method that handles the full workflow:
        1. Create session with input files
        2. Run restructuring job
        3. Wait for completion
        4. Download results
        5. Clean up session
        
        Args:
            input_html: HTML content to restructure
            prompt: Restructuring prompt for the job
            output_file: Name of the expected output file
            config_files: Additional configuration files to include
            timeout_s: Job timeout in seconds
            
        Returns:
            Restructured HTML content
            
        Raises:
            RuntimeError: If any step in the process fails
        """
        config_files = config_files or []
        
        # Prepare input files
        files = [FileInput(path="input.html", content=input_html)] + config_files
        
        try:
            # Create session
            self.create_session(files, ttl_seconds=max(timeout_s * 2, 3600))
            
            # Run job
            job_id = self.run_job(prompt, timeout_s)
            
            # Wait for completion
            self.wait_for_job_completion(job_id, timeout_s)
            
            # Get results and verify output file exists
            results = self.get_session_results()
            output_files = results.output_files
            
            # Find the output file
            output_html_file = None
            for file_info in output_files:
                if file_info.path == output_file:
                    output_html_file = file_info
                    break
            
            if not output_html_file:
                available_files = [f.path for f in output_files]
                raise RuntimeError(f"Output file '{output_file}' not found in results. Available files: {available_files}")
            
            # Download the output
            return self.download_file(output_file)
            
        finally:
            # Always clean up
            self.delete_session()
    
    def execute_fixup_job(
        self,
        original_html: str,
        current_output: str,
        fixup_prompt: str,
        output_file: str = "output.html",
        config_files: Optional[List[FileInput]] = None,
        timeout_s: int = 3600
    ) -> str:
        """
        Execute a fixup job to correct issues in existing output.
        
        This method creates a new session with both the original input and current output,
        then runs a fixup job to address specific issues identified after validation.
        
        Args:
            original_html: Original input HTML for context
            current_output: Current output that needs fixing
            fixup_prompt: Specific prompt describing what needs to be fixed
            output_file: Name of the expected output file
            config_files: Additional configuration files to include
            timeout_s: Job timeout in seconds (shorter default for fixups)
            
        Returns:
            Fixed HTML content
            
        Raises:
            RuntimeError: If any step in the process fails
        """
        config_files = config_files or []
        
        # Prepare input files - include both original and current output for context
        files = [
            FileInput(path="input.html", content=original_html),
            FileInput(path="output.html", content=current_output)
        ] + config_files
        
        try:
            # Create session
            self.create_session(files, ttl_seconds=max(timeout_s * 2, 3600))
            
            # Run fixup job
            job_id = self.run_job(fixup_prompt, timeout_s)
            
            # Wait for completion
            self.wait_for_job_completion(job_id, timeout_s)
            
            # Get results and verify output file exists
            results = self.get_session_results()
            output_files = results.output_files
            
            # Find the output file
            output_html_file = None
            for file_info in output_files:
                if file_info.path == output_file:
                    output_html_file = file_info
                    break
            
            if not output_html_file:
                available_files = [f.path for f in output_files]
                raise RuntimeError(f"Output file '{output_file}' not found in results. Available files: {available_files}")
            
            # Download the output
            return self.download_file(output_file)
            
        finally:
            # Always clean up
            self.delete_session()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures session cleanup."""
        self.delete_session()
