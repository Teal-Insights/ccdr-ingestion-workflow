"""CCDR Ingestion Pipeline

Main orchestration script for the CCDR (Country and Climate Development Reports) ingestion workflow.
Transforms World Bank PDF documents into a structured graph database format suitable for semantic search.

This script orchestrates the full pipeline by running both phases:
1. generate_content_blocks.py - PDF processing to content blocks
2. process_content_blocks.py - content blocks to database

Usage:
    uv run ingest_ccdrs.py                    # Run full pipeline
    uv run generate_content_blocks.py         # Run phase 1 only
    uv run process_content_blocks.py          # Run phase 2 only

The script processes documents in batches (configurable LIMIT) and outputs intermediate
artifacts to a working directory for debugging and pipeline inspection.
"""

import subprocess
import sys


def run_subprocess(script_name: str) -> int:
    """Run a subprocess and return the exit code."""
    print(f"Running {script_name}...")
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"{script_name} completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return e.returncode


def main() -> None:
    """Run the complete CCDR ingestion pipeline by orchestrating both phases."""
    print("Starting CCDR Ingestion Pipeline...")
    print("=" * 50)
    
    # Phase 1: Generate content blocks from PDFs
    print("Phase 1: Generating content blocks from PDFs")
    exit_code_1 = run_subprocess("generate_content_blocks.py")
    if exit_code_1 != 0:
        print(f"Phase 1 failed with exit code {exit_code_1}")
        sys.exit(exit_code_1)
    
    print("\n" + "=" * 50)
    
    # Phase 2: Process content blocks to database
    print("Phase 2: Processing content blocks to database")
    exit_code_2 = run_subprocess("generate_structured_nodes.py")
    if exit_code_2 != 0:
        print(f"Phase 2 failed with exit code {exit_code_2}")
        sys.exit(exit_code_2)
    
    print("\n" + "=" * 50)
    print("CCDR Ingestion Pipeline completed successfully!")
    print("Both phases completed without errors.")


if __name__ == "__main__":
    main()