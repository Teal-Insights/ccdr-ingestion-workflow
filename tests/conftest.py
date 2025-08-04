import logging
from pathlib import Path
from typing import List
import pytest
from botocore.exceptions import NoCredentialsError, ClientError
from utils.aws import get_s3_client, verify_environment_variables

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def sample_data_dir() -> Path:
    """
    Ensures sample data files are available for testing by downloading from S3 if needed.
    
    Returns the path to the sample_data directory.
    """
    sample_data_path = Path(__file__).parent / "sample_data"
    sample_data_path.mkdir(exist_ok=True)
    
    # List of expected sample data files
    expected_files = [
        "doc_601.json",
        "doc_601_content_blocks.json", 
        "doc_601_content_blocks_with_text.json",
        "doc_601_content_blocks_with_styles.json",
        "doc_601_content_blocks_with_images.json",
        "doc_601_content_blocks_with_descriptions.json",
        "doc_601_with_logical_page_numbers.json"
    ]
    
    # Check which files are missing
    missing_files = [
        filename for filename in expected_files 
        if not (sample_data_path / filename).exists()
    ]
    
    if missing_files:
        logger.info(f"Missing {len(missing_files)} sample data files, downloading from S3...")
        _download_sample_data_from_s3(sample_data_path, missing_files)
    else:
        logger.info("All sample data files present locally")
    
    return sample_data_path


def _download_sample_data_from_s3(local_dir: Path, filenames: List[str]) -> None:
    """Download missing sample data files from S3."""
    try:
        # Use the same environment verification as the main AWS utils
        bucket_name, _ = verify_environment_variables()
        s3_client = get_s3_client()
        
        # S3 prefix where sample data is stored
        s3_prefix = "sample_data/"
        
        for filename in filenames:
            s3_key = f"{s3_prefix}{filename}"
            local_file = local_dir / filename
            
            try:
                logger.info(f"Downloading {s3_key} to {local_file}")
                s3_client.download_file(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Filename=str(local_file)
                )
                logger.info(f"Successfully downloaded {filename}")
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'NoSuchKey':
                    logger.warning(f"File {s3_key} not found in S3 bucket {bucket_name}")
                else:
                    logger.error(f"Failed to download {s3_key}: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error downloading {s3_key}: {e}")
                raise
                
    except (NoCredentialsError, ValueError) as e:
        # If AWS credentials aren't configured, skip the download
        # This allows tests to run locally even without S3 access
        logger.warning(f"Cannot download sample data from S3: {e}")
        logger.warning("Tests requiring sample data may fail")
        
    except Exception as e:
        logger.error(f"Failed to download sample data: {e}")
        raise


@pytest.fixture
def doc_601_data(sample_data_dir: Path) -> dict:
    """Load the basic doc_601.json data for tests."""
    import json
    
    data_file = sample_data_dir / "doc_601.json"
    if not data_file.exists():
        pytest.skip("doc_601.json sample data not available")
        
    with open(data_file) as f:
        return json.load(f)


@pytest.fixture  
def doc_601_with_images(sample_data_dir: Path) -> dict:
    """Load the doc_601_content_blocks_with_images.json data for tests."""
    import json
    
    data_file = sample_data_dir / "doc_601_content_blocks_with_images.json"
    if not data_file.exists():
        pytest.skip("doc_601_content_blocks_with_images.json sample data not available")
        
    with open(data_file) as f:
        return json.load(f)


@pytest.fixture
def doc_601_with_descriptions(sample_data_dir: Path) -> dict:
    """Load the doc_601_content_blocks_with_descriptions.json data for tests."""
    import json
    
    data_file = sample_data_dir / "doc_601_content_blocks_with_descriptions.json" 
    if not data_file.exists():
        pytest.skip("doc_601_content_blocks_with_descriptions.json sample data not available")
        
    with open(data_file) as f:
        return json.load(f)


@pytest.fixture
def doc_601_with_logical_pages(sample_data_dir: Path) -> dict:
    """Load the doc_601_with_logical_page_numbers.json data for tests."""
    import json
    
    data_file = sample_data_dir / "doc_601_with_logical_page_numbers.json"
    if not data_file.exists():
        pytest.skip("doc_601_with_logical_page_numbers.json sample data not available")
        
    with open(data_file) as f:
        return json.load(f)