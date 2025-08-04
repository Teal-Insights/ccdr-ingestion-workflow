import os
from pathlib import Path
from typing import Tuple, Literal
import boto3
from types_boto3_s3 import S3Client
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def verify_environment_variables() -> Tuple[str, str]:
    """Verify required environment variables are set."""
    bucket_name = os.getenv("S3_BUCKET_NAME")
    aws_region = os.getenv("AWS_REGION")

    if not bucket_name or not aws_region:
        raise ValueError("ERROR: S3_BUCKET_NAME and AWS_REGION must be set")

    # Verify AWS credentials are configured
    aws_profile = os.getenv("AWS_PROFILE")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not aws_profile and not (aws_access_key and aws_secret_key):
        raise ValueError(
            "ERROR: Must set either AWS_PROFILE or both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
        )

    return bucket_name, aws_region


def get_aws_session() -> boto3.Session:
    """Get AWS session using SSO profile if available."""
    # Try to use a specified AWS profile, fall back to default
    profile_name = os.getenv("AWS_PROFILE")

    if profile_name:
        try:
            session = boto3.Session(profile_name=profile_name)
            # Test the session
            sts_client = session.client("sts")
            identity = sts_client.get_caller_identity()
            logger.info(f"Using AWS profile: {profile_name}")
            logger.info(f"Authenticated as: {identity.get('Arn', 'Unknown')}")
            return session
        except Exception as e:
            logger.warning(f"Could not use AWS profile {profile_name}: {e}")
            logger.info("Falling back to default credential chain...")
    else:
        logger.info("No AWS_PROFILE specified, using default credential chain...")

    # Fall back to default session (will use SSO if configured as default)
    return boto3.Session()


def get_s3_client() -> S3Client:
    """Initializes and returns a boto3 S3 client."""
    # This reuses the logic from the original script to get a session.
    # It assumes credentials/profile are configured via environment variables or SSO.
    try:
        session = get_aws_session()
        sts_client = session.client("sts")
        sts_client.get_caller_identity()  # Test credentials
        return session.client("s3")
    except (NoCredentialsError, Exception):
        print("Falling back to default AWS credential chain.")
        return boto3.client("s3")


def download_pdf_from_s3(
    temp_dir: str, ids: Tuple[int, int] | None = None, storage_url: str | None = None
) -> Tuple[str, str | None]:
    """Download a PDF and, if available, the corresponding layout JSON file
    from S3 to a local temp file and save it to the temp directory.

    You must provide at least one of ids or storage_url.

    The storage URL is formatted as:
    https://{bucket_name}.s3.{region}.amazonaws.com/pub_{publication_id}/doc_{document_id}.pdf

    Args:
        temp_dir: The temporary directory to save the PDF to.
        ids: The publication ID and document ID of the PDF.
        storage_url: The S3 storage URL of the PDF.
        
    Returns:
        The local file path of the downloaded PDF.
    """
    if storage_url:
        # Parse the S3 URL to extract bucket and key
        # URL format: https://{bucket}.s3.{region}.amazonaws.com/{key}
        from urllib.parse import urlparse
        parsed = urlparse(storage_url)
        if not parsed.hostname or not parsed.path:
            raise ValueError(f"Invalid storage URL: {storage_url}")
        bucket = parsed.hostname.split('.')[0]  # Extract bucket from hostname
        pdf_key = parsed.path.lstrip('/')  # Remove leading slash to get the key
        layout_key = pdf_key.replace('.pdf', '.json')
    else:
        if not ids:
            raise ValueError("Must provide either storage_url or ids")
        # Construct from IDs
        bucket_name, _ = verify_environment_variables()
        publication_id, document_id = ids
        bucket = bucket_name
        pdf_key = f"pub_{publication_id}/doc_{document_id}.pdf"
        layout_key = pdf_key.replace('.pdf', '.json')

    # Create local filename
    pdf_filename = os.path.basename(pdf_key)
    layout_filename = os.path.basename(layout_key)
    local_pdf_path = os.path.join(temp_dir, pdf_filename)
    local_layout_path = os.path.join(temp_dir, layout_filename)

    # Download the file
    s3_client = get_s3_client()
    s3_client.download_file(Bucket=bucket, Key=pdf_key, Filename=local_pdf_path)
    try:
        s3_client.download_file(Bucket=bucket, Key=layout_key, Filename=local_layout_path)

        logger.info(f"Downloaded {pdf_key} from bucket {bucket} to {local_pdf_path}")
        if local_layout_path:
            logger.info(f"Downloaded {layout_key} from bucket {bucket} to {local_layout_path}")
        return local_pdf_path, local_layout_path
    except Exception:
        logger.error(f"Failed to download {layout_key} from bucket {bucket}")
        return local_pdf_path, None


def upload_json_to_s3(
    temp_dir: str, ids: Tuple[int, int]
) -> str:
    """
    Uploads a single JSON file to S3 and returns its public URL.

    Args:
        temp_dir: The temporary directory to save the JSON file to.
        ids: The publication ID and document ID of the JSON file.

    Returns:
        The final S3 storage URL for the object.
    """
    bucket_name, _ = verify_environment_variables()
    local_file = Path(temp_dir) / f"doc_{ids[1]}.json"
    s3_key = f"pub_{ids[0]}/doc_{ids[1]}.json"

    s3_client = get_s3_client()
    s3_client.upload_file(str(local_file), bucket_name, s3_key)

    region = s3_client.meta.region_name
    storage_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
    logger.info(f"Upload complete. Storage URL: {storage_url}")

    return storage_url


def upload_image_to_s3(
    temp_dir: str, ids: Tuple[int, int], block_number: int, extension: Literal["png", "webp"] = "png"
) -> str:
    """
    Uploads an image to S3 and returns its public URL.
    """
    bucket_name, _ = verify_environment_variables()
    local_file = Path(temp_dir) / "images" / f"doc_{ids[1]}_{block_number}.{extension}"
    s3_key = f"pub_{ids[0]}/doc_{ids[1]}_{block_number}.{extension}"

    s3_client = get_s3_client()
    s3_client.upload_file(str(local_file), bucket_name, s3_key)

    region = s3_client.meta.region_name
    storage_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
    logger.info(f"Upload complete. Storage URL: {storage_url}")

    return storage_url