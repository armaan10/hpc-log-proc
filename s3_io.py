"""
s3_io.py — S3 streaming utilities.

Streams CSV files directly from a public S3 bucket into pandas DataFrames
without writing to disk.

Bucket: mit-supercloud-dataset (public, no credentials required)
S3 path format: datacenter-challenge/202201/cpu/0000/<jobid>-timeseries.csv
Full S3 URI:    s3://mit-supercloud-dataset/<path>
"""

import io
import logging

import boto3
import pandas
from botocore import UNSIGNED
from botocore.config import Config

logger = logging.getLogger(__name__)

S3_BUCKET = "mit-supercloud-dataset"

# Module-level client — one per process (safe with multiprocessing since each
# worker process gets its own copy via fork/spawn)
_s3_client = None


def get_s3_client():
    """Return a process-local unsigned S3 client (lazy init)."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED),
        )
    return _s3_client


def read_csv_from_s3(s3_path: str, **pandas_kwargs) -> pandas.DataFrame | None:
    """
    Stream a CSV directly from S3 into a DataFrame.

    Args:
        s3_path: path within the bucket, e.g.
                 "datacenter-challenge/202201/cpu/0000/123-timeseries.csv"
        **pandas_kwargs: passed through to pandas.read_csv()

    Returns:
        DataFrame or None on failure.
    """
    client = get_s3_client()
    try:
        response = client.get_object(Bucket=S3_BUCKET, Key=s3_path)
        body = response["Body"].read()
        return pandas.read_csv(io.BytesIO(body), **pandas_kwargs)
    except client.exceptions.NoSuchKey:
        logger.warning("S3 key not found: %s", s3_path)
        return None
    except Exception as exc:
        logger.warning("Failed to read s3://%s/%s: %s", S3_BUCKET, s3_path, exc)
        return None
