"""
file_index.py — loads the S3 file index from the JSON map.

JSON format (one entry per job):
    {
        "89213993887039": {
            "cpu": ["datacenter-challenge/202201/cpu/0058/89213993887039-timeseries.csv"],
            "gpu": ["datacenter-challenge/202201/gpu/0047/89213993887039-r629115-n830961.csv"]
        },
        ...
    }

Paths in the JSON are relative to the S3 bucket root (s3://mit-supercloud-dataset/).
s3_io.py prepends the bucket name when making requests.

This module is kept for any code that imports build_file_index by name.
In the new pipeline, pipeline.py loads the JSON directly.
"""

import json
import logging

logger = logging.getLogger(__name__)


def load_file_index(json_path: str) -> dict:
    """
    Load and return the file index from a JSON file.

    Returns:
        {
            "<job_id_str>": {
                "cpu": [s3_path, ...],
                "gpu": [s3_path, ...],
            },
            ...
        }
    """
    with open(json_path, "r") as f:
        index = json.load(f)

    gpu_count = sum(1 for v in index.values() if v.get("gpu"))
    cpu_count = sum(1 for v in index.values() if v.get("cpu"))

    logger.info(
        "File index loaded: %d jobs total, %d with CPU files, %d with GPU files.",
        len(index), cpu_count, gpu_count,
    )
    return index


# Alias for backwards compatibility with any code using build_file_index
def build_file_index(json_path: str) -> dict:
    return load_file_index(json_path)
