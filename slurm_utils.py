"""
slurm_utils.py — cleans a raw slurm CSV row into typed scalar fields.
"""

import numpy as np
import pandas


def _parse_num_alloc_gpus(tres_alloc: str) -> int:
    """Extract GPU count from tres_alloc string (keys 1001=tesla, 1002=volta)."""
    if not isinstance(tres_alloc, str):
        return 0
    if "1001=" in tres_alloc or "1002=" in tres_alloc:
        try:
            return int(tres_alloc.split(",")[-1].split("=")[-1])
        except (ValueError, IndexError):
            return 0
    return 0


def _parse_gpu_type(tres_alloc: str) -> str | None:
    if not isinstance(tres_alloc, str):
        return None
    if "1001=" in tres_alloc:
        return "tesla"
    if "1002=" in tres_alloc:
        return "volta"
    return None


def clean_slurm_row(row: pandas.Series) -> dict:
    """
    Returns a flat dict of scalar slurm metadata fields.
    All timestamps converted to pandas Timestamp (UTC).
    Runtime in seconds (int).
    """
    start_ts = pandas.to_datetime(row.get("time_start"), unit="s", origin="unix", utc=True)
    end_ts   = pandas.to_datetime(row.get("time_end"),   unit="s", origin="unix", utc=True)

    runtime_s = int((end_ts - start_ts).total_seconds()) if (
        pandas.notna(start_ts) and pandas.notna(end_ts)
    ) else 0

    tres_alloc = row.get("tres_alloc", "")
    num_alloc_gpus = _parse_num_alloc_gpus(tres_alloc)
    gpu_type = _parse_gpu_type(tres_alloc)

    num_nodes = float(row.get("nodes_alloc", np.nan))
    node_hours = num_nodes * runtime_s / 3600.0 if runtime_s > 0 else np.nan
    gpu_hours  = num_nodes * runtime_s * num_alloc_gpus / 3600.0 if runtime_s > 0 else np.nan

    return {
        "start_timestamp":  start_ts,
        "end_timestamp":    end_ts,
        "runtime_seconds":  runtime_s,
        "node_hours":       node_hours,
        "gpu_hours":        gpu_hours,
        "num_nodes":        num_nodes,
        "num_alloc_gpus":   num_alloc_gpus,
        "gpu_type":         gpu_type,
        "user_id":          row.get("id_user"),
        "queue_name":       row.get("partition"),
        "exit_code":        row.get("exit_code"),
        "cpus_req":         row.get("cpus_req", np.nan),
        "project_id":       np.nan,   # not present in source data
        "domain":           np.nan,
        "subdomain":        np.nan,
    }
