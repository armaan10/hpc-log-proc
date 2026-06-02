"""
ri.py — Resource Imbalance calculation.

Implements the formal definitions exactly:

    RI_temporal(r) = max_{1<=n<=N} ( 1 - (sum_{t=0}^{T} U_{n,t}) /
                                         (sum_{t=0}^{T} max_{0<=t<=T} U_{n,t}) )

    RI_spatial(r)  = 1 - ( sum_{n=1}^{N} max_{0<=t<=T}(U_{n,t}) ) /
                         ( sum_{n=1}^{N} max_{0<=t<=T, 1<=n<=N}(U_{n,t}) )

Where:
    n = node index  (1..N)
    t = time index  (0..T)
    U_{n,t} = utilisation of resource r on node n at time t

Notes:
- Applied to ALL jobs (GPU and CPU-only).
- GPU RI is computed from GPU time-series (one metric per GPU,
  then averaged across GPUs on the same node to get a node-level signal).
- CPU RI is computed from CPU time-series.
- If a job has only 1 node, RI_spatial = 0 by definition (no cross-node
  variation possible).
- If all U values are 0, both RIs are 0.
"""

import logging
import numpy as np
import pandas

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Core formula                                                                 #
# --------------------------------------------------------------------------- #

def _ri_temporal(node_series: dict[str, np.ndarray]) -> float:
    """
    RI_temporal over a dict {node_id: 1-D array of U_{n,t}}.

    Returns value in [0, 1].  Returns 0 if all values are zero or there
    is only one measurement per node.
    """
    max_temporal = 0.0

    for node_id, u in node_series.items():
        u = np.asarray(u, dtype=float)
        u = u[~np.isnan(u)]
        if len(u) == 0:
            continue
        node_max = u.max()
        if node_max == 0:
            # RI is 0 for this node (no utilisation at any point)
            continue
        # denominator: T+1 copies of node_max
        denom = node_max * len(u)
        ri_n = 1.0 - u.sum() / denom
        if ri_n > max_temporal:
            max_temporal = ri_n

    return round(float(np.clip(max_temporal, 0.0, 1.0)), 4)


def _ri_spatial(node_series: dict[str, np.ndarray]) -> float:
    """
    RI_spatial over a dict {node_id: 1-D array of U_{n,t}}.

    Returns value in [0, 1].  Returns 0 for single-node jobs.
    """
    if len(node_series) <= 1:
        return 0.0

    node_maxes = {}
    for node_id, u in node_series.items():
        u = np.asarray(u, dtype=float)
        u = u[~np.isnan(u)]
        node_maxes[node_id] = u.max() if len(u) > 0 else 0.0

    global_max = max(node_maxes.values())
    if global_max == 0:
        return 0.0

    numerator   = sum(global_max - v for v in node_maxes.values())
    denominator = global_max * len(node_maxes)

    return round(float(np.clip(numerator / denominator, 0.0, 1.0)), 4)


def _compute_ri_pair(node_series: dict[str, np.ndarray]) -> tuple[float, float]:
    """Returns (ri_temporal, ri_spatial) for a metric."""
    return _ri_temporal(node_series), _ri_spatial(node_series)


# --------------------------------------------------------------------------- #
# Helpers to extract per-node arrays from tidy DataFrames                     #
# --------------------------------------------------------------------------- #

def _gpu_metric_by_node(
    ts_by_node: dict[str, pandas.DataFrame],
    column: str,
) -> dict[str, np.ndarray]:
    """
    For GPU ts, average across all GPUs on each node at each timestamp,
    giving a single node-level utilisation signal U_{n,t}.
    """
    result = {}
    for node_id, df in ts_by_node.items():
        if column not in df.columns:
            continue
        # Mean across GPUs at each timestep -> node-level signal
        node_ts = (
            df.groupby("timestamp")[column]
            .mean()
            .sort_index()
            .values
        )
        result[node_id] = node_ts
    return result


def _cpu_metric_by_node(
    ts_by_node: dict[str, pandas.DataFrame],
    column: str,
) -> dict[str, np.ndarray]:
    result = {}
    for node_id, df in ts_by_node.items():
        if column not in df.columns:
            continue
        ts = df.sort_values("timestamp")[column].values
        result[node_id] = ts
    return result


# --------------------------------------------------------------------------- #
# Public interface                                                              #
# --------------------------------------------------------------------------- #

def calculate_ri_for_job(
    gpu_ts_by_node: dict[str, pandas.DataFrame],
    cpu_ts_by_node: dict[str, pandas.DataFrame],
    is_gpu_job: bool,
) -> dict:
    """
    Compute all RI metrics for one job.
    Returns a flat dict of RI scalars.
    All metrics computed for all jobs; GPU RIs are NaN for CPU-only jobs.
    """
    res = {}

    # ------------------------------------------------------------------ #
    # GPU RIs (only when GPU time-series are available)                   #
    # ------------------------------------------------------------------ #
    if is_gpu_job and gpu_ts_by_node:
        gpu_metrics = {
            "gpu_util":     "util_pct",
            "gpu_mem_util": "mem_util_pct",
            "gpu_mem_used": "mem_used_kib",
            "gpu_power":    "power_w",
            "gpu_temp":     "temperature",
        }
        for label, col in gpu_metrics.items():
            series = _gpu_metric_by_node(gpu_ts_by_node, col)
            if series:
                ri_t, ri_s = _compute_ri_pair(series)
            else:
                ri_t, ri_s = np.nan, np.nan
            res[f"ri_temporal_{label}"] = ri_t
            res[f"ri_spatial_{label}"]  = ri_s
    else:
        # Fill GPU RI fields with NaN for CPU-only jobs
        for label in ["gpu_util", "gpu_mem_util", "gpu_mem_used", "gpu_power", "gpu_temp"]:
            res[f"ri_temporal_{label}"] = np.nan
            res[f"ri_spatial_{label}"]  = np.nan

    # ------------------------------------------------------------------ #
    # CPU RIs (all jobs)                                                  #
    # ------------------------------------------------------------------ #
    cpu_metrics = {
        "cpu_util":     "cpu_util_pct",
        "mem_rss":      "mem_rss_kb",
        "mem_avail":    "mem_avail_kb",
        "mem_pct_util": "mem_pct_utilization",
        "read_kb":      "read_kb",
        "write_kb":     "write_kb",
    }
    for label, col in cpu_metrics.items():
        series = _cpu_metric_by_node(cpu_ts_by_node, col)
        if series:
            ri_t, ri_s = _compute_ri_pair(series)
        else:
            ri_t, ri_s = np.nan, np.nan
        res[f"ri_temporal_{label}"] = ri_t
        res[f"ri_spatial_{label}"]  = ri_s

    return res
