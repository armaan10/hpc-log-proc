"""
gpu_processing.py

Reads raw GPU CSV files for a job (one file per node, multiple GPUs per file
identified by gpu_index column), computes all summary statistics, and returns:

    (summary_dict, gpu_ts_by_node)

where gpu_ts_by_node is {node_id: tidy_df} kept ONLY for RI calculation.

Aggregation philosophy:
  All summary stats (mean, max, min, std, skew, kurtosis, IoF, percentiles,
  time_above_threshold) are computed per-GPU first, then averaged across active
  GPUs.  This answers "how does a typical GPU in this job behave over time"
  rather than conflating between-GPU variance with temporal variance.
  Between-GPU variance is already captured by spatial RI.

  Exception — power and mem_used totals (sum): these are summed across active
  GPUs to give the true job-level total draw/consumption.

Power/energy fields:
  *_total_w / *_total_wh         — active GPUs only
  *_including_idle_w / *_wh     — all GPUs present in logs (active + idle)
  "including idle" = all GPUs in log files, not necessarily all num_alloc_gpus
  (a GPU with no rows after coercion won't appear).

Active GPU:  mean util_pct > GPU_BUSY_THRESHOLD (default 2%)
"""

import os
import datetime
import logging
import time

import numpy as np
import pandas
from s3_io import read_csv_from_s3
from scipy import stats as scipy_stats
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
GPU_BUSY_THRESHOLD = 2.0  # percent
QUANTILES = [0.25, 0.75, 0.95]


# --------------------------------------------------------------------------- #
# File loading                                                                 #
# --------------------------------------------------------------------------- #

def _load_gpu_file(path: str, job_id: int) -> pandas.DataFrame | None:
    """
    Load one GPU CSV file (one node). Applies timezone correction and unit
    conversions. Returns a tidy long-form DataFrame or None on failure.
    """
    try:
        from s3_io import read_csv_from_s3
        raw = read_csv_from_s3(path)
        if raw is None:
            return None
    except Exception as exc:
        logger.warning("Cannot read GPU file %s: %s", path, exc)
        return None

    if raw.empty:
        return None

    # Coerce all numeric columns — guard against stray strings
    numeric_cols = [
        "utilization_gpu_pct", "utilization_memory_pct",
        "memory_used_MiB", "memory_free_MiB",
        "temperature_gpu", "temperature_memory",
        "power_draw_W", "pcie_link_width_current",
        "clocks_current_sm_MHz", "clocks_current_memory_MHz",
        "clocks_current_video_MHz", "power_limit_W",
        "timestamp", "gpu_index",
    ]
    for col in numeric_cols:
        if col in raw.columns:
            raw[col] = pandas.to_numeric(raw[col], errors="coerce")

    # Drop rows where core columns are unusable
    core = ["timestamp", "gpu_index", "utilization_gpu_pct"]
    raw = raw.dropna(subset=[c for c in core if c in raw.columns])
    if raw.empty:
        return None

    node_id = os.path.basename(path).split(".")[0].split("-", 1)[1]

    frames = []
    for gpu_idx in raw["gpu_index"].unique():
        gdf = raw[raw["gpu_index"] == gpu_idx].copy()

        # Timezone correction: logs in US/Eastern → UTC
        first_ts = gdf["timestamp"].iloc[0]
        utc_offset_s = datetime.datetime(
            *time.gmtime(first_ts)[:6], tzinfo=EASTERN
        ).utcoffset().total_seconds()
        gdf["timestamp"] = pandas.to_datetime(
            gdf["timestamp"] - utc_offset_s, unit="s", origin="unix", utc=True
        )

        gdf = gdf.rename(columns={
            "utilization_gpu_pct":    "util_pct",
            "utilization_memory_pct": "mem_util_pct",
            "memory_used_MiB":        "mem_used_kib",
            "memory_free_MiB":        "mem_free_kib",
            "temperature_gpu":        "temperature",
            "power_draw_W":           "power_w",
        })

        # MiB -> KiB
        gdf["mem_used_kib"] = gdf["mem_used_kib"] * 1024.0
        gdf["mem_free_kib"] = gdf["mem_free_kib"] * 1024.0

        gdf["node_id"]   = node_id
        gdf["gpu_index"] = gpu_idx

        keep = ["timestamp", "node_id", "gpu_index",
                "util_pct", "mem_util_pct", "mem_used_kib",
                "mem_free_kib", "power_w", "temperature"]
        frames.append(gdf[[c for c in keep if c in gdf.columns]])

    if not frames:
        return None

    return pandas.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Per-GPU stat helpers                                                         #
# --------------------------------------------------------------------------- #

def _per_gpu_stats(
    active_ts: pandas.DataFrame,
    col: str,
    prefix: str,
) -> dict:
    """
    Compute all distribution stats per GPU, then average across active GPUs.
    Captures temporal variability within a typical GPU, not between-GPU
    variance (which is already captured by spatial RI).
    """
    if col not in active_ts.columns or active_ts.empty:
        return {
            f"{prefix}_mean":     np.nan,
            f"{prefix}_max":      np.nan,
            f"{prefix}_min":      np.nan,
            f"{prefix}_std":      np.nan,
            f"{prefix}_skew":     np.nan,
            f"{prefix}_kurtosis": np.nan,
            f"{prefix}_iof":      np.nan,
        }

    def gpu_stats(s):
        s = s.dropna()
        if len(s) == 0:
            return pandas.Series({
                "mean": np.nan, "max": np.nan, "min": np.nan,
                "std": np.nan, "skew": np.nan, "kurtosis": np.nan, "iof": np.nan,
            })
        m = s.mean()
        return pandas.Series({
            "mean":     m,
            "max":      s.max(),
            "min":      s.min(),
            "std":      s.std(),
            "skew":     scipy_stats.skew(s),
            "kurtosis": scipy_stats.kurtosis(s),
            "iof":      s.var() / m if m != 0 else 0.0,
        })

    # Explicit loop — immune to MultiIndex level numbering changes across
    # pandas versions. Equivalent to groupby(level=2).mean() but unambiguous.
    records = []
    for (node_id, gpu_idx), group in active_ts.groupby(["node_id", "gpu_index"]):
        records.append(gpu_stats(group[col]))
    if not records:
        per_gpu = pandas.Series({
            "mean": np.nan, "max": np.nan, "min": np.nan,
            "std": np.nan, "skew": np.nan, "kurtosis": np.nan, "iof": np.nan,
        })
    else:
        per_gpu = pandas.DataFrame(records).mean()

    return {
        f"{prefix}_mean":     float(per_gpu.get("mean",     np.nan)),
        f"{prefix}_max":      float(per_gpu.get("max",      np.nan)),
        f"{prefix}_min":      float(per_gpu.get("min",      np.nan)),
        f"{prefix}_std":      float(per_gpu.get("std",      np.nan)),
        f"{prefix}_skew":     float(per_gpu.get("skew",     np.nan)),
        f"{prefix}_kurtosis": float(per_gpu.get("kurtosis", np.nan)),
        f"{prefix}_iof":      float(per_gpu.get("iof",      np.nan)),
    }


def _per_gpu_percentiles(
    active_ts: pandas.DataFrame,
    col: str,
    prefix: str,
) -> dict:
    """Percentiles computed per GPU then averaged across active GPUs."""
    if col not in active_ts.columns or active_ts.empty:
        return {f"{prefix}_p{int(q*100)}": np.nan for q in QUANTILES}

    result = {}
    for q in QUANTILES:
        per_gpu = active_ts.groupby(["node_id", "gpu_index"])[col].quantile(q)
        result[f"{prefix}_p{int(q*100)}"] = float(per_gpu.mean())
    return result


def _per_gpu_time_above(
    active_ts: pandas.DataFrame,
    col: str,
    threshold: float,
    out_key: str,
) -> dict:
    """Fraction of timesteps above threshold per GPU, averaged across active GPUs."""
    if col not in active_ts.columns or active_ts.empty:
        return {out_key: np.nan}

    def frac(s):
        s = s.dropna()
        return (s > threshold).sum() / len(s) if len(s) > 0 else np.nan

    val = active_ts.groupby(["node_id", "gpu_index"])[col].apply(frac).mean()
    return {out_key: float(val)}


# --------------------------------------------------------------------------- #
# Main summarization                                                           #
# --------------------------------------------------------------------------- #

def summarize_gpu(
    gpu_files: list,
    base_folder: str,   # unused — paths are absolute from file_index
    job_id: int,
    threshold: float,
    runtime_s: float,
    num_alloc_gpus: int = 0,
) -> tuple[dict, dict]:
    """
    Returns:
        summary   : flat dict of scalar GPU metrics for this job
        ts_by_node: {node_id: tidy_df}  (raw ts, for RI only)
    """
    all_frames = []
    for path in gpu_files:
        df = _load_gpu_file(path, job_id)
        if df is not None:
            all_frames.append(df)

    if not all_frames:
        return {}, {}

    full_ts = pandas.concat(all_frames, ignore_index=True)

    # ts_by_node for RI — before any aggregation
    ts_by_node = {
        node: grp.reset_index(drop=True)
        for node, grp in full_ts.groupby("node_id")
    }

    # ------------------------------------------------------------------ #
    # Identify active GPUs                                                 #
    # ------------------------------------------------------------------ #
    per_gpu_mean_util = (
        full_ts.groupby(["node_id", "gpu_index"])["util_pct"]
        .mean()
        .reset_index(name="mean_util")
    )
    per_gpu_mean_util["is_active"] = per_gpu_mean_util["mean_util"] > threshold
    num_gpus_used    = int(per_gpu_mean_util["is_active"].sum())
    num_gpus_in_logs = len(per_gpu_mean_util)

    active_gpu_ids = per_gpu_mean_util.loc[
        per_gpu_mean_util["is_active"], ["node_id", "gpu_index"]
    ]
    active_ts = full_ts.merge(active_gpu_ids, on=["node_id", "gpu_index"], how="inner")

    # ------------------------------------------------------------------ #
    # Time-to-first-use                                                    #
    # ------------------------------------------------------------------ #
    job_start_ts = full_ts["timestamp"].min()
    first_use_ts = full_ts.loc[full_ts["util_pct"] > threshold, "timestamp"].min()
    time_to_gpu_use_s = (
        (first_use_ts - job_start_ts).total_seconds()
        if pandas.notna(first_use_ts) else np.nan
    )

    # ------------------------------------------------------------------ #
    # Power and memory — per-GPU scalars (active GPUs only)               #
    # ------------------------------------------------------------------ #
    per_gpu_scalars = (
        active_ts.groupby(["node_id", "gpu_index"])
        .agg(
            mean_power    = ("power_w",      "mean"),
            max_power     = ("power_w",      "max"),
            mean_mem_used = ("mem_used_kib", "mean"),
            max_mem_used  = ("mem_used_kib", "max"),
        )
        .reset_index()
    )
    per_gpu_scalars["energy_wh"] = per_gpu_scalars["mean_power"] * runtime_s / 3600.0

    # Power — all GPUs in logs (active + idle), for total energy accounting
    per_gpu_power_all = (
        full_ts.groupby(["node_id", "gpu_index"])["power_w"]
        .mean()
        .reset_index(name="mean_power_all")
    )
    per_gpu_power_all["energy_wh_all"] = (
        per_gpu_power_all["mean_power_all"] * runtime_s / 3600.0
    )

    summary = {}

    # ------------------------------------------------------------------ #
    # Counts and ratios                                                    #
    # ------------------------------------------------------------------ #
    summary["num_gpus_used"]            = num_gpus_used
    summary["num_gpus_total_in_logs"]   = num_gpus_in_logs
    summary["gpu_idle_allocated_ratio"] = (
        float(1.0 - num_gpus_used / num_alloc_gpus)
        if num_alloc_gpus > 0 else np.nan
    )

    # ------------------------------------------------------------------ #
    # GPU utilisation (SM)                                                 #
    # ------------------------------------------------------------------ #
    summary.update(_per_gpu_stats(active_ts, "util_pct", "gpu_util"))
    summary.update(_per_gpu_percentiles(active_ts, "util_pct", "gpu_util"))
    summary.update(_per_gpu_time_above(active_ts, "util_pct", threshold, "gpu_time_above_threshold_pct"))

    # ------------------------------------------------------------------ #
    # GPU memory utilisation                                               #
    # ------------------------------------------------------------------ #
    summary.update(_per_gpu_stats(active_ts, "mem_util_pct", "gpu_mem_util"))
    summary.update(_per_gpu_percentiles(active_ts, "mem_util_pct", "gpu_mem_util"))

    # ------------------------------------------------------------------ #
    # GPU memory used (KiB)                                                #
    # std/mean = per-GPU averaged (temporal variability)                  #
    # sum      = job-level total VRAM consumed (physical)                 #
    # ------------------------------------------------------------------ #
    summary.update(_per_gpu_stats(active_ts, "mem_used_kib", "gpu_mem_used_kib"))
    summary["gpu_mem_used_kib_sum"] = (
        float(per_gpu_scalars["mean_mem_used"].sum())
        if not per_gpu_scalars.empty else np.nan
    )

    # ------------------------------------------------------------------ #
    # GPU temperature                                                      #
    # ------------------------------------------------------------------ #
    summary.update(_per_gpu_stats(active_ts, "temperature", "gpu_temp"))

    # ------------------------------------------------------------------ #
    # GPU power                                                            #
    # Active-only totals: mean/peak/energy summed across active GPUs      #
    # All-GPU totals: mean/energy summed across all GPUs in logs           #
    # Temporal variability: per-GPU std/skew/kurtosis averaged            #
    # ------------------------------------------------------------------ #

    # Active GPUs only
    summary["gpu_power_mean_total_w"] = (
        float(per_gpu_scalars["mean_power"].sum()) if not per_gpu_scalars.empty else np.nan
    )
    summary["gpu_power_peak_w"] = (
        float(per_gpu_scalars["max_power"].sum()) if not per_gpu_scalars.empty else np.nan
    )
    summary["gpu_energy_total_wh"] = (
        float(per_gpu_scalars["energy_wh"].sum()) if not per_gpu_scalars.empty else np.nan
    )

    # All GPUs in logs (active + idle) — true energy cost of the allocation
    summary["gpu_power_mean_total_including_idle_w"] = (
        float(per_gpu_power_all["mean_power_all"].sum())
        if not per_gpu_power_all.empty else np.nan
    )
    summary["gpu_energy_total_including_idle_wh"] = (
        float(per_gpu_power_all["energy_wh_all"].sum())
        if not per_gpu_power_all.empty else np.nan
    )

    # Temporal variability of power draw — per-GPU averaged
    power_dist = _per_gpu_stats(active_ts, "power_w", "_tmp_power")
    summary["gpu_power_std_w"]    = power_dist.get("_tmp_power_std",      np.nan)
    summary["gpu_power_skew"]     = power_dist.get("_tmp_power_skew",     np.nan)
    summary["gpu_power_kurtosis"] = power_dist.get("_tmp_power_kurtosis", np.nan)
    summary["gpu_power_min_w"]    = power_dist.get("_tmp_power_min",      np.nan)

    # ------------------------------------------------------------------ #
    # Time to first GPU use                                                #
    # ------------------------------------------------------------------ #
    summary["time_to_gpu_use_s"]  = time_to_gpu_use_s
    summary["minutes_to_gpu_use"] = (
        time_to_gpu_use_s / 60.0 if pandas.notna(time_to_gpu_use_s) else np.nan
    )

    return summary, ts_by_node