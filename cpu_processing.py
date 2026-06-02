"""
cpu_processing.py

Reads the CPU time-series CSV for a job, computes summary statistics,
and returns:

    (summary_dict, cpu_ts_by_node)

where cpu_ts_by_node is {node_id: df} kept ONLY for RI calculation.

CPU utilisation normalisation:
    CPUUtilization is the aggregate across all CPUs on that node.
    cpus_per_node = cpus_req / num_nodes  (even distribution assumption)
    cpu_util_pct  = CPUUtilization / cpus_per_node  (per-row, before any agg)

Aggregation philosophy (mirrors gpu_processing.py):
    All distribution stats (mean, max, min, std, skew, kurtosis, IoF,
    percentiles, time_above_threshold) are computed per-node first, then
    averaged across nodes.  This answers "how does a typical node behave
    over time" — temporal variability — rather than conflating cross-node
    variance with temporal variance.  Cross-node variance is already
    captured by spatial RI.

    Exception — I/O totals (sum): summed across nodes for job-level total.
"""

import logging

import numpy as np
import pandas
from s3_io import read_csv_from_s3
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

CPU_BUSY_THRESHOLD = 50.0  # percent
QUANTILES = [0.25, 0.75, 0.95]

_RENAME = {
    "Node":           "node_id",
    "CPUUtilization": "cpu_util_raw",
    "RSS":            "mem_rss_kb",
    "VMSize":         "mem_avail_kb",
    "ReadMB":         "read_kb",
    "WriteMB":        "write_kb",
}

_KEEP = [
    "timestamp", "node_id",
    "cpu_util_pct",
    "mem_rss_kb", "mem_avail_kb", "mem_pct_utilization",
    "read_kb", "write_kb",
]


# --------------------------------------------------------------------------- #
# File loading                                                                 #
# --------------------------------------------------------------------------- #

def _load_cpu_file(path: str, cpus_per_node: float) -> pandas.DataFrame | None:
    raw = read_csv_from_s3(path)
    if raw is None or raw.empty:
        return None

    # Coerce numeric columns — guard against stray strings
    for col in ["CPUUtilization", "RSS", "VMSize", "ReadMB", "WriteMB", "EpochTime"]:
        if col in raw.columns:
            raw[col] = pandas.to_numeric(raw[col], errors="coerce")

    # Drop internal accounting steps
    raw = raw.loc[~raw["Step"].isin(["-4", "-1", -4, -1])].copy()
    if raw.empty:
        return None

    raw["timestamp"] = pandas.to_datetime(
        raw["EpochTime"], unit="s", origin="unix", utc=True
    )
    raw = raw.rename(columns=_RENAME)

    # Normalise per-row BEFORE any aggregation:
    #   cpu_util_pct = CPUUtilization / cpus_per_node
    if cpus_per_node and cpus_per_node > 0:
        raw["cpu_util_pct"] = raw["cpu_util_raw"] / cpus_per_node
    else:
        raw["cpu_util_pct"] = raw["cpu_util_raw"]

    # MB -> KB
    raw["read_kb"]  = raw["read_kb"]  * 1024.0
    raw["write_kb"] = raw["write_kb"] * 1024.0

    # Memory utilisation: RSS / VMSize
    raw["mem_pct_utilization"] = np.where(
        raw["mem_avail_kb"] > 0,
        raw["mem_rss_kb"] / raw["mem_avail_kb"],
        np.nan,
    )

    keep = [c for c in _KEEP if c in raw.columns]
    return raw[keep].dropna(subset=["timestamp"])


# --------------------------------------------------------------------------- #
# Per-node stat helpers                                                        #
# --------------------------------------------------------------------------- #

def _per_node_stats(
    df: pandas.DataFrame,
    col: str,
    prefix: str,
) -> dict:
    """
    Compute all distribution stats per node, then average across nodes.
    Captures temporal variability within a typical node, not cross-node
    variance (which is already captured by spatial RI).
    """
    if col not in df.columns or df.empty:
        return {
            f"{prefix}_mean":     np.nan,
            f"{prefix}_max":      np.nan,
            f"{prefix}_min":      np.nan,
            f"{prefix}_std":      np.nan,
            f"{prefix}_skew":     np.nan,
            f"{prefix}_kurtosis": np.nan,
            f"{prefix}_iof":      np.nan,
        }

    def node_stats(s):
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
    # pandas versions. Equivalent to groupby(level=1).mean() but unambiguous.
    records = []
    for node_id, group in df.groupby("node_id"):
        records.append(node_stats(group[col]))
    if not records:
        per_node = pandas.Series({
            "mean": np.nan, "max": np.nan, "min": np.nan,
            "std": np.nan, "skew": np.nan, "kurtosis": np.nan, "iof": np.nan,
        })
    else:
        per_node = pandas.DataFrame(records).mean()

    return {
        f"{prefix}_mean":     float(per_node.get("mean",     np.nan)),
        f"{prefix}_max":      float(per_node.get("max",      np.nan)),
        f"{prefix}_min":      float(per_node.get("min",      np.nan)),
        f"{prefix}_std":      float(per_node.get("std",      np.nan)),
        f"{prefix}_skew":     float(per_node.get("skew",     np.nan)),
        f"{prefix}_kurtosis": float(per_node.get("kurtosis", np.nan)),
        f"{prefix}_iof":      float(per_node.get("iof",      np.nan)),
    }


def _per_node_percentiles(
    df: pandas.DataFrame,
    col: str,
    prefix: str,
) -> dict:
    """Percentiles computed per node then averaged across nodes."""
    if col not in df.columns or df.empty:
        return {f"{prefix}_p{int(q*100)}": np.nan for q in QUANTILES}

    result = {}
    for q in QUANTILES:
        per_node = df.groupby("node_id")[col].quantile(q)
        result[f"{prefix}_p{int(q*100)}"] = float(per_node.mean())
    return result


def _per_node_time_above(
    df: pandas.DataFrame,
    col: str,
    threshold: float,
    out_key: str,
) -> dict:
    """Fraction of timesteps above threshold per node, averaged across nodes."""
    if col not in df.columns or df.empty:
        return {out_key: np.nan}

    def frac(s):
        s = s.dropna()
        return (s > threshold).sum() / len(s) if len(s) > 0 else np.nan

    val = df.groupby("node_id")[col].apply(frac).mean()
    return {out_key: float(val)}


# --------------------------------------------------------------------------- #
# Main summarization                                                           #
# --------------------------------------------------------------------------- #

def summarize_cpu(
    cpu_file: str,
    base_folder: str,   # kept for API symmetry
    job_row: pandas.Series,
) -> tuple[dict, dict]:
    """
    Returns:
        summary     : flat dict of scalar CPU metrics
        ts_by_node  : {node_id: df}  (raw ts, for RI only)
    """
    cpus_req  = float(job_row.get("cpus_req",    0) or 0)
    num_nodes = float(job_row.get("nodes_alloc", 1) or 1)
    if num_nodes <= 0:
        num_nodes = 1.0

    cpus_per_node = cpus_req / num_nodes if cpus_req > 0 else 0.0

    df = _load_cpu_file(cpu_file, cpus_per_node)
    if df is None or df.empty:
        return {}, {}

    # ts_by_node for RI — before any aggregation
    ts_by_node = {
        node: grp.reset_index(drop=True)
        for node, grp in df.groupby("node_id")
    }

    summary = {}

    # ------------------------------------------------------------------ #
    # CPU utilisation                                                      #
    # All stats: per-node then averaged → temporal variability of typical node
    # ------------------------------------------------------------------ #
    summary.update(_per_node_stats(df, "cpu_util_pct", "cpu_util"))
    summary.update(_per_node_percentiles(df, "cpu_util_pct", "cpu_util"))
    summary.update(_per_node_time_above(df, "cpu_util_pct", CPU_BUSY_THRESHOLD, "cpu_time_above_50pct"))

    # ------------------------------------------------------------------ #
    # Memory RSS                                                           #
    # ------------------------------------------------------------------ #
    summary.update(_per_node_stats(df, "mem_rss_kb", "mem_rss_kb"))
    summary.update(_per_node_percentiles(df, "mem_rss_kb", "mem_rss_kb"))

    # ------------------------------------------------------------------ #
    # Memory available                                                     #
    # ------------------------------------------------------------------ #
    summary.update(_per_node_stats(df, "mem_avail_kb", "mem_avail_kb"))

    # ------------------------------------------------------------------ #
    # Memory utilisation pct (RSS / VMSize)                               #
    # ------------------------------------------------------------------ #
    summary.update(_per_node_stats(df, "mem_pct_utilization", "mem_pct_util"))
    summary.update(_per_node_percentiles(df, "mem_pct_utilization", "mem_pct_util"))
    summary.update(_per_node_time_above(df,"mem_pct_utilization",0.9,"mem_util_time_above_90pct"))

    # Memory pressure: peak RSS / mean available (OOM proximity)
    max_rss    = df["mem_rss_kb"].max()    if "mem_rss_kb"   in df.columns else np.nan
    mean_avail = df["mem_avail_kb"].mean() if "mem_avail_kb" in df.columns else np.nan
    summary["mem_pressure"] = (
        float(max_rss / mean_avail)
        if (pandas.notna(max_rss) and pandas.notna(mean_avail) and mean_avail > 0)
        else np.nan
    )

    # ------------------------------------------------------------------ #
    # I/O                                                                  #
    # std/skew etc: per-node averaged (temporal variability of typical node)
    # totals: summed across nodes (job-level physical I/O)                #
    # ------------------------------------------------------------------ #
    summary.update(_per_node_stats(df, "read_kb",  "read_kb"))
    summary.update(_per_node_stats(df, "write_kb", "write_kb"))

    # Job-level totals
    summary["read_kb_total"]  = float(df["read_kb"].sum())  if "read_kb"  in df.columns else np.nan
    summary["write_kb_total"] = float(df["write_kb"].sum()) if "write_kb" in df.columns else np.nan

    return summary, ts_by_node