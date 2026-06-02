"""
overview_plots.py — High-level distribution plots for HPC dataset overview.

Usage:
    python overview_plots.py <input.pkl.zst> [--output-dir ./plots]

Produces one PNG per plot topic:
    01_jobs_by_month.png
    02_jobs_by_day_of_week.png
    03_nodes_allocated.png
    04_cpu_vs_gpu_jobs.png
    05_error_rate_by_month.png
    06_job_duration.png
    07_daily_throughput.png
    08_partition_breakdown.png
    09_node_gpu_hours_over_time.png
    10_runtime_by_job_type.png
"""

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Style                                                                        #
# --------------------------------------------------------------------------- #

PALETTE    = sns.color_palette("muted")
GPU_COLOR  = PALETTE[0]   # blue
CPU_COLOR  = PALETTE[1]   # orange
ERR_COLOR  = PALETTE[3]   # red
OK_COLOR   = PALETTE[2]   # green
CDF_COLOR  = "#444444"
FIG_DPI    = 150
PARTITIONS = ["normal", "test", "gaia", "xeon-p8", "db"]
PART_COLORS= dict(zip(PARTITIONS, sns.color_palette("tab10", len(PARTITIONS))))

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi":       FIG_DPI,
    "savefig.dpi":      FIG_DPI,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
plt.rcParams.update({
    "font.family":      "monospace",       # utilitarian/technical feel
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.spines.left": False,
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.4,
    "grid.alpha":       0.5,
    "axes.axisbelow":   True,
    # "figure.facecolor": "#F7F6F2",
    #"axes.facecolor":   "#F7F6F2",
})

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


def add_cdf(ax, data, color=CDF_COLOR, label="CDF"):
    """Overlay a CDF on a twin y-axis."""
    ax2 = ax.twinx()
    sorted_data = np.sort(data.dropna())
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax2.plot(sorted_data, cdf, color=color, lw=1.5, label=label)
    ax2.set_ylabel("CDF", color=color)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.spines["top"].set_visible(False)
    return ax2


def format_month_axis(ax):
    ax.tick_params(axis="x", rotation=45)


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_pickle(path)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Parse timestamps
    for col in ["start_timestamp", "end_timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Derive time columns
    if "start_timestamp" in df.columns:
        df["month"]       = df["start_timestamp"].dt.to_period("M").astype(str)
        df["day_of_week"] = df["start_timestamp"].dt.dayofweek   # 0=Mon
        df["date"]        = df["start_timestamp"].dt.date

    # Normalise partition
    if "partition" in df.columns:
        df["partition"] = df["partition"].str.strip().str.lower().fillna("unknown")
    elif "queue_name" in df.columns:
        df["partition"] = df["queue_name"].str.strip().str.lower().fillna("unknown")
        print("here")

    # is_gpu_job fallback
    if "is_gpu_job" not in df.columns:
        df["is_gpu_job"] = df.get("num_alloc_gpus", 0) > 0

    # runtime in hours
    if "runtime_seconds" in df.columns:
        df["runtime_hours"] = df["runtime_seconds"] / 3600.0

    # success flag
    if "exit_code" in df.columns:
        df["exit_code_num"] = pd.to_numeric(df["exit_code"], errors="coerce").fillna(0)
        df["is_success"]    = df["exit_code_num"] == 0
    
    df.loc[(df['is_gpu_job'] == False) & (df['num_alloc_gpus'] > 0), 'is_gpu_job'] = True
    return df


# --------------------------------------------------------------------------- #
# 1. Jobs submitted by month                                                   #
# --------------------------------------------------------------------------- #

def plot_jobs_by_month(df, out_dir):
    if "month" not in df.columns:
        print("  [skip] no start_timestamp")
        return

    counts = df.groupby("month").size().reset_index(name="count")
    counts = counts.sort_values("month")

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(counts["month"], counts["count"],
                  color=GPU_COLOR, alpha=0.8, width=0.6)

    # annotate bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + counts["count"].max() * 0.01,
                f"{int(h):,}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of jobs")
    ax.set_title("Jobs submitted per month", fontweight = 'bold')
    format_month_axis(ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    save(fig, os.path.join(out_dir, "01_jobs_by_month.png"))


# --------------------------------------------------------------------------- #
# 2. Jobs by day of week                                                       #
# --------------------------------------------------------------------------- #

def plot_jobs_by_day(df, out_dir):
    if "day_of_week" not in df.columns:
        print("  [skip] no start_timestamp")
        return

    counts = df["day_of_week"].value_counts().reindex(range(7), fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([DAY_LABELS[i] for i in range(7)], counts.values,
                  color=GPU_COLOR, alpha=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + counts.max() * 0.01,
                f"{int(h):,}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Day of week")
    ax.set_ylabel("Number of jobs")
    ax.set_title("Jobs submitted by day of week")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    save(fig, os.path.join(out_dir, "02_jobs_by_day_of_week.png"))


# --------------------------------------------------------------------------- #
# 3. Nodes allocated distribution                                              #
# --------------------------------------------------------------------------- #
def plot_nodes_allocated_2(df, out_dir):
  

    nodes = df['num_nodes']

    # Basic stats
    total_jobs = len(nodes)
    one_node_pct = (nodes == 1).mean() * 100
    median_nodes = nodes.median()
    p90 = nodes.quantile(0.9)
    p99 = nodes.quantile(0.99)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ----------------------
    # (1) Histogram (log-y)
    # ----------------------
    axes[0].hist(nodes, bins=50)
    axes[0].set_yscale('log')
    axes[0].set_xlabel("Nodes Allocated")
    axes[0].set_ylabel("Count (log scale)")
    axes[0].set_title("Histogram of Nodes per Job")

    # Annotate key stat
    axes[0].text(
        0.95, 0.95,
        f"{one_node_pct:.1f}% jobs use 1 node",
        transform=axes[0].transAxes,
        ha='right',
        va='top'
    )

    # ----------------------
    # (2) CDF (log-x)
    # ----------------------
    sorted_vals = np.sort(nodes)
    cdf = np.arange(len(sorted_vals)) / len(sorted_vals)

    axes[1].plot(sorted_vals, cdf)
    axes[1].set_xscale('log')
    axes[1].set_xlabel("Nodes Allocated (log scale)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("CDF of Nodes per Job")

    # Annotate percentiles
    axes[1].axvline(p90, linestyle='--')
    axes[1].axvline(p99, linestyle='--')

    axes[1].text(p90, 0.9, "p90", rotation=90, va='bottom')
    axes[1].text(p99, 0.99, "p99", rotation=90, va='top')
    fig.suptitle('Node Allocation Distribution', fontweight = 'bold')
    save(fig, os.path.join(out_dir, "03_nodes_allocated_v2.png"))

def plot_nodes_allocated(df, out_dir):
    col = "num_nodes"
    if col not in df.columns:
        print("  [skip] no num_nodes")
        return

    data = pd.to_numeric(df[col], errors="coerce").dropna()
    data = data[data > 0]

    counts = data.value_counts().sort_index()
    # cap display at 99th percentile
    cap = int(data.quantile(0.99)) + 1
    counts = counts[counts.index <= cap]

    fig, ax = plt.subplots(figsize=(12, 5))
    #ax.bar(counts.index.astype(int), counts.values, color=GPU_COLOR, alpha=0.8, width=0.8)

    add_cdf(ax, data[data <= cap])

    ax.set_xlabel("Number of nodes allocated")
    ax.set_ylabel("Job count")
    ax.set_title("Distribution of nodes allocated per job  (CDF overlay, 99th pct cap)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    p50 = data.median()
    p95 = data.quantile(0.95)
    ax.axvline(p50, color="orange", lw=1.5, linestyle="--", label=f"p50 = {p50:.0f}")
    ax.axvline(p95, color="red",    lw=1.5, linestyle="--", label=f"p95 = {p95:.0f}")
    ax.legend(loc="upper right")

    save(fig, os.path.join(out_dir, "03_nodes_allocated.png"))


# --------------------------------------------------------------------------- #
# 4. CPU-only vs GPU jobs                                                      #
# --------------------------------------------------------------------------- #

def plot_cpu_vs_gpu(df, out_dir):
    if "is_gpu_job" not in df.columns:
        print("  [skip] no is_gpu_job")
        return

    # Overall pie
    counts = df["is_gpu_job"].value_counts()
    print(counts)
    labels = {True: "GPU jobs", False: "CPU-only jobs"}
    colors = {True: GPU_COLOR, False: CPU_COLOR}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall pie
    ax = axes[0]
    wedge_labels = [labels[k] for k in counts.index]
    wedge_colors = [colors[k] for k in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=wedge_labels, colors=wedge_colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(11)
    ax.set_title("Overall job type split")

    # Right: stacked bar by month
    ax2 = axes[1]
    if "month" in df.columns:
        monthly = (
            df.groupby(["month", "is_gpu_job"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        months = monthly.index.tolist()
        cpu_counts = monthly.get(False, pd.Series(0, index=months)).values
        gpu_counts = monthly.get(True,  pd.Series(0, index=months)).values

        x = range(len(months))
        ax2.bar(x, cpu_counts, label="CPU-only", color=CPU_COLOR, alpha=0.85)
        ax2.bar(x, gpu_counts, bottom=cpu_counts, label="GPU", color=GPU_COLOR, alpha=0.85)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(months, rotation=45, ha="right")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Job count")
        ax2.set_title("CPU vs GPU jobs per month")
        ax2.legend()
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    else:
        ax2.axis("off")

    fig.suptitle("CPU-only vs GPU job distribution", fontweight="bold")
    save(fig, os.path.join(out_dir, "04_cpu_vs_gpu_jobs.png"))


# --------------------------------------------------------------------------- #
# 5. Error rate by month                                                       #
# --------------------------------------------------------------------------- #

def plot_error_rate_by_month(df, out_dir):
    if "is_success" not in df.columns or "month" not in df.columns:
        print("  [skip] no exit_code or start_timestamp")
        return

    monthly = df.groupby("month").agg(
        total   = ("is_success", "count"),
        success = ("is_success", "sum"),
    ).sort_index()
    monthly["errors"]     = monthly["total"] - monthly["success"]
    monthly["error_rate"] = monthly["errors"] / monthly["total"] * 100

    fig, ax1 = plt.subplots(figsize=(12, 5))

    x = range(len(monthly))
    ax1.bar(x, monthly["success"], label="Success", color=OK_COLOR,  alpha=0.8)
    ax1.bar(x, monthly["errors"],  label="Error",   color=ERR_COLOR, alpha=0.8,
            bottom=monthly["success"])

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(monthly.index, rotation=45, ha="right")
    ax1.set_ylabel("Job count")
    ax1.set_xlabel("Month")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax1.legend(loc="upper left")

    # error rate line on twin axis
    ax2 = ax1.twinx()
    ax2.plot(list(x), monthly["error_rate"], color="darkred",
             lw=2, marker="o", markersize=5, label="Error rate %")
    ax2.set_ylabel("Error rate (%)", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax2.set_ylim(0, max(monthly["error_rate"].max() * 1.4, 5))
    ax2.spines["top"].set_visible(False)

    ax1.set_title("Job success vs error by month  (error rate % overlay)")
    fig.tight_layout()

    save(fig, os.path.join(out_dir, "05_error_rate_by_month.png"))


# --------------------------------------------------------------------------- #
# 6. Job duration distribution                                                 #
# --------------------------------------------------------------------------- #

def plot_job_duration(df, out_dir):
    if "runtime_hours" not in df.columns:
        print("  [skip] no runtime_seconds")
        return

    data = df["runtime_hours"].dropna()
    data = data[data > 0]

    fig, ax = plt.subplots(figsize=(10, 5))

    # log-scale histogram
    log_data = np.log10(data.clip(lower=1e-3))
    ax.hist(log_data, bins=80, color=GPU_COLOR, alpha=0.75, edgecolor="none")

    add_cdf(ax, log_data)

    # x-axis: show human-readable labels
    tick_vals  = [-2, -1, 0, 1, 2, np.log10(24), np.log10(48), np.log10(168)]
    tick_labels= ["36s", "6m", "1h", "10h", "4d", "24h", "48h", "1wk"]
    valid = [(v, l) for v, l in zip(tick_vals, tick_labels) if log_data.min() <= v <= log_data.max()]
    ax.set_xticks([v for v, _ in valid])
    ax.set_xticklabels([l for _, l in valid])

    p50 = data.median()
    p95 = data.quantile(0.95)
    ax.axvline(np.log10(p50), color="orange", lw=1.5, linestyle="--",
               label=f"p50 = {p50:.1f}h")
    ax.axvline(np.log10(p95), color="red", lw=1.5, linestyle="--",
               label=f"p95 = {p95:.1f}h")

    ax.set_xlabel("Job runtime (log scale)")
    ax.set_ylabel("Job count")
    ax.set_title("Job duration distribution  (log scale, CDF overlay)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    save(fig, os.path.join(out_dir, "06_job_duration.png"))


# --------------------------------------------------------------------------- #
# 7. Daily throughput over time                                                #
# --------------------------------------------------------------------------- #

def plot_daily_throughput(df, out_dir):
    if "date" not in df.columns:
        print("  [skip] no start_timestamp")
        return

    daily = df.groupby("date").size().reset_index(name="count")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    print(daily.sort_values('count',ascending = False))
    # 7-day rolling average
    daily["rolling7"] = daily["count"].rolling(7, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(daily["date"], daily["count"], color=GPU_COLOR, alpha=0.4,
           width=0.9, label="Daily jobs")
    ax.plot(daily["date"], daily["rolling7"], color="darkblue",
            lw=2, label="7-day rolling avg")

    ax.set_xlabel("Date")
    ax.set_ylabel("Jobs submitted")
    ax.set_title("Daily job submission throughput  (7-day rolling average)", fontweight = 'bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend()

    # shade weekends
    for _, row in daily.iterrows():
        if row["date"].dayofweek >= 5:
            ax.axvspan(row["date"] - pd.Timedelta(hours=12),
                       row["date"] + pd.Timedelta(hours=12),
                       alpha=0.06, color="gray", lw=0)

    fig.autofmt_xdate()
    save(fig, os.path.join(out_dir, "07_daily_throughput.png"))


# --------------------------------------------------------------------------- #
# 8. Partition breakdown                                                       #
# --------------------------------------------------------------------------- #

def plot_partition_breakdown(df, out_dir):
    if "partition" not in df.columns:
        print("  [skip] no partition column")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Job count by partition
    ax = axes[0]
    counts = df["partition"].value_counts()
    colors = [PART_COLORS.get(p, PALETTE[4]) for p in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + counts.max() * 0.01,
                f"{int(h):,}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Partition")
    ax.set_ylabel("Job count")
    ax.set_title("Job count by partition")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # GPU-hours by partition
    ax2 = axes[1]
    if "gpu_hours" in df.columns:
        gpu_hrs = df.groupby("partition")["gpu_hours"].sum().sort_values(ascending=False)
        colors2 = [PART_COLORS.get(p, PALETTE[4]) for p in gpu_hrs.index]
        ax2.bar(gpu_hrs.index, gpu_hrs.values, color=colors2, alpha=0.85)
        ax2.set_xlabel("Partition")
        ax2.set_ylabel("GPU-hours")
        ax2.set_title("Total GPU-hours by partition")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    else:
        ax2.axis("off")

    # GPU job % by partition
    ax3 = axes[2]
    if "is_gpu_job" in df.columns:
        gpu_pct = (
            df.groupby("partition")["is_gpu_job"]
            .mean()
            .sort_values(ascending=False) * 100
        )
        colors3 = [PART_COLORS.get(p, PALETTE[4]) for p in gpu_pct.index]
        bars3 = ax3.bar(gpu_pct.index, gpu_pct.values, color=colors3, alpha=0.85)
        for bar in bars3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                     f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
        ax3.set_xlabel("Partition")
        ax3.set_ylabel("% GPU jobs")
        ax3.set_title("GPU job percentage by partition")
        ax3.set_ylim(0, 110)
    else:
        ax3.axis("off")

    fig.suptitle("Partition breakdown", fontweight="bold")
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "08_partition_breakdown.png"))


# --------------------------------------------------------------------------- #
# 9. Node-hours vs GPU-hours over time                                         #
# --------------------------------------------------------------------------- #

def plot_node_gpu_hours(df, out_dir):
    if "month" not in df.columns:
        print("  [skip] no start_timestamp")
        return
    if "node_hours" not in df.columns and "gpu_hours" not in df.columns:
        print("  [skip] no node_hours or gpu_hours")
        return

    monthly = df.groupby("month").agg(
        node_hours = ("node_hours", "sum"),
        gpu_hours  = ("gpu_hours",  "sum"),
    ).sort_index()

    fig, ax1 = plt.subplots(figsize=(12, 5))

    x = range(len(monthly))
    w = 0.35

    ax1.bar([i - w/2 for i in x], monthly["node_hours"], width=w,
            label="Node-hours", color=CPU_COLOR, alpha=0.85)
    ax1.set_ylabel("Node-hours", color=CPU_COLOR)
    ax1.tick_params(axis="y", labelcolor=CPU_COLOR)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    ax2 = ax1.twinx()
    ax2.bar([i + w/2 for i in x], monthly["gpu_hours"], width=w,
            label="GPU-hours", color=GPU_COLOR, alpha=0.85)
    ax2.set_ylabel("GPU-hours", color=GPU_COLOR)
    ax2.tick_params(axis="y", labelcolor=GPU_COLOR)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax2.spines["top"].set_visible(False)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(monthly.index, rotation=45, ha="right")
    ax1.set_xlabel("Month")
    ax1.set_title("Node-hours vs GPU-hours per month")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    save(fig, os.path.join(out_dir, "09_node_gpu_hours_over_time.png"))


# --------------------------------------------------------------------------- #
# 10. Runtime distribution by job type (overlaid CDF)                         #
# --------------------------------------------------------------------------- #

def plot_runtime_by_type(df, out_dir):
    if "runtime_hours" not in df.columns or "is_gpu_job" not in df.columns:
        print("  [skip] no runtime_seconds or is_gpu_job")
        return

    gpu_data = df.loc[df["is_gpu_job"],  "runtime_hours"].dropna()
    cpu_data = df.loc[~df["is_gpu_job"], "runtime_hours"].dropna()
    gpu_data = gpu_data[gpu_data > 0]
    cpu_data = cpu_data[cpu_data > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overlaid histograms (log scale)
    ax = axes[0]
    bins = np.logspace(np.log10(1e-3), np.log10(max(gpu_data.max(), cpu_data.max()) + 1), 60)
    ax.hist(cpu_data, bins=bins, alpha=0.55, color=CPU_COLOR, label="CPU-only", density=True)
    ax.hist(gpu_data, bins=bins, alpha=0.55, color=GPU_COLOR, label="GPU",      density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (hours, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Runtime distribution: CPU vs GPU (overlaid histogram)")
    ax.legend()

    # Right: overlaid CDFs
    ax2 = axes[1]
    for data, label, color in [
        (cpu_data, "CPU-only", CPU_COLOR),
        (gpu_data, "GPU",      GPU_COLOR),
    ]:
        sorted_d = np.sort(data)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax2.plot(sorted_d, cdf, color=color, lw=2, label=label)

        p50 = np.percentile(sorted_d, 50)
        p95 = np.percentile(sorted_d, 95)
        ax2.axvline(p50, color=color, lw=1, linestyle=":", alpha=0.7)

    ax2.set_xscale("log")
    ax2.set_xlabel("Runtime (hours, log scale)")
    ax2.set_ylabel("CDF")
    ax2.set_title("Runtime CDF: CPU vs GPU")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    # Annotate medians
    for data, color in [(cpu_data, CPU_COLOR), (gpu_data, GPU_COLOR)]:
        p50 = data.median()
        ax2.text(p50 * 1.1, 0.52, f"p50={p50:.1f}h", color=color, fontsize=8)

    fig.suptitle("Job runtime by type", fontweight="bold")
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "10_runtime_by_job_type.png"))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="HPC dataset overview plots.")
    parser.add_argument("input",        help="Processed dataset (.pkl.zst)")
    parser.add_argument("--output-dir", default="./plots", dest="output_dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_data(args.input)
    #print(len(df.columns))
    
    plots = [
        ("01 Jobs by month",           plot_jobs_by_month),
        ("02 Jobs by day of week",     plot_jobs_by_day),
        ("03 Nodes allocated",         plot_nodes_allocated_2),
        ("04 CPU vs GPU jobs",         plot_cpu_vs_gpu),
        ("05 Error rate by month",     plot_error_rate_by_month),
        ("06 Job duration",            plot_job_duration),
        ("07 Daily throughput",        plot_daily_throughput),
        ("08 Partition breakdown",     plot_partition_breakdown),
        ("09 Node/GPU hours over time",plot_node_gpu_hours),
        ("10 Runtime by job type",     plot_runtime_by_type),
    ]

    for name, fn in plots:
        print(f"\n→ {name}")
        try:
            fn(df, args.output_dir)
        except Exception as exc:
            print(f"  [ERROR] {exc}")

    print(f"\nDone. Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()