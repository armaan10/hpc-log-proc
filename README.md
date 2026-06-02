# hpc-log-proc

A high-throughput data processing pipeline for HPC job telemetry. Streams CPU and GPU time-series logs directly from Amazon S3, computes per-job resource utilisation summaries and Resource Imbalance metrics, and writes a compressed output dataset — one row per job — suitable for downstream analysis.

Built for the [MIT Supercloud Dataset](https://supercloud.mit.edu/), but future work will include modularizing the codebase to be adaptable to other datasets.
---

## Features

- **S3 streaming** — reads logs directly from a public S3 bucket with no intermediate disk writes, eliminating storage bottlenecks on datasets too large to download in full
- **Parallel processing** — `ProcessPoolExecutor` with configurable worker count (default 64); each job is processed independently with no shared state
- **Chunked pool recycling** — worker processes are recycled every N jobs to prevent heap fragmentation-driven OOM on Python 3.10 (no `max_tasks_per_child` available)
- **Crash-safe checkpointing** — strict save-then-commit ordering ensures no data loss or duplicate rows on resume; failed jobs are retried automatically
- **Resource Imbalance metrics** — temporal and spatial RI computed from raw time-series per job, for both GPU and CPU resources
- **Principled aggregation** — all distribution stats (mean, std, skew, kurtosis, IoF, percentiles) computed per-GPU/per-node first, then averaged, isolating temporal variability from spatial variance
- **Overview plots** — a companion script generates 10 distribution plots for dataset overview

---

## Repository structure

```
hpc-log-proc/
├── pipeline.py          # Entry point — orchestration, CLI, chunked MapReduce loop
├── gpu_processing.py    # GPU log streaming, metric summarisation, aggregation
├── cpu_processing.py    # CPU log streaming, metric summarisation, aggregation
├── ri.py                # Resource Imbalance (temporal + spatial) implementation
├── slurm_utils.py       # Slurm metadata cleaning and field derivation
├── s3_io.py             # S3 client (unsigned), CSV streaming
├── checkpoint.py        # Chunk files, done_jobs.csv, errors.csv management
├── file_index.py        # JSON file index loader (job_id → S3 paths)
├── plots/
│   └── overview_plots.py  # Dataset overview distribution plots
└── tests.py             # Unit test suite (71 tests)
```

---

## Installation

Python 3.10+ required.

```bash
git clone https://github.com/your-username/hpc-log-proc.git
cd hpc-log-proc
pip install -r requirements.txt
```

**`requirements.txt`**
```
boto3
pandas>=2.0
numpy
scipy
tqdm
zstandard
matplotlib
seaborn
```

---

## Usage

### Basic run

```bash
python pipeline.py slurm.csv file_index.json output.pkl.zst
```

### With options

```bash
python pipeline.py slurm.csv file_index.json output.pkl.zst \
    --workers 64 \
    --chunk-size 500
```

### Dry run — check resume state without processing

```bash
python pipeline.py slurm.csv file_index.json output.pkl.zst --dry-run
```

Output:
```
[DRY RUN] Would process 194,312 jobs with 64 workers.
          Already done:    136,163 jobs.
          Chunk size:      500  (389 pool cycles)
          Chunk offset:    272 existing chunks
          Output:          output.pkl.zst
```

### Resume after crash

Just re-run the same command. The pipeline reads `done_jobs.csv` to skip completed jobs, offsets new chunk indices past existing chunk files, and retries any jobs listed in `errors.csv`.

```bash
python pipeline.py slurm.csv file_index.json output.pkl.zst --workers 64
```

### Generate overview plots

```bash
python plots/overview_plots.py output.pkl.zst --output-dir ./plots
```

---

## Arguments and options

| Argument | Description |
|---|---|
| `slurm_csv` | Path to local Slurm job metadata CSV |
| `file_index_json` | Path to JSON mapping `job_id` → `{cpu: [...], gpu: [...]}` S3 paths |
| `output` | Output file path (`.pkl.zst`) |
| `--workers N` | Number of parallel worker processes (default: 64) |
| `--chunk-size N` | Jobs per pool cycle before recycling workers (default: 500) |
| `--dry-run` | Print job counts and resume state, then exit |

---

## Inputs

### Slurm metadata CSV

One row per job. Required columns:

| Column | Description |
|---|---|
| `id_job` | Unique job identifier |
| `time_start` / `time_end` | Unix timestamps |
| `nodes_alloc` | Number of nodes allocated |
| `cpus_req` | Total CPUs requested |
| `tres_alloc` | Resource allocation string (GPU type/count parsed from here) |
| `partition` | Queue/partition name |
| `exit_code` | Job exit code (0 = success) |
| `id_user` | User identifier |

### File index JSON

Pre-generated from the S3 bucket manifest. Maps each job ID to its CPU and GPU log paths relative to the bucket root:

```json
{
  "89213993887039": {
    "cpu": ["datacenter-challenge/202201/cpu/0058/89213993887039-timeseries.csv"],
    "gpu": [
      "datacenter-challenge/202201/gpu/0047/89213993887039-r629115-n830961.csv",
      "datacenter-challenge/202201/gpu/0047/89213993887039-r9175025-n851693.csv"
    ]
  }
}
```

S3 prefix `s3://mit-supercloud-dataset/` is prepended automatically by `s3_io.py`.

---

## Output

A single compressed pickle file (`output.pkl.zst`) — one row per job — readable with:

```python
import pandas as pd
df = pd.read_pickle("output.pkl.zst")
```

### Output columns

**Job metadata**

| Column | Description |
|---|---|
| `job_id` | Job identifier |
| `is_gpu_job` | Boolean — whether GPU logs were found |
| `start_timestamp` / `end_timestamp` | UTC timestamps |
| `runtime_seconds` | Wall-clock runtime |
| `node_hours` / `gpu_hours` | Derived resource-hour totals |
| `num_nodes` / `num_alloc_gpus` | Allocated resources |
| `gpu_type` | `tesla` or `volta` (from `tres_alloc`) |
| `partition` | Queue name |
| `exit_code` | Job exit code |

**CPU metrics** — all distribution stats computed per-node then averaged

| Column | Description |
|---|---|
| `cpu_util_mean/max/min/std/skew/kurtosis/iof` | CPU utilisation (normalised by `cpus_req / num_nodes`) |
| `cpu_util_p25/p75/p95` | Percentiles (per-node averaged) |
| `cpu_time_above_50pct` | Fraction of time CPU utilisation exceeded 50% |
| `mem_rss_kb_*` | RSS memory stats |
| `mem_pct_util_*` | Memory utilisation (RSS / VMSize) |
| `mem_pressure` | Peak RSS / mean available (OOM proximity) |
| `read_kb_* / write_kb_*` | I/O stats and totals |

**GPU metrics** — all distribution stats computed per active GPU then averaged; physical totals summed

| Column | Description |
|---|---|
| `num_gpus_used` | GPUs with mean utilisation > 2% |
| `gpu_idle_allocated_ratio` | `1 - num_gpus_used / num_alloc_gpus` |
| `gpu_util_mean/max/min/std/skew/kurtosis/iof` | SM utilisation |
| `gpu_util_p25/p75/p95` | Percentiles |
| `gpu_time_above_threshold_pct` | Fraction of time GPU util exceeded threshold |
| `gpu_mem_util_*` | Memory utilisation stats |
| `gpu_mem_used_kib_mean/sum/max` | VRAM usage (sum = total across active GPUs) |
| `gpu_temp_*` | Temperature stats |
| `gpu_power_mean_total_w` | Mean total power draw (active GPUs) |
| `gpu_power_peak_w` | Peak total power draw (active GPUs) |
| `gpu_energy_total_wh` | Total energy consumed (active GPUs) |
| `gpu_power_mean_total_including_idle_w` | Mean total power draw (all GPUs in logs) |
| `gpu_energy_total_including_idle_wh` | Total energy including idle GPUs |
| `gpu_power_std_w / skew / kurtosis / min_w` | Power variability stats |
| `time_to_gpu_use_s / minutes_to_gpu_use` | Time from job start to first GPU activity |

**Resource Imbalance metrics** — computed from raw time-series before aggregation

| Column | Description |
|---|---|
| `ri_temporal_gpu_util` | Temporal RI of GPU utilisation |
| `ri_spatial_gpu_util` | Spatial RI of GPU utilisation |
| `ri_temporal_gpu_mem_util` | Temporal RI of GPU memory utilisation |
| `ri_spatial_gpu_mem_util` | Spatial RI of GPU memory utilisation |
| `ri_temporal_gpu_power` | Temporal RI of GPU power draw |
| `ri_spatial_gpu_power` | Spatial RI of GPU power draw |
| `ri_temporal_cpu_util` | Temporal RI of CPU utilisation |
| `ri_spatial_cpu_util` | Spatial RI of CPU utilisation |
| `ri_temporal_mem_rss` | Temporal RI of RSS memory |
| `ri_spatial_mem_rss` | Spatial RI of RSS memory |
| *(+ additional CPU RI columns)* | See `ri.py` for full list |

GPU RI columns are `NaN` for CPU-only jobs.

---

## Checkpoint layout

```
./processed_data/
├── chunks/
│   ├── chunk_0001.pkl.zst   ← completed chunk 1 (permanent)
│   ├── chunk_0002.pkl.zst   ← completed chunk 2
│   └── ...
├── done_jobs.csv            ← append-only, all successfully completed job IDs
└── errors.csv               ← failed jobs with error messages (retried on resume)
```

**Crash safety guarantees:**

| Crash point | On resume |
|---|---|
| Mid-chunk | No chunk file written → full chunk reprocessed |
| After `save_chunk`, before `append_done_jobs` | Guard detects chunk file → commits missing IDs → chunk skipped |
| After full commit | Filtered by `done_jobs.csv` → skipped cleanly |

In all cases: no data loss, no duplicate rows.

---

## Resource Imbalance definition

Temporal and spatial RI are computed per the following definitions, where $U_{n,t}$ is the utilisation of resource $r$ on node $n$ at time $t$:

$$RI_{\text{temporal}}(r) = \max_{1 \le n \le N} \left( 1 - \frac{\sum_{t=0}^{T} U_{n,t}}{\sum_{t=0}^{T} \max_{0 \le t \le T} U_{n,t}} \right)$$

$$RI_{\text{spatial}}(r) = 1 - \frac{\sum_{n=1}^{N} \max_{0 \le t \le T}(U_{n,t})}{\sum_{n=1}^{N} \max_{0 \le t \le T,\, 1 \le n \le N}(U_{n,t})}$$

Both metrics range from 0 (perfectly balanced) to 1 (maximally imbalanced). For GPU jobs, the node-level signal $U_{n,t}$ is the mean utilisation across all GPUs on node $n$ at time $t$.

---

## Running the tests

```bash
python tests.py
```

71 unit tests covering RI formula correctness, GPU/CPU aggregation logic, S3 streaming (mocked), checkpoint behaviour, crash recovery scenarios, and edge cases.

---

## Aggregation design

All temporal distribution statistics (std, skew, kurtosis, IoF, percentiles) are computed **per GPU first, then averaged across active GPUs**. This isolates temporal variability — how a typical GPU behaves over time — from spatial variance, which is quantified separately by the RI metrics. Computing these statistics flat across all GPU × timestep rows would conflate the two sources of variance, producing misleading results when GPUs operate at different steady-state levels.

Physical totals (power, energy, VRAM consumption) are **summed** across GPUs rather than averaged, as they represent real aggregate resource consumption.

---

## Limitations


- CPU I/O columns (`ReadMB`, `WriteMB`) are treated as per-interval deltas. If the source data records cumulative counters, `read_kb_total` and `write_kb_total` will be incorrect.
- Only the first CPU log file per job is processed. If multiple CPU files exist per job (unusual), additional nodes are not included in CPU summaries.
- `gpu_idle_allocated_ratio` may exceed 1.0 if GPU log files contain more GPUs than `tres_alloc` reports, which can occur due to log file naming collisions between jobs.

---

## Citation
The methodology of using 1 row of aggergated data per job is inspired by:

```
@article{cornelius2025extracting,
  title={Extracting Practical, Actionable Energy Insights from Supercomputer Telemetry and Logs},
  author={Cornelius, Melanie and Cross, Greg and Shilpika, Shilpika and Dearing, Matthew T and Lan, Zhiling},
  journal={arXiv preprint arXiv:2505.14796},
  year={2025}
}

```

If you use this pipeline in academic work, please cite the MIT Supercloud dataset:

```
@article{reuther2018interactive,
  title={Interactive supercomputing on 40,000 cores for machine learning and data analysis},
  author={Reuther, Albert and others},
  journal={IEEE High Performance extreme Computing Conference (HPEC)},
  year={2018}
}
```

---

