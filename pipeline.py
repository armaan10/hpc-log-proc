"""
HPC Job Pipeline — single-pass, S3 streaming, multiprocessing, checkpointing.

Usage:
    python pipeline.py <slurm_csv> <file_index_json> <output_file.pkl.zst>
                       [--workers 64] [--chunk-size 500] [--dry-run]

Resume:
    Just re-run the same command. The pipeline will:
      - Filter out already-done jobs via done_jobs.csv
      - Offset new chunk indices past existing chunk files
      - Retry any previously failed jobs (not in done_jobs.csv)

Flow per chunk:
    1. Process all jobs → collect results + successful/failed job_ids
    2. save_chunk()        → chunk file written to disk
    3. append_done_jobs()  → successful job_ids committed to done_jobs.csv
    (failed jobs go to errors.csv and are retried on next run)

Chunk file guard:
    If crash between save_chunk() and append_done_jobs(), the chunk file
    exists but job_ids are not in done_jobs.csv. On resume, the guard
    detects the existing file, commits missing job_ids, and skips the chunk.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas
from tqdm import tqdm

from checkpoint import (
    append_done_jobs,
    append_error,
    chunk_path,
    ensure_dirs,
    get_max_chunk_idx,
    load_all_chunks,
    load_done_jobs,
    save_chunk,
)
from cpu_processing import summarize_cpu
from gpu_processing import summarize_gpu
from ri import calculate_ri_for_job
from slurm_utils import clean_slurm_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

GPU_USE_THRESHOLD = 2.0


# --------------------------------------------------------------------------- #
# Worker function — runs in a separate process                                 #
# --------------------------------------------------------------------------- #

def process_job(args: tuple) -> dict | None:
    """
    Process a single job end-to-end.
    Runs in a worker process — no shared state, no locks needed.
    Returns summary dict or None if job should be skipped.
    """
    job_row_dict, file_entry = args
    job_row = pandas.Series(job_row_dict)
    job_id  = int(job_row["id_job"])

    slurm     = clean_slurm_row(job_row)
    runtime_s = slurm["runtime_seconds"]

    if runtime_s <= 10:
        return None

    is_gpu_job     = len(file_entry.get("gpu", [])) > 0
    gpu_summary    = {}
    gpu_ts_by_node = {}
    cpu_summary    = {}
    cpu_ts_by_node = {}

    if is_gpu_job:
        try:
            gpu_summary, gpu_ts_by_node = summarize_gpu(
                gpu_files      = file_entry["gpu"],
                base_folder    = "",
                job_id         = job_id,
                threshold      = GPU_USE_THRESHOLD,
                runtime_s      = runtime_s,
                num_alloc_gpus = int(slurm.get("num_alloc_gpus", 0) or 0),
            )
        except Exception as exc:
            logger.warning("Job %s: GPU processing failed: %s", job_id, exc)
            is_gpu_job = False

    cpu_files = file_entry.get("cpu", [])
    if cpu_files:
        try:
            cpu_summary, cpu_ts_by_node = summarize_cpu(
                cpu_file    = cpu_files[0],
                base_folder = "",
                job_row     = job_row,
            )
        except Exception as exc:
            logger.warning("Job %s: CPU processing failed: %s", job_id, exc)

    ri_results = calculate_ri_for_job(
        gpu_ts_by_node = gpu_ts_by_node,
        cpu_ts_by_node = cpu_ts_by_node,
        is_gpu_job     = is_gpu_job,
    )

    summary = {"job_id": job_id, "is_gpu_job": is_gpu_job}
    summary.update(slurm)
    summary.update(cpu_summary)
    summary.update(gpu_summary)
    summary.update(ri_results)
    return summary


# --------------------------------------------------------------------------- #
# Chunk processing — one pool lifecycle                                        #
# --------------------------------------------------------------------------- #

def process_chunk(
    chunk: list[tuple],
    chunk_idx: int,
    workers: int,
    pbar: tqdm,
    t0: float,
) -> tuple[list[dict], list[str], list[str]]:
    """
    Process one chunk with a fresh ProcessPoolExecutor.
    Pool destroyed on return — OS reclaims all worker memory.

    Returns:
        results       : result dicts for successful jobs
        succeeded_ids : job_ids that completed successfully
        failed_ids    : job_ids that raised exceptions (to be retried)
    """
    results       = []
    succeeded_ids = []
    failed_ids    = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_job, task): task for task in chunk}

        for future in as_completed(futures):
            task       = futures[future]
            job_id_str = str(int(task[0]["id_job"]))

            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                # None result = skipped (runtime <= 10s) — still mark done
                succeeded_ids.append(job_id_str)

            except Exception as exc:
                logger.error("Job %s failed: %s", job_id_str, exc)
                append_error(job_id_str, str(exc))
                failed_ids.append(job_id_str)
                # NOT added to succeeded_ids — will be retried on resume

            elapsed = time.time() - t0
            rate    = pbar.n / elapsed if elapsed > 0 else 0
            pbar.set_postfix(
                rate    = f"{rate:.1f} jobs/s",
                chunk   = chunk_idx,
                results = len(results),
                failed  = len(failed_ids),
                refresh = False,
            )
            pbar.update(1)

    # Pool destroyed here
    logger.info("Chunk %d: pool shutdown complete.", chunk_idx)
    return results, succeeded_ids, failed_ids


# --------------------------------------------------------------------------- #
# Task list builder                                                            #
# --------------------------------------------------------------------------- #

def build_job_list(
    slurm_csv: str,
    file_index: dict,
    done_jobs: set,
) -> list[tuple]:
    """
    Build list of (job_row_dict, file_entry) tuples.
    Filters out jobs with no file entry and jobs in done_jobs.csv.
    Failed jobs are NOT in done_jobs.csv so they appear here and get retried.
    """
    job_meta = pandas.read_csv(slurm_csv)
    job_meta = job_meta.sort_values('time_submit')
    logger.info("Slurm CSV: %d total rows.", len(job_meta))

    tasks            = []
    skipped_no_files = 0
    skipped_done     = 0

    for _, row in job_meta.iterrows():
        job_id     = str(int(row["id_job"]))
        file_entry = file_index.get(job_id)

        if file_entry is None:
            skipped_no_files += 1
            continue
        if job_id in done_jobs:
            skipped_done += 1
            continue

        tasks.append((row.to_dict(), file_entry))

    logger.info(
        "Jobs to process: %d  |  no files: %d  |  already done: %d",
        len(tasks), skipped_no_files, skipped_done,
    )
    return tasks


# --------------------------------------------------------------------------- #
# Chunk file existence guard                                                   #
# --------------------------------------------------------------------------- #

def _handle_existing_chunk(chunk_idx: int, done_jobs: set) -> bool:
    """
    If chunk_idx file already exists, commit any job_ids not yet in
    done_jobs.csv (handles crash between save_chunk and append_done_jobs),
    then signal the caller to skip this chunk.

    Returns True if chunk already exists (skip it), False otherwise.
    """
    path = chunk_path(chunk_idx)
    if not os.path.exists(path):
        return False

    logger.info(
        "Chunk %d: file already exists — committing any missing job_ids "
        "to done_jobs.csv and skipping.", chunk_idx
    )
    try:
        df      = pandas.read_pickle(path)
        missing = [
            str(jid) for jid in df["job_id"].astype(str).tolist()
            if str(jid) not in done_jobs
        ]
        if missing:
            append_done_jobs(missing)
            done_jobs.update(missing)
            logger.info("Committed %d missing job_ids from chunk %d.", len(missing), chunk_idx)
    except Exception as exc:
        logger.error("Could not read existing chunk %d: %s", chunk_idx, exc)

    return True


# --------------------------------------------------------------------------- #
# Main orchestration                                                           #
# --------------------------------------------------------------------------- #

def run(
    slurm_csv: str,
    file_index_json: str,
    output_path: str,
    workers: int,
    chunk_size: int,
    dry_run: bool,
):
    ensure_dirs()

    logger.info("Loading file index from %s ...", file_index_json)
    with open(file_index_json, "r") as f:
        file_index = json.load(f)
    logger.info("File index: %d job entries.", len(file_index))

    # Step 1: read done_jobs.csv
    done_jobs = load_done_jobs()

    # Step 2: build filtered task list
    tasks = build_job_list(slurm_csv, file_index, done_jobs)

    # Step 3: split into chunks
    chunks   = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    n_chunks = len(chunks)

    # Step 4: offset chunk indices past existing chunk files
    chunk_offset = get_max_chunk_idx()

    if dry_run:
        print(f"\n[DRY RUN] Would process {len(tasks)} jobs with {workers} workers.")
        print(f"          Already done:    {len(done_jobs)} jobs.")
        print(f"          Chunk size:      {chunk_size}  ({n_chunks} pool cycles)")
        print(f"          Chunk offset:    {chunk_offset} existing chunks")
        print(f"          Output:          {output_path}")
        return

    if not tasks:
        logger.info("Nothing to process — assembling output from existing chunks.")
        _assemble_output(output_path)
        return

    logger.info(
        "Processing %d jobs in %d chunks of ~%d  |  %d workers  |  chunk offset %d.",
        len(tasks), n_chunks, chunk_size, workers, chunk_offset,
    )

    signal.signal(signal.SIGINT,  lambda s, f: (
        logger.warning("Interrupted — done_jobs.csv is up to date. Re-run to resume."),
        sys.exit(1),
    ))
    signal.signal(signal.SIGTERM, lambda s, f: (
        logger.warning("Terminated — done_jobs.csv is up to date. Re-run to resume."),
        sys.exit(1),
    ))

    t0 = time.time()

    with tqdm(total=len(tasks), unit="job", dynamic_ncols=True) as pbar:
        for i, chunk in enumerate(chunks):
            chunk_idx = chunk_offset + i + 1  # 1-indexed, offset past existing

            logger.info(
                "Chunk %d (%d/%d) — %d jobs.",
                chunk_idx, i + 1, n_chunks, len(chunk),
            )

            # Chunk file guard — handles crash between save_chunk and append_done_jobs
           # if _handle_existing_chunk(chunk_idx, done_jobs):
            #    pbar.update(len(chunk))
            #    continue

            try:
                # Step 5a: process all jobs
                results, succeeded_ids, failed_ids = process_chunk(
                    chunk     = chunk,
                    chunk_idx = chunk_idx,
                    workers   = workers,
                    pbar      = pbar,
                    t0        = t0,
                )
                logger.info("Chunk %d: process_chunk returned — %d results, %d succeeded, %d failed.",
            chunk_idx, len(results), len(succeeded_ids), len(failed_ids))
                if failed_ids:
                    logger.warning(
                        "Chunk %d: %d jobs failed and will be retried on resume.",
                        chunk_idx, len(failed_ids),
                    )

                # Step 5b: save chunk file
                save_chunk(results, chunk_idx)
                logger.info("Chunk %d: save_chunk complete.", chunk_idx)
                # Step 5c: commit successful job_ids to done_jobs.csv
                append_done_jobs(succeeded_ids)
                logger.info("Chunk %d: append_done_jobs complete.", chunk_idx)
                done_jobs.update(succeeded_ids)

            except Exception as exc:
                logger.error(
                    "Chunk %d failed: %s — re-run to resume.", chunk_idx, exc,
                )

    elapsed = time.time() - t0
    logger.info("All chunks complete in %.1fs.", elapsed)
    _assemble_output(output_path)


def _assemble_output(output_path: str):
    """Concat all chunk files into the final output pickle."""
    final_df = load_all_chunks()
    if final_df is None or final_df.empty:
        logger.warning("No chunk files found — nothing to assemble.")
        return
    final_df.to_pickle(output_path, compression="zstd")
    logger.info("Final output: %d rows → %s", len(final_df), output_path)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="HPC Job Pipeline — stream from S3, process in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("slurm_csv",       help="Local slurm metadata CSV.")
    parser.add_argument("file_index_json", help="JSON mapping job_id to S3 paths.")
    parser.add_argument("output",          help="Output file path (.pkl.zst).")
    parser.add_argument("--workers",    type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=500, dest="chunk_size",
        help="Jobs per pool cycle. Smaller = more frequent memory resets.")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    return parser.parse_args()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    args = parse_args()
    run(
        slurm_csv       = args.slurm_csv,
        file_index_json = args.file_index_json,
        output_path     = args.output,
        workers         = args.workers,
        chunk_size      = args.chunk_size,
        dry_run         = args.dry_run,
    )