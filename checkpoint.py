"""
checkpoint.py — checkpointing and resume utilities.

Layout of ./processed_data/:
    chunks/
        chunk_0001.pkl.zst   — completed chunk 1 (permanent, never overwritten)
        chunk_0002.pkl.zst   — completed chunk 2
        ...
    done_jobs.csv            — append-only superset of ALL successfully completed
                               job_ids. Never includes failed jobs.
    errors.csv               — job_id,error_message for failed jobs.
                               These are retried on next resume run since they
                               are NOT in done_jobs.csv.

Resume flow:
    1. Read done_jobs.csv → filter task list
    2. Read max chunk index in chunks/ → offset new chunks to avoid overwriting
    3. Chunk file existence guard handles the rare crash between save_chunk()
       and append_done_job() — reads chunk file, commits missing job_ids, skips.

done_jobs.csv:
    - Appended to ONLY after save_chunk() succeeds
    - Never includes failed job_ids (so they are retried on resume)
    - Append-only, never overwritten
"""

import csv
import logging
import os

import pandas

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = "./processed_data"
CHUNKS_DIR         = os.path.join(PROCESSED_DATA_DIR, "chunks")
DONE_JOBS_FILE     = os.path.join(PROCESSED_DATA_DIR, "done_jobs.csv")
ERRORS_FILE        = os.path.join(PROCESSED_DATA_DIR, "errors.csv")


def ensure_dirs():
    """Create ./processed_data/chunks/ if needed."""
    os.makedirs(CHUNKS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Chunk files                                                                  #
# --------------------------------------------------------------------------- #

def chunk_path(chunk_idx: int) -> str:
    return os.path.join(CHUNKS_DIR, f"chunk_{chunk_idx:04d}.pkl.zst")


def get_max_chunk_idx() -> int:
    """
    Return the highest chunk index already saved in chunks/.
    Returns 0 if no chunks exist.
    New chunks should start at get_max_chunk_idx() + 1.
    """
    if not os.path.exists(CHUNKS_DIR):
        return 0
    indices = []
    for fname in os.listdir(CHUNKS_DIR):
        if fname.startswith("chunk_") and fname.endswith(".pkl.zst"):
            try:
                idx = int(fname.replace("chunk_", "").replace(".pkl.zst", ""))
                indices.append(idx)
            except ValueError:
                pass
    return max(indices) if indices else 0


def save_chunk(results: list[dict], chunk_idx: int):
    """Save a completed chunk's results to its permanent chunk file."""
    if not results:
        logger.warning("Chunk %d: no results to save.", chunk_idx)
        return
    path = chunk_path(chunk_idx)
    pandas.DataFrame.from_records(results).to_pickle(path, compression="zstd")
    logger.info("Chunk %d saved: %d rows → %s", chunk_idx, len(results), path)


def load_all_chunks() -> pandas.DataFrame | None:
    """Load and concat all existing chunk files for final output assembly."""
    if not os.path.exists(CHUNKS_DIR):
        return None
    chunk_files = sorted(
        f for f in os.listdir(CHUNKS_DIR)
        if f.startswith("chunk_") and f.endswith(".pkl.zst")
    )
    if not chunk_files:
        return None
    dfs = []
    for fname in chunk_files:
        path = os.path.join(CHUNKS_DIR, fname)
        try:
            dfs.append(pandas.read_pickle(path))
        except Exception as exc:
            logger.error("Failed to load chunk file %s: %s", path, exc)
    if not dfs:
        return None
    result = pandas.concat(dfs, ignore_index=True)
    logger.info("Loaded %d chunk files → %d total rows.", len(dfs), len(result))
    return result


# --------------------------------------------------------------------------- #
# done_jobs.csv                                                                #
# --------------------------------------------------------------------------- #

def load_done_jobs() -> set:
    """
    Return set of all successfully completed job_ids (strings).
    Does NOT include failed jobs — those are retried on resume.
    """
    if not os.path.exists(DONE_JOBS_FILE):
        return set()
    done = set()
    with open(DONE_JOBS_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)   # skip header
        for row in reader:
            if row:
                done.add(row[0].strip())
    logger.info("done_jobs.csv: %d completed job_ids.", len(done))
    return done


def append_done_jobs(job_ids: list):
    """
    Append a list of successfully completed job_ids to done_jobs.csv.
    Called ONLY after save_chunk() succeeds.
    Failed jobs are never passed here — they go to errors.csv instead.
    """
    with open(DONE_JOBS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        for job_id in job_ids:
            writer.writerow([str(job_id)])


# --------------------------------------------------------------------------- #
# errors.csv                                                                   #
# --------------------------------------------------------------------------- #

def append_error(job_id, error_msg: str):
    """
    Log a failed job to errors.csv.
    Failed jobs are NOT added to done_jobs.csv so they are retried on resume.
    """
    write_header = not os.path.exists(ERRORS_FILE)
    with open(ERRORS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["job_id", "error"])
        writer.writerow([str(job_id), str(error_msg)])