"""Provenance and atomic-write helpers, shared by every stage.

tsfmvol copy-pastes these into three ingest modules and the copies have drifted, so they are
single-sourced here.
"""
import datetime as dt
import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pandas as pd

from ecalls.config import paths as P


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


SHA_FILE = ".git_sha"   # written by cluster/sync_to_dcc.sh, read where there is no git checkout


def git_sha() -> str:
    """The commit that produced an artifact, with a fallback for machines that have no checkout.

    The cluster copy is an rsync of `src/`, not a clone, so `git rev-parse` fails there and every
    manifest row and parquet sidecar written on a compute node recorded `unknown`. That defeats the
    purpose of the field, since the embeddings are exactly the artifacts whose provenance matters
    most. `cluster/sync_to_dcc.sh` now writes the sha it pushed into `<root>/.git_sha`, and this reads
    it when git is unavailable.
    """
    try:
        return subprocess.check_output(
            ["git", "-C", str(P.PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # noqa: BLE001 -- provenance must never break a pull
        pass
    for base in (P.PROJECT_ROOT, P.DATA_ROOT.parent):
        f = base / SHA_FILE
        try:
            if f.is_file():
                sha = f.read_text().strip()
                if sha:
                    return sha
        except OSError:
            continue
    return "unknown"


def atomic_write_table(table, path: Path) -> None:
    """Write to a private temp file, then rename. The destination's existence is the authoritative
    done-marker, so a kill mid-write must never leave a truncated file that looks complete.

    **The temp name carries the writer's identity, and that is not cosmetic.** It used to be a fixed
    `<name>.tmp`, which is safe for one writer and unsafe for two. On 2026-09-27 a mis-specified SLURM
    array put three tasks on the same shard at once: they interleaved their writes into one temp file
    and the first rename could have moved a corrupt file into place, while the others failed with
    `FileNotFoundError` on a temp file that was no longer there. A per-writer name makes concurrent
    writers merely redundant instead of destructive, since each one renames a file only it wrote.

    This is the one atomic-write path in the package. `atomic_write_parquet` converts a frame and
    routes through here, so the rename semantics cannot drift between callers.
    """
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_sibling(path)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)      # never leave a stray temp behind for the next reader to glob
        raise


def temp_sibling(path: Path) -> Path:
    """A temp path beside `path` that no other writer will pick.

    Host and pid make it diagnosable, so a stray temp file names the node and process that abandoned
    it. The random token makes it *correct*: host and pid alone still collide between threads of one
    process, which a concurrency test caught immediately. It stays in the destination's own directory,
    because `os.replace` is only atomic within a filesystem.
    """
    token = uuid.uuid4().hex[:8]
    return path.with_name(f"{path.name}.{socket.gethostname()}.{os.getpid()}.{token}.tmp")


def claim_dir(d: Path, stale_after: int = 1800, label: str = "") -> Path:
    """Refuse to start if another live process is already writing this directory.

    Written after two identical SLURM arrays, 57156367 and 57156375, ran concurrently over the same
    fifteen model directories because a submission that looked like it had not happened had. The
    per-writer temp name keeps a single *file* from being spliced, but nothing stopped two trainers
    from taking turns on one model, and the result is weights that belong to neither run.

    The lock carries host, pid and SLURM job id so a stale one names what left it. It is treated as
    stale after `stale_after` seconds without a refresh, because a preempted scavenger task cannot
    clean up after itself and a requeued run of the *same* job must be allowed to resume. A lock from
    the same job id is always reclaimable for that reason.
    """
    d.mkdir(parents=True, exist_ok=True)
    f = d / "_lock.json"
    me = {"host": socket.gethostname(), "pid": os.getpid(),
          "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
          "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
          "label": label, "claimed_at": utc_now()}
    if f.exists():
        try:
            other = json.loads(f.read_text())
        except Exception:      # noqa: BLE001 -- an unreadable lock is a stale lock
            other = {}
        age = time.time() - f.stat().st_mtime
        same_job = other.get("slurm_job") and other["slurm_job"] == me["slurm_job"]
        if age < stale_after and not same_job:
            raise SystemExit(
                f"{d.name} is already being written by {other.get('host')} pid "
                f"{other.get('pid')} (SLURM {other.get('slurm_job')}"
                f"{'_' + other['slurm_array_task'] if other.get('slurm_array_task') else ''}), "
                f"last refreshed {age:.0f}s ago. Two trainers on one model directory produce weights "
                f"that belong to neither run. Cancel one, or wait {stale_after - age:.0f}s if that "
                "job is gone.")
    f.write_text(json.dumps(me, indent=2) + "\n")
    return f


def refresh_claim(d: Path) -> None:
    """Touch the lock so a long run is not mistaken for an abandoned one."""
    f = d / "_lock.json"
    if f.exists():
        os.utime(f, None)


def release_dir(d: Path) -> None:
    f = d / "_lock.json"
    if f.exists():
        f.unlink(missing_ok=True)


def atomic_write_json(obj, path: Path) -> None:
    """A small JSON file written atomically, through a temp name no other writer will pick."""
    tmp = temp_sibling(path)
    try:
        tmp.write_text(json.dumps(obj, indent=2) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_parquet(df: pd.DataFrame, path: Path, schema=None) -> None:
    """A pandas frame, written atomically. Passing `schema` enforces the declared types at write
    time, which is what stops one wrongly typed shard from silently coercing a DuckDB union."""
    import pyarrow as pa
    atomic_write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False), path)


def read_manifest(manifest: Path | None = None) -> pd.DataFrame:
    """The manifest as a frame, with missing fields as NaN. Empty frame when there is no manifest."""
    manifest = manifest or P.MANIFEST
    if not manifest.exists():
        return pd.DataFrame()
    rows = [json.loads(ln) for ln in manifest.read_text().splitlines() if ln.strip()]
    return pd.DataFrame(rows)


def append_manifest(row: dict, manifest: Path | None = None) -> None:
    """Append one provenance record, idempotently: skip if a row with the same
    (artifact, git_sha) already exists, so a restart never duplicates.

    JSONL rather than CSV. Each stage records its own fields, so the rows are heterogeneous, and a
    single CSV header written by whichever stage ran first leaves every later stage's values
    misaligned. That produced a file with rows of 6, 7, 8 and 9 columns which pandas could not parse
    at all, and the integrity gate caught it only because it tried to read the manifest.
    """
    manifest = manifest or P.MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        key = (str(row.get("artifact")), str(row.get("git_sha")))
        for ln in manifest.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                prev = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if (str(prev.get("artifact")), str(prev.get("git_sha"))) == key:
                return
    with open(manifest, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def shard_key(date) -> str:
    """Calendar-quarter shard key, e.g. "2014Q3"."""
    ts = pd.Timestamp(date)
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


def batched(seq, n: int):
    """Yield successive n-sized batches. The union of batches equals the input, in order."""
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
