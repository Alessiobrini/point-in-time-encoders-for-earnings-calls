"""Every artifact path in the project, resolved from a single root.

The root is `$ECALLS_DATA_ROOT`, defaulting to `<repo>/data`. Nothing else in the package knows an
absolute path, so relocating the corpus to an external disk, a cluster or a server is setting one
environment variable and copying a directory.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # <root>/src/ecalls/config/paths.py

DATA_ROOT = Path(os.environ.get("ECALLS_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()

RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
INDEX_DIR = DATA_ROOT / "index"

# Measurement output, which is **not** under DATA_ROOT. Reports are written for people, they are
# small, and they are the one thing here worth committing, while everything under DATA_ROOT is
# licensed or regenerable and git-ignored. A measurement that landed in `data/processed/` was neither
# committed nor picked up by the cluster sync, so it existed only on whichever machine produced it.
REPORTS_DIR = PROJECT_ROOT / "reports"
# Copies of the models' `_run.json` training records, shipped with the code so the paper's tables
# regenerate on a machine that holds no model.
RUN_RECORDS_DIR = PROJECT_ROOT / "runs"


def run_records_dir() -> Path:
    """Where the training records are read: the models themselves when present, else the copies."""
    models = PROCESSED_DIR / "models"
    return models if any(models.glob("*/_run.json")) else RUN_RECORDS_DIR

# --- stage outputs, in pipeline order -------------------------------------------------------
ACCESS_PROBE = RAW_DIR / "wrds_access_probe.json"
MEASUREMENT_CHECK = RAW_DIR / "measurement_check.json"

UNIVERSE_DIR = RAW_DIR / "universe"
SP500_MEMBERSHIP = UNIVERSE_DIR / "sp500_membership.parquet"
LINK_CHAIN_RAW = UNIVERSE_DIR / "link_chain_raw.parquet"
SP500_UNIVERSE = PROCESSED_DIR / "sp500_universe.parquet"

CIQ_DIR = RAW_DIR / "ciq"
TRANSCRIPT_INDEX = CIQ_DIR / "transcript_index.parquet"
LENGTHS_DIR = CIQ_DIR / "lengths"
COMPONENTS_DIR = CIQ_DIR / "components"
FAILED_BATCHES = COMPONENTS_DIR / "_failed_batches.csv"

CALL_COMPANIES = PROCESSED_DIR / "call_companies.parquet"
VERSION_CHOICE = PROCESSED_DIR / "version_choice.parquet"
CALLS = PROCESSED_DIR / "calls.parquet"
COMPONENTS = PROCESSED_DIR / "components.parquet"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
CHUNKS = PROCESSED_DIR / "chunks.parquet"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"

COMPLETENESS_AUDIT = PROCESSED_DIR / "completeness_audit.csv"
COVERAGE_REPORT = PROCESSED_DIR / "coverage_report.csv"
REVISIONS_LOG = PROCESSED_DIR / "revisions.csv"

# JSONL, not CSV: each stage records a different set of fields, and a single CSV header
# cannot describe rows of 6, 7, 8 and 9 columns. One self-describing object per line.
MANIFEST = RAW_DIR / "_manifest.jsonl"
MANIFEST_LEGACY_CSV = RAW_DIR / "_manifest.csv"


def lengths_shard(year: int) -> Path:
    return LENGTHS_DIR / f"lengths_{year}.parquet"


def chunks_shard(shard: str) -> Path:
    """`shard` is a quarter key such as "2014Q3", matching the component shards."""
    return CHUNKS_DIR / f"chunks_{shard}.parquet"


def components_shard(shard: str) -> Path:
    """`shard` is a quarter key such as "2014Q3"."""
    return COMPONENTS_DIR / f"components_{shard}.parquet"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, INDEX_DIR, UNIVERSE_DIR, CIQ_DIR,
              LENGTHS_DIR, COMPONENTS_DIR, EMBEDDINGS_DIR, CHUNKS_DIR):
        d.mkdir(parents=True, exist_ok=True)
