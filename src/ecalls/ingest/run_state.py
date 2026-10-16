"""Run state for incremental refreshes (D9).

A refresh needs three things to be correct, and a bare "last run" timestamp supplies only the first:

  1. a **watermark**, the newest `transcriptcreationdate_utc` already ingested,
  2. a **margin**, because that field is Capital IQ's clock and the WRDS snapshot trails it, so a
     row created before a run can land after it,
  3. **counts per call year**, which catch anything the margin still missed, at any age.

Without (3) a wrong margin fails silently, which is the failure mode this project keeps meeting.
"""
import datetime as dt
import json
from pathlib import Path

from ecalls.config import constants as C


def _default_path() -> Path:
    from ecalls.config import paths as P
    return P.CIQ_DIR / "_run_state.json"


def load(stage: str, path: Path | None = None) -> dict:
    path = path or _default_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get(stage, {})


def save(stage: str, state: dict, path: Path | None = None) -> None:
    path = path or _default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    allst = json.loads(path.read_text()) if path.exists() else {}
    allst[stage] = state
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(allst, indent=2, default=str))
    tmp.replace(path)


def watermark_cutoff(state: dict, margin_days: int | None = None) -> dt.date | None:
    """The date to filter on. None means no prior run, so pull everything."""
    wm = state.get("max_creation_date")
    if not wm:
        return None
    margin = C.WATERMARK_MARGIN_DAYS if margin_days is None else margin_days
    return dt.date.fromisoformat(str(wm)) - dt.timedelta(days=margin)


def years_needing_reindex(state: dict, live_counts: dict[int, int]) -> list[int]:
    """Years whose live transcript count differs from what we recorded.

    This is the backstop for a wrong margin. A year appears here when rows arrived that the
    watermark did not see, and it is re-indexed in full regardless of creation date.
    """
    prev = {int(k): int(v) for k, v in (state.get("counts_by_year") or {}).items()}
    if not prev:
        return []
    return sorted(y for y, n in live_counts.items() if prev.get(y) != n)
