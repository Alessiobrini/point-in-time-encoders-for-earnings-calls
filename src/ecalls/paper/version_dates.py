"""How often the transcript version we keep for a call is dated after the cutoff of a vintage that reads it.

    ECALLS_DATA_ROOT=<data root> python -m ecalls.paper.version_dates

Capital IQ revises a call's transcript for weeks after the call, so the version we keep can carry a
creation date after the cutoff of a vintage whose window holds the call. For each cutoff Y from 2010 to
2024, among the trainable calls (not held out, `ecalls.train.holdout`) dated on or before 31 December of
Y, this counts the calls whose kept version was created after that day. A version created on the cutoff
day itself is not late.

It reads metadata only: the kept version of each call from `identifiers/calls.csv.gz` and the version's
creation date from the transcript index (`P.TRANSCRIPT_INDEX`), never a transcript's text. A kept
version missing from the index has no date, so it is counted as undated and left out of the share.
Writes `reports/paper/version_dates.json`, which `ecalls.paper.values` reads, so the paper's macros
regenerate without the licensed data.
"""
import datetime as dt
import json

import polars as pl

from ecalls.config import paths as P
from ecalls.train.holdout import is_holdout

IDENTIFIERS = P.PROJECT_ROOT / "identifiers" / "calls.csv.gz"
OUT = P.REPORTS_DIR / "paper" / "version_dates.json"
YEARS = range(2010, 2025)


def trainable(keydevids: pl.Series) -> pl.Series:
    """True for a call that a vintage may train on, by the fixed hash rule of the held-out split."""
    return pl.Series("trainable", [not is_holdout(int(k)) for k in keydevids], dtype=pl.Boolean)


def late_shares(calls: pl.DataFrame, years=YEARS) -> dict[str, dict]:
    """Per cutoff year, the trainable calls on or before the cutoff and the share of them whose kept
    version was created after it.

    `calls` holds one row per call with `call_date` and `created_date` (dates, `created_date` may be
    null) and a boolean `trainable`. The share is over the calls with a creation date.
    """
    out = {}
    for y in years:
        cutoff = dt.date(y, 12, 31)
        window = calls.filter(pl.col("trainable") & (pl.col("call_date") <= cutoff))
        dated = window.filter(pl.col("created_date").is_not_null())
        late = int(dated.filter(pl.col("created_date") > cutoff).height)
        out[str(y)] = {"trainable_calls": window.height, "dated": dated.height,
                       "undated": window.height - dated.height, "late": late,
                       "share_late": late / dated.height if dated.height else None}
    return out


def load() -> pl.DataFrame:
    """One row per call: its date, its trainable flag and the creation date of its kept version."""
    ids = pl.read_csv(IDENTIFIERS, columns=["keydevid", "transcriptid", "call_date"],
                      schema_overrides={"keydevid": pl.Int64, "transcriptid": pl.Int64, "call_date": pl.Date})
    # the index repeats some versions on several rows with one creation date, so it is deduplicated
    index = (pl.scan_parquet(P.TRANSCRIPT_INDEX).select("transcriptid", "created_date").unique().collect())
    if index["transcriptid"].n_unique() != index.height:
        raise ValueError("a transcript version carries two creation dates in the index")
    calls = ids.join(index, on="transcriptid", how="left")
    if calls.height != ids.height:
        raise ValueError("the join changed the number of calls")
    return calls.with_columns(trainable(calls["keydevid"]))


def main() -> int:
    shares = late_shares(load())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cutoffs": shares}, indent=2) + "\n")
    for y, e in shares.items():
        print(f"{y}: {e['late']:>5} of {e['dated']:>7} trainable calls, {100 * e['share_late']:.2f}%"
              f" ({e['undated']} undated)")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
