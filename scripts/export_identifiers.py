"""Write the call identifiers that let anyone with WRDS access rebuild the corpus.

    python scripts/export_identifiers.py

One row per call: the Capital IQ event (`keydevid`), the transcript version kept for it
(`transcriptid`), the company (`companyid`), the call date, the Compustat and CRSP identifiers where the
call links to them, and whether the call is held out from training. No transcript text, name or headline
is written. The held-out flag is the salted hash of `ecalls.train.holdout`, included for convenience.
"""
import gzip
import pathlib

import polars as pl

from ecalls.config import paths as P
from ecalls.train import holdout as HO

OUT = pathlib.Path(__file__).resolve().parents[1] / "identifiers" / "calls.csv.gz"
COLUMNS = ["keydevid", "transcriptid", "companyid", "call_date", "gvkey", "permno", "in_index_at_call"]


def main() -> int:
    calls = (pl.read_parquet(P.PROCESSED_DIR / "calls.parquet", columns=COLUMNS)
             .with_columns(pl.col("call_date").dt.date(),
                           pl.col("keydevid").map_elements(HO.is_holdout, return_dtype=pl.Boolean).alias("held_out"))
             .sort(["call_date", "keydevid"]))
    assert calls["keydevid"].n_unique() == calls.height, "one row per call"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wb") as f:
        calls.write_csv(f)
    print(f"wrote {OUT}: {calls.height:,} calls, {calls['held_out'].sum():,} held out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
