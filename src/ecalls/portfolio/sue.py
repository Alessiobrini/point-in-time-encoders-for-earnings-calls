"""D27's second target: next quarter's standardized unexpected earnings, matched to each call.

The surprise is the seasonal random walk scaled by price, after Livnat and Mendenhall (doi
10.1111/j.1475-679X.2006.00196.x): split-adjusted EPS minus that of the same fiscal quarter a year
earlier, over the split-adjusted quarter-end price. EPS is `epspxq`, basic EPS excluding extraordinary
items, and the split adjustment divides by `ajexq`.

A call is matched to the quarter whose report date `rdq` falls within 3 calendar days of the call date
for the call's gvkey, since the call accompanies that release. Its target is the surprise of the next
fiscal quarter, known on that quarter's `rdq`, and its control is the surprise of the matched quarter,
which the call discusses and which is known by day 0.

The pull runs on the Mac, where the WRDS credential lives:

    python -m ecalls.portfolio.sue pull
    python -m ecalls.portfolio.sue build
"""
import argparse
import datetime as dt

import numpy as np
import polars as pl

from ecalls.config import paths as P
from ecalls.ingest import manifest as M

FUNDQ_PATH = P.RAW_DIR / "compustat" / "fundq.parquet"
SUE_PATH = P.PROCESSED_DIR / "pf_sue.parquet"
MATCH_DAYS = 3
FUNDQ_SCHEMA = {"gvkey": pl.String, "datadate": pl.Date, "fyearq": pl.Int32, "fqtr": pl.Int32,
                "rdq": pl.Date, "epspxq": pl.Float64, "ajexq": pl.Float64, "prccq": pl.Float64}


def pull(gvkeys: list[str]) -> pl.DataFrame:
    from ecalls.ingest import wrds_conn as W
    conn = W.connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT gvkey, datadate, fyearq, fqtr, rdq, epspxq::float8, ajexq::float8, prccq::float8 "
        "FROM comp.fundq WHERE gvkey = ANY(%s) AND datadate >= '2008-01-01' "
        "AND indfmt = 'INDL' AND datafmt = 'STD' AND consol = 'C' AND popsrc = 'D'",
        (gvkeys,))
    return pl.DataFrame(cur.fetchall(), schema=FUNDQ_SCHEMA, orient="row")


def surprises(fq: pl.DataFrame) -> pl.DataFrame:
    """One row per gvkey and fiscal quarter with its surprise, from the same quarter a year earlier."""
    fq = (fq.filter(pl.col("fyearq").is_not_null() & pl.col("fqtr").is_not_null())
            .unique(["gvkey", "fyearq", "fqtr"], keep="last")
            .with_columns(eps_adj=pl.col("epspxq") / pl.col("ajexq"),
                          price_adj=pl.col("prccq") / pl.col("ajexq")))
    prev = fq.select("gvkey", (pl.col("fyearq") + 1).alias("fyearq"), "fqtr",
                     pl.col("eps_adj").alias("eps_adj_lag4"))
    out = fq.join(prev, on=["gvkey", "fyearq", "fqtr"], how="left")
    sue = (pl.col("eps_adj") - pl.col("eps_adj_lag4")) / pl.col("price_adj")
    out = out.with_columns(sue=pl.when(pl.col("price_adj") > 0).then(sue).otherwise(None))
    # The quarter after, by fiscal sequence: (fyearq, fqtr) + 1, wrapping from 4 to 1.
    nxt = out.select("gvkey",
                     pl.when(pl.col("fqtr") == 1).then(pl.col("fyearq") - 1).otherwise(pl.col("fyearq")).alias("fyearq"),
                     pl.when(pl.col("fqtr") == 1).then(4).otherwise(pl.col("fqtr") - 1).alias("fqtr"),
                     pl.col("sue").alias("sue_next"), pl.col("rdq").alias("rdq_next"))
    return out.join(nxt, on=["gvkey", "fyearq", "fqtr"], how="left")


def match_calls(calls: pl.DataFrame, sq: pl.DataFrame) -> pl.DataFrame:
    """Each call to the quarter it accompanies: same gvkey, `rdq` within `MATCH_DAYS` of the call."""
    cand = calls.join(sq.filter(pl.col("rdq").is_not_null()), on="gvkey", how="inner")
    cand = cand.with_columns(gap=(pl.col("rdq") - pl.col("call_date")).dt.total_days().abs())
    cand = cand.filter(pl.col("gap") <= MATCH_DAYS).sort(["keydevid", "gap", "datadate"])
    best = cand.unique("keydevid", keep="first")
    return best.select("keydevid", pl.col("sue").alias("sue_cur"), "sue_next",
                       pl.col("rdq_next").alias("sue_known_date"), "datadate", "fyearq", "fqtr")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("pull", "build"))
    args = ap.parse_args(argv)
    calls = pl.read_parquet(P.CALLS, columns=["keydevid", "gvkey", "call_date", "in_index_at_call"])
    if args.what == "pull":
        gv = sorted(calls.filter(pl.col("in_index_at_call"))["gvkey"].drop_nulls().unique().to_list())
        fq = pull(gv)
        M.atomic_write_table(fq.to_arrow(), FUNDQ_PATH)
        print(f"fundq: {fq.height:,} rows for {fq['gvkey'].n_unique():,} of {len(gv):,} gvkeys")
        return 0
    from ecalls.portfolio import panel as PN
    ev = pl.read_parquet(PN.PANEL_PATH, columns=["keydevid"])
    c = calls.with_columns(pl.col("call_date").cast(pl.Date)).join(ev, on="keydevid", how="inner")
    m = match_calls(c.drop_nulls("gvkey"), surprises(pl.read_parquet(FUNDQ_PATH)))
    usable = m.filter(pl.col("sue_next").is_not_null() & pl.col("sue_cur").is_not_null()
                      & pl.col("sue_known_date").is_not_null())
    M.atomic_write_table(usable.to_arrow(), SUE_PATH)
    stats = {"events": ev.height, "matched_to_a_quarter": m.height, "usable": usable.height}
    M.atomic_write_json({**stats, "built_at": M.utc_now(), "git_sha": M.git_sha()},
                        SUE_PATH.with_suffix(".json"))
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
