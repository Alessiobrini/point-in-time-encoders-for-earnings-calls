"""Daily stock returns and factor returns for the portfolio test (D26).

Pulled on the Mac, where the WRDS credential lives. Two artifacts:

- `crsp/dsf_v2_<year>.parquet`: daily total return, market cap, SIC code and delisting flag for every
  permno that was ever an S&P 500 member. All members rather than only the ones with calls, because
  the equally weighted S&P 500 benchmark needs the whole index on every day. `crsp.dsf_v2` rather than
  the legacy `crsp.dsf`, which stops at 2024-12-31. Its `dlyret` includes the delisting return on the
  delisting day, checked against the legacy `dlret` on three 2023 delistings, so no separate delisting
  file is needed.
- `ff/ff_daily.parquet`: the Fama-French five factors with the risk-free rate, plus momentum.

One query per year, because `wrds_conn.connect` sets a 300 s statement timeout. A year is skipped
when its file exists and records the same permno set, so a rerun after a universe change refetches.

    python -m ecalls.ingest.crsp_returns
"""
import hashlib
import json

import polars as pl

from ecalls.config import paths as P
from ecalls.ingest import manifest as M
from ecalls.ingest import wrds_conn as W

CRSP_DIR = P.RAW_DIR / "crsp"
FF_PATH = P.RAW_DIR / "ff" / "ff_daily.parquet"
FIRST_YEAR = 2009            # 252 trading days of momentum before the first 2011 call need 2010
LAST_YEAR = 2025

DSF_SCHEMA = {"permno": pl.Int32, "dlycaldt": pl.Date, "dlyret": pl.Float64, "dlycap": pl.Float64,
              "siccd": pl.Int32, "dlydelflg": pl.String}
FF_SCHEMA = {"date": pl.Date, **{c: pl.Float64 for c in ("mktrf", "smb", "hml", "rmw", "cma", "rf", "umd")}}


def member_permnos() -> list[int]:
    """Every S&P 500 member, plus every permno a call links to.

    The two differ: the link chain can route a company to a share class other than the one in the
    index, and 498 in-index calls carried a permno outside the membership list. Leaving those out
    would drop their calls from the backtest without an error.
    """
    m = pl.read_parquet(P.UNIVERSE_DIR / "sp500_membership.parquet", columns=["permno"])
    c = pl.read_parquet(P.CALLS, columns=["permno"]).drop_nulls()
    return sorted({int(x) for x in m["permno"].unique().to_list()}
                  | {int(x) for x in c["permno"].unique().to_list()})


def permno_digest(permnos: list[int]) -> str:
    return hashlib.sha1(",".join(map(str, sorted(permnos))).encode()).hexdigest()[:12]


def year_path(year: int):
    return CRSP_DIR / f"dsf_v2_{year}.parquet"


def pull_year(conn, year: int, permnos: list[int]) -> pl.DataFrame:
    cur = conn.cursor()
    cur.execute(
        # numeric columns come back as Decimal, which polars will not mix with None, so cast in SQL
        "SELECT permno, dlycaldt, dlyret::float8, dlycap::float8, siccd, dlydelflg FROM crsp.dsf_v2 "
        "WHERE dlycaldt BETWEEN %s AND %s AND permno = ANY(%s)",
        (f"{year}-01-01", f"{year}-12-31", permnos))
    # An explicit schema, because polars infers a column's type from its first rows and a leading
    # null then rejects the first real value.
    return pl.DataFrame(cur.fetchall(), schema=DSF_SCHEMA, orient="row")


# D28: the full linked universe, with the open price, the close and the flags the common-stock filter
# needs. A separate directory, so the S&P files D26 and D27 read are never rewritten.
CRSP_ALL_DIR = P.RAW_DIR / "crsp_all"
DSF_ALL_SCHEMA = {**DSF_SCHEMA, "dlyopen": pl.Float64, "dlyprc": pl.Float64, "sharetype": pl.String,
                  "securitytype": pl.String, "securitysubtype": pl.String, "primaryexch": pl.String}
BATCH = 1500


def pull_year_all(conn, year: int, permnos: list[int]) -> pl.DataFrame:
    """One year for the full universe, in permno batches so no query nears the statement timeout."""
    frames = []
    for i in range(0, len(permnos), BATCH):
        cur = conn.cursor()
        cur.execute(
            "SELECT permno, dlycaldt, dlyret::float8, dlycap::float8, siccd, dlydelflg, "
            "dlyopen::float8, dlyprc::float8, sharetype, securitytype, securitysubtype, primaryexch "
            "FROM crsp.dsf_v2 WHERE dlycaldt BETWEEN %s AND %s AND permno = ANY(%s)",
            (f"{year}-01-01", f"{year}-12-31", permnos[i:i + BATCH]))
        frames.append(pl.DataFrame(cur.fetchall(), schema=DSF_ALL_SCHEMA, orient="row"))
    return pl.concat(frames)


def main_all() -> int:
    permnos = sorted(int(x) for x in
                     pl.read_parquet(P.PROCESSED_DIR / "pf_universe_all.parquet")["permno"].unique().to_list())
    digest = permno_digest(permnos)
    CRSP_ALL_DIR.mkdir(parents=True, exist_ok=True)
    conn = W.connect()
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        path = CRSP_ALL_DIR / f"dsf_v2_{year}.parquet"
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists() and json.loads(meta.read_text()).get("permnos") == digest:
            print(f"{year}: present for this permno set")
            continue
        df = pull_year_all(conn, year, permnos)
        M.atomic_write_table(df.to_arrow(), path)
        M.atomic_write_json({"year": year, "rows": df.height, "permnos": digest,
                             "n_permnos_requested": len(permnos),
                             "n_permnos_returned": df["permno"].n_unique(),
                             "pulled_at": M.utc_now(), "git_sha": M.git_sha()}, meta)
        print(f"{year}: {df.height:,} rows, {df['permno'].n_unique():,} permnos", flush=True)
    return 0


def pull_ff(conn) -> pl.DataFrame:
    cur = conn.cursor()
    cur.execute(
        "SELECT f.date, f.mktrf::float8, f.smb::float8, f.hml::float8, f.rmw::float8, "
        "f.cma::float8, f.rf::float8, m.umd::float8 "
        "FROM ff.fivefactors_daily f JOIN ff.factors_daily m ON f.date = m.date "
        "WHERE f.date BETWEEN %s AND %s ORDER BY f.date",
        (f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-31"))
    return pl.DataFrame(cur.fetchall(), schema=FF_SCHEMA, orient="row")


def main() -> int:
    permnos = member_permnos()
    digest = permno_digest(permnos)
    CRSP_DIR.mkdir(parents=True, exist_ok=True)
    conn = W.connect()
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        path = year_path(year)
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists() and json.loads(meta.read_text()).get("permnos") == digest:
            print(f"{year}: present for this permno set")
            continue
        df = pull_year(conn, year, permnos)
        M.atomic_write_table(df.to_arrow(), path)
        M.atomic_write_json({"year": year, "rows": df.height, "permnos": digest,
                             "n_permnos_requested": len(permnos),
                             "n_permnos_returned": df["permno"].n_unique(),
                             "pulled_at": M.utc_now(), "git_sha": M.git_sha()}, meta)
        print(f"{year}: {df.height:,} rows, {df['permno'].n_unique():,} permnos", flush=True)
    ff = pull_ff(conn)
    M.atomic_write_table(ff.to_arrow(), FF_PATH)
    print(f"ff: {ff.height:,} days, {ff['date'].min()} to {ff['date'].max()}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main_all() if "--universe" in sys.argv and "all" in sys.argv else main())
