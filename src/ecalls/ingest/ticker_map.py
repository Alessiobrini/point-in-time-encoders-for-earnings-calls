"""Map a handful of tickers to CRSP permnos, point in time.

Needed because `calls.parquet` carries `permno` and the realized-volatility panel the Phase 2
diagnostic uses is keyed by ticker. Matching on company name would work for most mega-caps and fail
silently for the rest, and a wrong row there would move the diagnostic's answer without announcing
itself. `crsp.stocknames` is authoritative and the query is small.

**A ticker is not a stable key.** It is reassigned: a ticker belonging to one company in 2015 can
belong to another later, and a company can change ticker without changing permno. So the mapping is
pulled with its `namedt` and `nameenddt` validity window and joined **as of the call date**, never as
a plain dictionary. This module returns the windows and leaves the join to the caller.

Run on the Mac, where the credential lives:
    python -m ecalls.ingest.ticker_map --tickers AAPL,MSFT,...
"""
import argparse
import json

import pandas as pd

from ecalls.config import paths as P
from ecalls.ingest import wrds_conn
from ecalls.ingest.manifest import atomic_write_parquet, git_sha, utc_now

TICKER_MAP = P.PROCESSED_DIR / "ticker_permno_map.parquet"


def fetch(tickers: list[str]) -> pd.DataFrame:
    """Every (permno, ticker, validity window) CRSP has for these tickers."""
    q = """
        SELECT permno, namedt, nameenddt, ticker, comnam, shrcd, exchcd
        FROM crsp.stocknames
        WHERE ticker = ANY(%(tickers)s)
        ORDER BY ticker, namedt
    """
    with wrds_conn.connect() as conn:
        df = pd.read_sql(q, conn, params={"tickers": [t.upper() for t in tickers]})
    df["namedt"] = pd.to_datetime(df["namedt"])
    df["nameenddt"] = pd.to_datetime(df["nameenddt"])
    return df


def build(tickers: list[str], quiet: bool = False) -> dict:
    df = fetch(tickers)
    if df.empty:
        raise SystemExit("CRSP returned no rows for those tickers")

    # Common stock on a major exchange, which is what a realized-volatility panel of US equities is.
    # Keeping the raw rows too, because a ticker that only ever appears as something else is a
    # finding rather than an empty filter.
    keep = df[(df["shrcd"].isin([10, 11])) & (df["exchcd"].isin([1, 2, 3]))].copy()
    atomic_write_parquet(keep.reset_index(drop=True), TICKER_MAP)

    per_ticker = keep.groupby("ticker")["permno"].nunique().to_dict()
    missing = sorted(set(t.upper() for t in tickers) - set(keep["ticker"]))
    ambiguous = {t: int(n) for t, n in per_ticker.items() if n > 1}
    meta = {
        "artifact": "ticker_permno_map",
        "n_tickers_requested": len(tickers),
        "n_tickers_matched": len(per_ticker),
        "n_rows": int(len(keep)),
        "missing": missing,
        # Not an error. A ticker with two permnos is a reassignment, which is exactly why the join is
        # as of the call date rather than by dictionary lookup.
        "tickers_with_more_than_one_permno": ambiguous,
        "rows_dropped_by_share_or_exchange_filter": int(len(df) - len(keep)),
        "git_sha": git_sha(),
        "built_at": utc_now(),
    }
    TICKER_MAP.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    if not quiet:
        print(f"{meta['n_tickers_matched']} of {len(tickers)} tickers matched, "
              f"{meta['n_rows']} validity rows")
        if missing:
            print(f"  no common-stock row: {missing}")
        if ambiguous:
            print(f"  more than one permno, so the as-of join matters: {ambiguous}")
    return meta


def as_of(calls: pd.DataFrame, mapping: pd.DataFrame, date_col: str = "call_date") -> pd.DataFrame:
    """Attach `ticker` to each call by the window containing its date.

    An interval join rather than a merge on permno alone, because a permno's ticker changes and the
    diagnostic's target comes from a ticker-keyed panel.
    """
    m = mapping[["permno", "ticker", "namedt", "nameenddt"]]
    out = calls.merge(m, on="permno", how="inner")
    d = pd.to_datetime(out[date_col])
    return out[(d >= out["namedt"]) & (d <= out["nameenddt"])].drop(
        columns=["namedt", "nameenddt"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers", required=True, help="comma separated")
    args = ap.parse_args(argv)
    build([t.strip() for t in args.tickers.split(",") if t.strip()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
