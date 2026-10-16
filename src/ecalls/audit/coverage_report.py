"""Stage 8, second axis: how many calls we hold per index member per quarter.

Descriptive rather than a gate, because a firm does not always hold a call in a quarter it was an
index member, and Capital IQ's coverage ramps over the early years. The point is to make the shape of
the coverage visible instead of assumed.

Run:
    python -m ecalls.audit.coverage_report
"""
import sys

import pandas as pd

from ecalls.config import paths as P
from ecalls.ingest.manifest import git_sha, utc_now


def main() -> int:
    if not P.CALLS.exists():
        print("FAIL: run ecalls.ingest.consolidate first", flush=True)
        return 1
    calls = pd.read_parquet(P.CALLS, columns=["keydevid", "call_date", "companyid",
                                             "in_index_at_call", "n_chars", "n_components"])
    calls["quarter"] = pd.to_datetime(calls["call_date"]).dt.to_period("Q").astype(str)
    calls["year"] = pd.to_datetime(calls["call_date"]).dt.year

    rows = []
    for y, g in calls.groupby("year"):
        idx = g.loc[g["in_index_at_call"]]
        rows.append({
            "year": int(y),
            "calls_all": len(g),
            "calls_in_index": len(idx),
            "companies_all": g["companyid"].nunique(),
            "companies_in_index": idx["companyid"].nunique(),
            "calls_per_index_company": round(len(idx) / max(idx["companyid"].nunique(), 1), 2),
            "chars_all_G": round(g["n_chars"].sum() / 1e9, 3),
            "mean_components": round(g["n_components"].mean(), 1),
        })
    rep = pd.DataFrame(rows)
    rep["pull_ts"] = utc_now(); rep["git_sha"] = git_sha()
    rep.to_csv(P.COVERAGE_REPORT, index=False)
    print(rep.drop(columns=["pull_ts", "git_sha"]).to_string(index=False))
    print(f"\nwrote {P.COVERAGE_REPORT}")

    print("\n" + member_quarter_coverage())
    return 0


def member_quarter_coverage(y0: int = 2008, y1: int = 2025) -> str:
    """The axis the roadmap actually specified: of the quarters a company was an index member, in
    how many did we get a call?

    Calls per company per year cannot answer this. A company averaging 3.9 calls a year looks like
    97.5% of four quarters only if every member reports every quarter, and membership starts and
    ends mid-quarter. This counts the member-quarters directly.
    """
    memb = pd.read_parquet(P.SP500_MEMBERSHIP, columns=["permno", "mbr_start", "mbr_end"])
    cal = pd.read_parquet(P.CALLS, columns=["permno", "call_date", "in_index_at_call"])
    cal = cal[cal["in_index_at_call"] & cal["permno"].notna()].copy()
    cal["q"] = pd.to_datetime(cal["call_date"]).dt.to_period("Q")
    have = set(zip(cal["permno"].astype("int64"), cal["q"].astype(str)))

    expected = set()
    for pn, a, b in memb.itertuples(index=False):
        for q in pd.period_range(pd.Timestamp(a), pd.Timestamp(b), freq="Q"):
            if y0 <= q.year <= y1:
                expected.add((int(pn), str(q)))
    hit = expected & have
    pct = 100 * len(hit) / max(len(expected), 1)
    return (f"=== member-quarter coverage, {y0} to {y1} ===\n"
            f"  member-quarters expected : {len(expected):>8,}\n"
            f"  with at least one call   : {len(hit):>8,}\n"
            f"  coverage                 : {pct:>7.1f}%\n"
            f"  empty member-quarters    : {len(expected) - len(hit):>8,}\n"
            f"  A company does not always hold a call in a quarter it was a member, so this is\n"
            f"  descriptive rather than a gate. It is NOT 'near-complete quarterly coverage'.")


if __name__ == "__main__":
    sys.exit(main())
