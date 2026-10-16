"""Stage 7: consolidate the component shards into the two canonical tables.

`components.parquet` is one row per speaker turn and is written by DuckDB straight from the shard
glob, because at 16 GB of text it does not fit in memory. `calls.parquet` is one row per call and is
small, so it is assembled in pandas from the component aggregates plus the metadata and diagnostics
already on disk.

Call-level counts come from the components actually pulled rather than from stage 5b's expectations,
so the two can be compared and a shortfall is visible instead of assumed away.

Run:
    python -m ecalls.ingest.consolidate
    python -m ecalls.ingest.consolidate --no-strict     # report integrity failures without raising
"""
import argparse
import sys
import time

import duckdb
import pandas as pd

from ecalls.config import constants as C
from ecalls.config import paths as P
from ecalls.ingest.manifest import append_manifest, atomic_write_parquet, git_sha, utc_now

STAGE = "consolidate"


def shard_paths() -> list:
    return sorted(P.COMPONENTS_DIR.glob("components_*.parquet"))


def component_aggregates(con, glob: str) -> pd.DataFrame:
    """Per-call counts and section breakdown, from what is on disk."""
    return con.execute(f"""
        select keydevid,
               any_value(transcriptid)          as transcriptid,
               any_value(companyid)             as companyid,
               any_value(call_date)             as call_date,
               any_value(shard)                 as shard,
               count(*)                         as n_components,
               sum(n_chars)                     as n_chars,
               sum(case when section='presentation' then n_chars else 0 end) as n_chars_presentation,
               sum(case when section='qa'           then n_chars else 0 end) as n_chars_qa,
               sum(case when section='operator'     then n_chars else 0 end) as n_chars_operator,
               sum(case when section='qa' then 1 else 0 end)                 as n_components_qa,
               count(distinct person_id)        as n_speakers,
               sum(case when n_chars=0 then 1 else 0 end) as n_empty_components
        from read_parquet('{glob}')
        group by keydevid
    """).df()


def index_label(calls: pd.DataFrame) -> pd.Series:
    """True when any company route for the call belongs to a firm in the index on the call date.

    Any route, not the primary one, because a call covering a parent and a subsidiary is reachable
    under either company and the membership may sit on either (see the grain note in D2).
    """
    routes = pd.read_parquet(P.CALL_COMPANIES, columns=["keydevid", "companyid"])
    cols = ["companyid", "mbr_start", "mbr_end", "win_start", "win_end"]
    uni = pd.read_parquet(P.SP500_UNIVERSE, columns=cols).dropna(subset=["companyid"])
    uni["companyid"] = uni["companyid"].astype("int64")

    # Use the INTERSECTED window, which is when this companyid actually corresponded to this permno,
    # and fall back to the membership spell only for the Compustat-only rows that have no window.
    # Using the raw spell put 122 calls inside the index that were not: companyid 386000 is Thryv
    # Holdings, which mapped to gvkey 004094 only until 1998-06-30, after which that gvkey belongs to
    # Moody's. Permno 48506's spell runs from 1979 to open, so Thryv's 2007 and even 2026 calls were
    # stamped as S&P 500 calls.
    lo = pd.to_datetime(uni["win_start"]).fillna(pd.to_datetime(uni["mbr_start"]))
    hi = pd.to_datetime(uni["win_end"]).fillna(pd.to_datetime(uni["mbr_end"]))
    uni = uni.assign(lo=lo.dt.date, hi=hi.dt.date)[["companyid", "lo", "hi"]]

    dates = calls[["keydevid", "call_date"]].copy()
    dates["call_date"] = pd.to_datetime(dates["call_date"]).dt.date
    m = routes.merge(dates, on="keydevid", how="inner").merge(uni, on="companyid", how="inner")
    hit = (m["lo"] <= m["call_date"]) & (m["hi"] >= m["call_date"])
    inside = set(m.loc[hit, "keydevid"])
    return calls["keydevid"].isin(inside)


def build_calls(agg: pd.DataFrame) -> pd.DataFrame:
    """Join the component aggregates to the transcript metadata and the version diagnostics."""
    idx = pd.read_parquet(P.TRANSCRIPT_INDEX, columns=[
        "transcriptid", "companyid", "company_name", "headline", "call_time_utc",
        "event_type_id", "event_type", "collection_type_id", "collection_type",
        "presentation_type_id", "presentation_type", "created_date", "audio_length_sec"])
    idx = idx.drop_duplicates(["transcriptid", "companyid"])

    vc = pd.read_parquet(P.VERSION_CHOICE, columns=[
        "keydevid", "chosen_transcriptid", "n_versions", "chosen_chars", "chosen_components",
        "runner_up_chars", "margin_ratio", "tiebreak_used", "suspect_short", "suspect_long",
        "rule_version"]).rename(columns={"chosen_chars": "expected_chars",
                                         "chosen_components": "expected_components"})

    out = (agg.merge(idx, on=["transcriptid", "companyid"], how="left")
              .merge(vc.drop(columns=["chosen_transcriptid"]), on="keydevid", how="left"))

    lnk = pd.read_parquet(P.SP500_UNIVERSE, columns=["companyid", "gvkey", "permno"]).dropna(
        subset=["companyid"])
    lnk["companyid"] = lnk["companyid"].astype("int64")
    lnk = lnk.drop_duplicates("companyid")
    out = out.merge(lnk, on="companyid", how="left")

    out["in_index_at_call"] = index_label(out)
    out["component_shortfall"] = out["expected_components"] - out["n_components"]
    out["cleaning_version"] = C.CLEANING_VERSION
    out["pull_ts"] = utc_now()
    out["git_sha"] = git_sha()
    return out.sort_values(["call_date", "keydevid"]).reset_index(drop=True)


def integrity_check(con, glob: str, calls: pd.DataFrame, strict: bool = True) -> list[str]:
    """Three-way equality between the shards, the manifest, and the consolidated output.

    A silently missing shard is the failure this exists to prevent, so the counts are asserted
    rather than reported.
    """
    problems = []
    on_disk = {p.stem.replace("components_", "") for p in shard_paths()}
    in_out = set(calls["shard"].dropna().unique())
    if on_disk != in_out:
        problems.append(f"shard files {len(on_disk)} do not match shards in output {len(in_out)}: "
                        f"only on disk {sorted(on_disk - in_out)[:5]}, "
                        f"only in output {sorted(in_out - on_disk)[:5]}")

    try:
        from ecalls.ingest.manifest import read_manifest
        man = read_manifest()
        man_shards = {a.replace("components_", "").replace(".parquet", "")
                      for a in man.loc[man["stage"] == "pull_components", "artifact"].astype(str)}
        if not on_disk.issubset(man_shards):
            problems.append(f"shards on disk with no manifest row: "
                            f"{sorted(on_disk - man_shards)[:5]}")
        # And the other direction, which is what a DELETED shard looks like. Testing only the subset
        # above could never catch it: every other check in this function derives from the same glob,
        # so removing a shard file moved both sides of each comparison in lockstep and the gate
        # printed "clean".
        vanished = man_shards - on_disk
        if vanished:
            problems.append(f"{len(vanished)} shard(s) have a manifest row but no file on disk: "
                            f"{sorted(vanished)[:5]}")
    except Exception as e:  # noqa: BLE001
        problems.append(f"manifest unreadable: {type(e).__name__}")

    n_comp = con.execute(f"select count(*) from read_parquet('{glob}')").fetchone()[0]
    if int(calls["n_components"].sum()) != n_comp:
        problems.append(f"component total {n_comp:,} != sum of per-call counts "
                        f"{int(calls['n_components'].sum()):,}")

    dup = int(calls["keydevid"].duplicated().sum())
    if dup:
        problems.append(f"{dup} duplicate keydevid rows in calls")

    short = int((calls["component_shortfall"].fillna(0) != 0).sum())
    if short:
        problems.append(f"{short} calls whose component count differs from stage 5b's expectation")

    if problems and strict:
        raise AssertionError("integrity check failed:\n  " + "\n  ".join(problems))
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-strict", action="store_true")
    args = ap.parse_args(argv)

    P.ensure_dirs()
    paths = shard_paths()
    if not paths:
        print("FAIL: no component shards; run ecalls.ingest.pull_components first", flush=True)
        return 1
    glob = str(P.COMPONENTS_DIR / "components_*.parquet")
    con = duckdb.connect()
    # Bound the sort of 22.8M rows carrying 16 GB of text. Left unbounded, DuckDB will take most of
    # the machine's RAM, and this runs on the laptop.
    con.execute("set memory_limit='8GB'")
    con.execute("set threads=4")
    con.execute(f"set temp_directory='{P.DATA_ROOT / '_duckdb_tmp'}'")
    t0 = time.time()
    print(f"shards: {len(paths)}", flush=True)

    agg = component_aggregates(con, glob)
    print(f"  aggregated {len(agg):,} calls ({time.time()-t0:.0f}s)", flush=True)
    calls = build_calls(agg)
    print(f"  calls table {len(calls):,} rows, {calls.in_index_at_call.sum():,} in index at call",
          flush=True)

    problems = integrity_check(con, glob, calls, strict=not args.no_strict)
    if problems:
        print("  INTEGRITY PROBLEMS:", flush=True)
        for p_ in problems:
            print(f"    {p_}", flush=True)
    else:
        print("  integrity: clean", flush=True)

    atomic_write_parquet(calls, P.CALLS)
    tmp = str(P.COMPONENTS) + ".tmp"
    con.execute(f"""copy (select * from read_parquet('{glob}') order by call_date, keydevid,
                          component_order) to '{tmp}' (format parquet, compression zstd)""")
    import os
    os.replace(tmp, P.COMPONENTS)

    mb = P.COMPONENTS.stat().st_size / 1e6
    print(f"\nwrote {P.CALLS} ({len(calls):,} rows)", flush=True)
    print(f"wrote {P.COMPONENTS} ({mb:,.0f} MB)  in {(time.time()-t0)/60:.1f} min", flush=True)
    append_manifest({"stage": STAGE, "artifact": P.COMPONENTS.name,
                     "n_rows": int(calls["n_components"].sum()), "n_calls": len(calls),
                     "total_chars": int(calls["n_chars"].sum()),
                     "integrity_problems": len(problems),
                     "pull_ts": utc_now(), "git_sha": git_sha()})
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
