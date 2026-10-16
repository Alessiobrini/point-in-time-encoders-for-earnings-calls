"""Stage 5a: total text size per transcript, aggregated server side.

This is the stage that lets the corpus pull move each call's text across the wire exactly once.
Version selection (5b) needs to know which version of a call carries the most text, and answering
that by downloading all 3.18 versions and measuring locally would move over three times the bytes
for no gain (D5). So the sum runs on WRDS and only a row per transcript comes back.

Sharded by year of the call date, resumable per year. A transcript belonging to a call with only
one version still gets a row, because downstream code should never special-case it.

Run:
    python -m ecalls.ingest.transcript_lengths
    python -m ecalls.ingest.transcript_lengths --years 2024 2025
"""
import argparse
import sys
import time

import pandas as pd

from ecalls.config import constants as C
from ecalls.config import paths as P
from ecalls.ingest.manifest import append_manifest, atomic_write_parquet, batched, git_sha, utc_now

STAGE = "transcript_lengths"

SQL = f"""select transcriptid, count(*) as n_components,
                 coalesce(sum(length(componenttext)), 0) as total_chars
          from {C.TBL_COMPONENT} where transcriptid = any(%s) group by transcriptid"""


def pull_year(cur, tids: list[int]) -> pd.DataFrame:
    out = []
    for b in batched(tids, C.LENGTH_BATCH):
        cur.execute(SQL, (b,))
        out.extend(cur.fetchall())
    df = pd.DataFrame(out, columns=["transcriptid", "n_components", "total_chars"])
    # A transcript with no component rows is real and must not vanish: it becomes a zero, which
    # the version rule then ranks last and the audit flags, rather than a silently missing row.
    missing = set(tids) - set(df["transcriptid"].astype("int64")) if len(df) else set(tids)
    if missing:
        df = pd.concat([df, pd.DataFrame({"transcriptid": sorted(missing),
                                          "n_components": 0, "total_chars": 0})],
                       ignore_index=True)
    for c in ("transcriptid", "total_chars"):
        df[c] = df[c].astype("int64")
    df["n_components"] = df["n_components"].astype("int32")
    return df.sort_values("transcriptid").reset_index(drop=True)


def shard_covers(path, expected_tids: set[int]) -> bool:
    """A length shard counts as done only when it holds every transcript the year now has.

    File existence is the wrong marker because stage 4 is incremental: a refresh adds transcripts to
    a year whose shard already exists, the shard is skipped, and those transcripts never get a length
    row. `select_versions.load_candidates` then drops them with an inner join, so they never reach
    `version_choice.parquet`, `pull_components` never pulls them, and `completeness` compares against
    `version_choice.parquet` and reports UNEXPLAINED = 0. The loss is silent end to end. Same class as
    the component shard marker that ignored which universe a file was built for.
    """
    if not path.exists():
        return False
    try:
        import pyarrow.parquet as pq
        have = set(pq.read_table(path, columns=["transcriptid"])["transcriptid"].to_pylist())
    except Exception:  # noqa: BLE001 -- an unreadable shard is an incomplete shard
        return False
    return expected_tids <= have


def main(argv=None) -> int:
    from ecalls.ingest.wrds_conn import connect

    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int, default=None)
    args = ap.parse_args(argv)

    P.ensure_dirs()
    idx = pd.read_parquet(P.TRANSCRIPT_INDEX, columns=["transcriptid", "call_date"])
    idx["year"] = pd.to_datetime(idx["call_date"]).dt.year
    years = args.years or sorted(idx["year"].dropna().unique().tolist())

    conn = connect(statement_timeout_ms=900_000)
    cur = conn.cursor()
    t_all = time.time()
    done_rows = 0

    for y in years:
        path = P.lengths_shard(int(y))
        tids = sorted(idx.loc[idx["year"] == y, "transcriptid"].astype("int64").tolist())
        if not tids:
            print(f"[{y}] no transcripts", flush=True)
            continue
        if shard_covers(path, set(tids)):
            print(f"[{y}] skip(covered)", flush=True)
            continue
        if path.exists():
            print(f"[{y}] rewriting: the year now has {len(tids):,} transcripts, more than the "
                  f"shard covers", flush=True)
        t0 = time.time()
        try:
            df = pull_year(cur, tids)
        except Exception as e:  # noqa: BLE001 -- unattended run: never let one year kill it
            print(f"[{y}] ERROR: {str(e).splitlines()[0][:120]} -- reconnecting, retry next run",
                  flush=True)
            try:
                cur.close(); conn.close()
            except Exception:  # noqa: BLE001
                pass
            conn = connect(statement_timeout_ms=900_000); cur = conn.cursor()
            continue
        df["git_sha"] = git_sha()
        atomic_write_parquet(df, path)
        done_rows += len(df)
        dt_s = time.time() - t0
        print(f"[{y}] {len(df):>7,} transcripts  {df.total_chars.sum()/1e9:6.2f} GB text  "
              f"({dt_s:5.1f}s, {len(df)/max(dt_s,1e-9):,.0f}/s)", flush=True)
        append_manifest({"stage": STAGE, "artifact": path.name, "n_rows": len(df),
                         "total_chars": int(df.total_chars.sum()),
                         "pull_ts": utc_now(), "git_sha": git_sha()})

    print(f"\ndone: {done_rows:,} new rows in {(time.time()-t_all)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
