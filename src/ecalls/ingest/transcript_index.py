"""Stage 4: the transcript metadata index for every earnings call.

This is the small table (one row per transcript version, not per component) that everything
downstream is driven from. It is pulled for ALL companies, not just the S&P 500 universe, because
the universe is a label rather than a pull filter (D2) and the metadata is only tens of MB.

Incremental by default (D9): with prior run state, only transcripts created since the watermark
minus the margin are fetched, plus any year whose live count disagrees with the recorded one.

Run:
    python -m ecalls.ingest.transcript_index            # incremental, or full on a first run
    python -m ecalls.ingest.transcript_index --full     # ignore run state and rebuild
"""
import argparse
import sys

import pandas as pd

from ecalls.config import constants as C
from ecalls.config import paths as P
from ecalls.ingest import run_state
from ecalls.ingest.manifest import append_manifest, atomic_write_parquet, git_sha, utc_now

STAGE = "transcript_index"

COLS = ["companyid", "keydevid", "transcriptid", "headline", "mostimportantdateutc",
        "mostimportanttimeutc", "keydeveventtypeid", "keydeveventtypename", "companyname",
        "transcriptcollectiontypeid", "transcriptcollectiontypename",
        "transcriptpresentationtypeid", "transcriptpresentationtypename",
        "transcriptcreationdate_utc", "transcriptcreationtime_utc", "audiolengthsec"]

RENAME = {"mostimportantdateutc": "call_date", "mostimportanttimeutc": "call_time_utc",
          "keydeveventtypeid": "event_type_id", "keydeveventtypename": "event_type",
          "companyname": "company_name",
          "transcriptcollectiontypeid": "collection_type_id",
          "transcriptcollectiontypename": "collection_type",
          "transcriptpresentationtypeid": "presentation_type_id",
          "transcriptpresentationtypename": "presentation_type",
          "transcriptcreationdate_utc": "created_date",
          "transcriptcreationtime_utc": "created_time",
          "audiolengthsec": "audio_length_sec"}


def live_counts_by_year(cur) -> dict[int, int]:
    cur.execute(
        f"select extract(year from mostimportantdateutc)::int, count(*) from {C.TBL_DETAIL} "
        "where keydeveventtypeid = any(%s) group by 1", (list(C.EARNINGS_EVENT_TYPE_IDS),))
    return {int(y): int(n) for y, n in cur.fetchall() if y is not None}


def fetch(cur, since=None, years=None) -> pd.DataFrame:
    sql = f"select {', '.join(COLS)} from {C.TBL_DETAIL} where keydeveventtypeid = any(%s)"
    params = [list(C.EARNINGS_EVENT_TYPE_IDS)]
    if since is not None or years:
        clauses, extra = [], []
        if since is not None:
            clauses.append("transcriptcreationdate_utc >= %s"); extra.append(since)
        if years:
            clauses.append("extract(year from mostimportantdateutc)::int = any(%s)")
            extra.append(list(years))
        sql += " and (" + " or ".join(clauses) + ")"
        params += extra
    cur.execute(sql, tuple(params))
    df = pd.DataFrame(cur.fetchall(), columns=COLS).rename(columns=RENAME)
    for c in ("companyid", "keydevid", "transcriptid"):
        df[c] = df[c].astype("int64")
    df["call_date"] = pd.to_datetime(df["call_date"]).dt.date
    df["created_date"] = pd.to_datetime(df["created_date"]).dt.date
    return df


GRAIN = ["transcriptid", "companyid"]


def merge_index(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Upsert on (transcriptid, companyid), keeping the newly fetched row when both exist.

    The grain is NOT transcriptid alone. One transcript is attributed to several companyids when a
    call covers a parent and a subsidiary (Brasil Telecom / Oi, CVB Financial / Citizens Business
    Bank), which accounts for 5,874 of the 1,311,639 rows. Everything except companyid and
    company_name is constant within a transcriptid. Deduplicating on transcriptid would drop one
    company's route to the call, and the index label depends on those routes.
    """
    if old is None or old.empty:
        return new.sort_values(GRAIN).reset_index(drop=True)
    keys = set(map(tuple, new[GRAIN].to_numpy().tolist()))
    mask = [tuple(r) not in keys for r in old[GRAIN].to_numpy().tolist()]
    return (pd.concat([old.loc[mask], new], ignore_index=True)
            .sort_values(GRAIN).reset_index(drop=True))


def main(argv=None) -> int:
    from ecalls.ingest.wrds_conn import connect

    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="ignore run state and rebuild")
    args = ap.parse_args(argv)

    P.ensure_dirs()
    state = {} if args.full else run_state.load(STAGE)
    old = pd.read_parquet(P.TRANSCRIPT_INDEX) if (P.TRANSCRIPT_INDEX.exists() and state) else None

    conn = connect(statement_timeout_ms=900_000); cur = conn.cursor()

    counts = live_counts_by_year(cur)
    stale_years = run_state.years_needing_reindex(state, counts)
    since = run_state.watermark_cutoff(state)

    if since is None and not stale_years:
        print("full pull (no prior run state)", flush=True)
    else:
        print(f"incremental: created >= {since}"
              + (f", plus full re-index of years {stale_years}" if stale_years else ""), flush=True)

    new = fetch(cur, since=since, years=stale_years or None)
    print(f"  fetched {len(new):,} transcript rows", flush=True)

    df = merge_index(old, new)
    added = len(df) - (0 if old is None else len(old))
    print(f"  index now {len(df):,} rows ({added:+,}), "
          f"{df.keydevid.nunique():,} distinct calls, {df.companyid.nunique():,} companies",
          flush=True)

    # Rewrite only when the content moved. Stage 4 has no output guard in run_all.sh because it is
    # the incremental stage and must always query, but rewriting an unchanged 28 MB artifact makes
    # the pipeline fail its own idempotency check ("a second run changes no file") for no reason.
    unchanged = old is not None and len(df) == len(old) and df.equals(
        old.sort_values(GRAIN).reset_index(drop=True))
    if unchanged:
        print("  content unchanged, artifact left alone", flush=True)
    else:
        atomic_write_parquet(df, P.TRANSCRIPT_INDEX)
    run_state.save(STAGE, {
        "last_run_utc": utc_now(),
        "max_creation_date": str(df["created_date"].max()),
        "watermark_margin_days": C.WATERMARK_MARGIN_DAYS,
        "counts_by_year": {str(k): v for k, v in sorted(counts.items())},
        "n_rows": len(df), "git_sha": git_sha(),
    })
    append_manifest({"stage": STAGE, "artifact": P.TRANSCRIPT_INDEX.name, "n_rows": len(df),
                     "n_calls": int(df.keydevid.nunique()), "rows_added": added,
                     "watermark": str(df["created_date"].max()),
                     "pull_ts": utc_now(), "git_sha": git_sha()})
    print(f"{'unchanged' if unchanged else 'wrote'} {P.TRANSCRIPT_INDEX}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
