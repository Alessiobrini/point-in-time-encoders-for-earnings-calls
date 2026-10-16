"""Stage 6: pull the component text for the selected transcript of every call.

Sharded by calendar quarter of the call date, about 88 shards (D7). The index on `transcriptid` makes
the server's access path free, so the pull is wire-bound and sharding is chosen for downstream
convenience and blast radius rather than for speed: quarter is the partition key every consumer
wants, and a crash loses minutes.

One connection, sequential, batches of 250 ids. WRDS throttles and the whole pull is short, so
parallel connections buy nothing. A batch that fails after its retries is logged to
`_failed_batches.csv`, which the completeness audit reads so an unrecoverable batch becomes a logged
DROPPED rather than a silent gap.

Run:
    python -m ecalls.ingest.pull_components --universe sp500    # the checkpoint, ~2.65 GB
    python -m ecalls.ingest.pull_components                     # everything, ~16 GB
    python -m ecalls.ingest.pull_components --shards 2024Q1 2024Q2
"""
import argparse
import csv
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ecalls.config import constants as C
from ecalls.config import paths as P
from ecalls.config.schemas import COMPONENTS
from ecalls.ingest import lookups as L
from ecalls.ingest.manifest import append_manifest, batched, git_sha, shard_key, utc_now

STAGE = "pull_components"

SQL = f"""
select tc.transcriptid, tc.transcriptcomponentid, tc.componentorder,
       tc.transcriptcomponenttypeid, tc.transcriptpersonid, tc.componenttext,
       p.transcriptpersonname, p.speakertypeid, p.companyname
from {C.TBL_COMPONENT} tc
left join {C.TBL_PERSON} p on p.transcriptpersonid = tc.transcriptpersonid
where tc.transcriptid = any(%s)
"""


def selected_calls(universe: str | None) -> pd.DataFrame:
    """One row per call: the chosen transcript, its company, its date, and its shard."""
    vc = pd.read_parquet(P.VERSION_CHOICE,
                         columns=["keydevid", "companyid", "call_date", "chosen_transcriptid"])
    if universe == "sp500":
        uni = pd.read_parquet(P.SP500_UNIVERSE, columns=["companyid"])
        cids = set(uni["companyid"].dropna().astype("int64"))
        routes = pd.read_parquet(P.CALL_COMPANIES, columns=["keydevid", "companyid"])
        keep = set(routes.loc[routes["companyid"].isin(cids), "keydevid"])
        vc = vc.loc[vc["keydevid"].isin(keep)].copy()
    vc["shard"] = pd.to_datetime(vc["call_date"]).map(shard_key)
    return vc.sort_values(["shard", "keydevid"]).reset_index(drop=True)


def build_frame(rows, meta: pd.DataFrame, shard: str) -> pd.DataFrame:
    """Decode, normalize, and attach call-level identifiers. Pure given `rows`."""
    df = pd.DataFrame(rows, columns=[
        "transcriptid", "component_id", "component_order", "component_type_id",
        "person_id", "raw_text", "speaker_name", "speaker_type_id", "speaker_company"])
    if df.empty:
        return df
    df["text"] = df["raw_text"].map(L.normalize_text)
    df["n_chars"] = df["text"].str.len().astype("int32")
    df["text_sha1"] = df["text"].map(L.text_sha1)
    df["component_type"] = df["component_type_id"].map(L.COMPONENT_TYPE)
    df["section"] = df["component_type_id"].map(L.SECTION)
    df["speaker_type"] = df["speaker_type_id"].map(L.SPEAKER_TYPE)

    m = meta.rename(columns={"chosen_transcriptid": "transcriptid"})[
        ["transcriptid", "keydevid", "companyid", "call_date"]]
    df = df.merge(m, on="transcriptid", how="inner")

    df["shard"] = shard
    df["cleaning_version"] = C.CLEANING_VERSION
    df["pull_ts"] = utc_now()
    df["git_sha"] = git_sha()
    for c in ("keydevid", "transcriptid", "companyid", "component_id"):
        df[c] = df[c].astype("int64")
    df["component_order"] = df["component_order"].astype("int32")
    for c in ("component_type_id", "speaker_type_id"):
        df[c] = df[c].astype("Int16")
    df["person_id"] = df["person_id"].astype("Int64")
    df["call_date"] = pd.to_datetime(df["call_date"]).dt.date
    return df.sort_values(["keydevid", "component_order"]).reset_index(drop=True)


def _write(df: pd.DataFrame, path) -> None:
    """Atomic write against the declared schema, so a wrong dtype fails here rather than at union."""
    cols = [f.name for f in COMPONENTS]
    tbl = pa.Table.from_pandas(df[cols], schema=COMPONENTS, preserve_index=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(tbl, tmp, compression="zstd")
    tmp.replace(path)


def shard_is_complete(path, expected_calls: int) -> bool:
    """A shard counts as done only when it holds every call expected for it under the CURRENT
    universe, not merely when the file exists.

    Existence alone is the wrong marker here because the shard name is the calendar quarter and
    carries no universe in it. Running the S&P 500 checkpoint first writes `components_2024Q1.parquet`
    holding only S&P 500 calls, and a later full-corpus run would then skip that file and drop every
    other company's calls in that quarter, with no error anywhere. Comparing the call count turns
    that silent gap into a re-pull, and it also catches a shard truncated by a crash.
    """
    if not path.exists():
        return False
    try:
        import pyarrow.parquet as pq
        got = pq.read_table(path, columns=["keydevid"])["keydevid"].to_pylist()
    except Exception:  # noqa: BLE001 -- an unreadable shard is an incomplete shard
        return False
    return len(set(got)) >= expected_calls


def log_failure(shard: str, tids, err: str) -> None:
    P.FAILED_BATCHES.parent.mkdir(parents=True, exist_ok=True)
    new = not P.FAILED_BATCHES.exists()
    with open(P.FAILED_BATCHES, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["shard", "n_transcripts", "transcriptids", "error", "ts", "git_sha"])
        w.writerow([shard, len(tids), " ".join(map(str, tids)), err[:300], utc_now(), git_sha()])


def main(argv=None) -> int:
    from ecalls.ingest.wrds_conn import connect

    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["sp500"], default=None,
                    help="restrict to companies EVER in the S&P 500 since 2004 (53,381 calls). This "
                         "is not the point-in-time slice: filter on calls.in_index_at_call for that, "
                         "which is 37,920 calls.")
    ap.add_argument("--shards", nargs="*", default=None, help="e.g. 2024Q1 2024Q2")
    args = ap.parse_args(argv)

    P.ensure_dirs()
    calls = selected_calls(args.universe)
    shards = args.shards or sorted(calls["shard"].unique())
    label = args.universe or "all"
    print(f"universe={label}  calls={len(calls):,}  shards={len(shards)}", flush=True)

    conn = connect(statement_timeout_ms=900_000)
    cur = conn.cursor()
    t_all = time.time()
    tot_rows = tot_chars = 0
    done = skipped = failed = 0

    for sh in shards:
        path = P.components_shard(sh)
        meta = calls.loc[calls["shard"] == sh]
        tids = meta["chosen_transcriptid"].astype("int64").tolist()
        if not tids:
            continue
        if shard_is_complete(path, len(meta)):
            skipped += 1
            continue
        if path.exists():
            print(f"[{sh}] rewriting: holds fewer calls than the {label} universe expects "
                  f"({len(meta):,})", flush=True)
        t0 = time.time()
        rows, bad = [], 0
        for b in batched(tids, C.COMPONENT_BATCH):
            try:
                cur.execute(SQL, (b,))
                rows.extend(cur.fetchall())
            except Exception as e:  # noqa: BLE001 -- one bad batch must not kill the run
                bad += 1
                log_failure(sh, b, f"{type(e).__name__}: {str(e).splitlines()[0]}")
                try:
                    cur.close(); conn.close()
                except Exception:  # noqa: BLE001
                    pass
                conn = connect(statement_timeout_ms=900_000); cur = conn.cursor()
        if not rows:
            print(f"[{sh}] NO ROWS ({len(tids):,} transcripts, {bad} failed batches)", flush=True)
            failed += 1
            continue
        df = build_frame(rows, meta, sh)
        _write(df, path)
        tot_rows += len(df); tot_chars += int(df.n_chars.sum()); done += 1
        dt = time.time() - t0
        print(f"[{sh}] {len(meta):>6,} calls  {len(df):>8,} components  "
              f"{df.n_chars.sum()/1e6:>7.1f} MB  ({dt:>5.1f}s)"
              + (f"  {bad} FAILED BATCHES" if bad else ""), flush=True)
        append_manifest({"stage": STAGE, "artifact": path.name, "n_rows": len(df),
                         "n_calls": int(meta.keydevid.nunique()),
                         "total_chars": int(df.n_chars.sum()), "failed_batches": bad,
                         "universe": label, "pull_ts": utc_now(), "git_sha": git_sha()})

    mins = (time.time() - t_all) / 60
    print(f"\ndone: {done} shards written, {skipped} skipped, {failed} empty  |  "
          f"{tot_rows:,} components, {tot_chars/1e9:.2f} GB text  in {mins:.1f} min", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
