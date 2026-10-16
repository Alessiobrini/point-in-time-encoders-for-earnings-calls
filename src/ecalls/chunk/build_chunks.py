"""Phase 3: build the chunk layer from the component shards.

Sharded by the same calendar quarters as the components, so a chunk shard can be rebuilt from one
component shard and the resume marker is per shard. The chunker is pure and lives in `chunker.py`;
this module is only I/O, the filter-column join, and provenance.

Run:
    python -m ecalls.chunk.build_chunks
    python -m ecalls.chunk.build_chunks --shards 2024Q1 --keep-operator
"""
import argparse
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ecalls.chunk import chunker
from ecalls.config import constants as C
from ecalls.config import paths as P
from ecalls.config.schemas import CHUNKS as CHUNK_SCHEMA
from ecalls.ingest.manifest import append_manifest, git_sha, utc_now

STAGE = "build_chunks"
READ_COLS = ["keydevid", "component_id", "component_order", "section", "component_type_id",
             "text", "speaker_name", "speaker_type", "text_sha1"]


def call_filters() -> pd.DataFrame:
    """The columns copied onto every chunk so retrieval can pre-filter without a join."""
    return pd.read_parquet(P.CALLS, columns=["keydevid", "companyid", "permno", "call_date",
                                             "in_index_at_call", "shard"])


def chunk_shard(shard: str, filters: pd.DataFrame, version: str,
                drop_operator: bool) -> pd.DataFrame:
    src = P.components_shard(shard)
    comp = pq.read_table(src, columns=READ_COLS).to_pandas()
    comp = comp.sort_values(["keydevid", "component_order"])
    rows = []
    for _kd, g in comp.groupby("keydevid", sort=True):
        rows.extend(chunker.chunk_call(g.to_dict("records"), version,
                                       drop_operator=drop_operator))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).merge(
        filters.drop(columns=["shard"]), on="keydevid", how="left")
    df["permno"] = df["permno"].astype("Int32")
    df["call_date"] = pd.to_datetime(df["call_date"]).dt.date
    df["in_index_at_call"] = df["in_index_at_call"].fillna(False).astype(bool)
    df["shard"] = shard
    df["pull_ts"] = utc_now()
    df["git_sha"] = git_sha()
    for c in ("n_components", "char_start", "char_end", "n_chars", "n_tokens_est"):
        df[c] = df[c].astype("int32")
    return df.sort_values(["keydevid", "char_start"]).reset_index(drop=True)


def _write(df: pd.DataFrame, path) -> None:
    cols = [f.name for f in CHUNK_SCHEMA]
    tbl = pa.Table.from_pandas(df[cols], schema=CHUNK_SCHEMA, preserve_index=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(tbl, tmp, compression="zstd")
    tmp.replace(path)


def shard_is_current(path, version: str) -> bool:
    """Done only if the shard exists AND was built by this chunker version, since a version bump
    changes every chunk_id and therefore invalidates every embedding keyed on it."""
    if not path.exists():
        return False
    try:
        got = pq.read_table(path, columns=["chunker_version"])["chunker_version"].to_pylist()
    except Exception:  # noqa: BLE001
        return False
    return bool(got) and set(got) == {version}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="*", default=None)
    ap.add_argument("--keep-operator", action="store_true",
                    help="keep operator turns, which are dropped by default as boilerplate")
    args = ap.parse_args(argv)

    P.ensure_dirs()
    version = C.CHUNKER_VERSION + ("-withop" if args.keep_operator else "")
    filters = call_filters()
    shards = args.shards or sorted(filters["shard"].dropna().unique())
    print(f"chunker={version}  shards={len(shards)}  calls={len(filters):,}", flush=True)

    t0 = time.time()
    tot = tot_chars = done = skipped = 0
    for sh in shards:
        path = P.chunks_shard(sh)
        if shard_is_current(path, version):
            skipped += 1
            continue
        t1 = time.time()
        df = chunk_shard(sh, filters, version, drop_operator=not args.keep_operator)
        if df.empty:
            print(f"[{sh}] no chunks", flush=True)
            continue
        _write(df, path)
        tot += len(df); tot_chars += int(df.n_chars.sum()); done += 1
        print(f"[{sh}] {len(df):>8,} chunks  {df.n_chars.sum()/1e6:>6.1f} MB  "
              f"mean {df.n_chars.mean():>5.0f} ch  ({time.time()-t1:>4.1f}s)", flush=True)
        append_manifest({"stage": STAGE, "artifact": path.name, "n_rows": len(df),
                         "n_calls": int(df.keydevid.nunique()),
                         "total_chars": int(df.n_chars.sum()),
                         "chunker_version": version,
                         "pull_ts": utc_now(), "git_sha": git_sha()})

    print(f"\ndone: {done} shards written, {skipped} current  |  {tot:,} chunks, "
          f"{tot_chars/1e9:.2f} G chars  in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
