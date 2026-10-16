"""Reconcile an embedding set against the chunk layer it was built from.

The phase 1 lesson, applied to phase 3. `completeness.py` exists because a corpus that reconciles
exactly can still sit behind a derived layer that quietly lost rows, and every silent loss in this
project so far happened in a derived layer rather than in the pull. An embedding set is a derived
layer written by sixteen concurrent tasks on a preemptible partition, which is the most likely place
for the next one.

Every chunk is EMBEDDED or MISSING, and **MISSING is the gate: the target is zero.** A shard with no
part file at all is reported as a whole missing shard rather than being skipped, because the failure
that matters is a shard nobody noticed was never written.

The checks past coverage exist because a vector can be present and useless:

- **Dimension** against the registry, since a mismatch means the registry describes something else.
- **L2 norm** near 1.0 for a normalized entry. A row of zeros has norm 0 and passes every count-based
  check, and mean pooling over an all-padding mask is how you get one.
- **Finiteness.** A NaN propagates through cosine similarity into every neighbour list silently.
- **Model identity** from the parquet metadata, because two embedding sets differing only by model
  are the one mix-up no downstream consumer could detect.

Runs offline from the artifacts on disk. No WRDS and no GPU.

Run:
    python -m ecalls.audit.embeddings --model chrono-bert-v1-20241231
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from ecalls.chunk import registry as R
from ecalls.config import paths as P
from ecalls.embed import encode as E
from ecalls.embed.encode import model_dir
from ecalls.ingest.manifest import append_manifest, git_sha, utc_now

STAGE = "embedding_audit"
NORM_TOL = 0.01        # float16 storage moves a unit vector by about 5e-4; 0.01 is slack, not noise
DEFAULT_EXTRA_YEARS = 1   # must match `encode.main`'s --extra-years default, so the audit expects what
                          # the builder produces rather than something narrower


def audit_shard(shard: str, entry: R.Entry, sample_rows: int = 20_000) -> dict:
    """One shard's row. `sample_rows` bounds the norm and finiteness pass, which would otherwise read
    18.6 GB of vectors to answer a question a sample answers."""
    src = P.chunks_shard(shard)
    part = E.part_path(entry.model_id, shard)
    expected = pl.read_parquet(src, columns=["chunk_id"])["chunk_id"].to_list()
    row = {"shard": shard, "n_chunks": len(expected), "part_exists": part.exists(),
           "n_embedded": 0, "n_missing": len(expected), "n_extra": 0, "dim": None,
           "norm_min": None, "norm_max": None, "n_nonfinite": 0, "n_zero_rows": 0,
           "model_id_in_file": None, "bytes": 0, "error": "", "status": "NO_PART_FILE"}
    if not part.exists():
        return row

    # An unreadable part file must be a reported row, not an exception. This audit loops over 88
    # shards, and a raise partway through says nothing about the other 87, which is the opposite of
    # what an audit is for. The case is real: three concurrent writers on one shard could have left a
    # spliced parquet whose footer does not parse.
    try:
        schema = pq.read_schema(part)
        meta = {k.decode(): v.decode() for k, v in (schema.metadata or {}).items()}
        got = pl.read_parquet(part, columns=["chunk_id"])["chunk_id"].to_list()
    except Exception as exc:      # noqa: BLE001 -- any read failure is the same finding
        row["status"] = "CORRUPT"
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        row["bytes"] = part.stat().st_size
        return row
    got_set, exp_set = set(got), set(expected)
    row.update({
        "n_embedded": len(got_set & exp_set),
        "n_missing": len(exp_set - got_set),
        "n_extra": len(got_set - exp_set),
        "dim": schema.field("embedding").type.list_size,
        "model_id_in_file": meta.get("model_id"),
        "bytes": part.stat().st_size,
    })

    # the vector health pass, on a bounded sample
    try:
        t = pq.read_table(part, columns=["embedding"])
    except Exception as exc:      # noqa: BLE001 -- a readable footer can still have a bad column
        row["status"] = "CORRUPT"
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return row
    n = t.num_rows
    take = np.linspace(0, n - 1, min(sample_rows, n)).round().astype(int) if n else np.array([], int)
    if len(take):
        v = np.stack(t.take(take)["embedding"].to_numpy(zero_copy_only=False)).astype(np.float32)
        norms = np.linalg.norm(v, axis=1)
        row.update({"norm_min": float(norms.min()), "norm_max": float(norms.max()),
                    "n_nonfinite": int((~np.isfinite(v)).any(axis=1).sum()),
                    "n_zero_rows": int((norms < 1e-6).sum())})

    row["status"] = classify(row, entry)
    return row


def classify(row: dict, entry: R.Entry) -> str:
    """The first failing check, so a shard's status names one thing to fix rather than a list."""
    if not row["part_exists"]:
        return "NO_PART_FILE"
    if row.get("error"):
        return "CORRUPT"
    if row["n_missing"]:
        return "INCOMPLETE"
    if row["dim"] != entry.dim:
        return "WRONG_DIM"
    if row["model_id_in_file"] not in (None, entry.model_id):
        return "WRONG_MODEL"
    if row["n_nonfinite"]:
        return "NON_FINITE"
    if row["n_zero_rows"]:
        return "ZERO_ROWS"
    if entry.normalized and row["norm_min"] is not None and (
            abs(row["norm_min"] - 1.0) > NORM_TOL or abs(row["norm_max"] - 1.0) > NORM_TOL):
        return "BAD_NORM"
    if row["n_extra"]:
        return "OK_WITH_EXTRA"      # not a failure: a superset still covers every expected chunk
    return "OK"


def expected_shards(entry: R.Entry, extra_years: int | None = None) -> tuple[list[str], str]:
    """The shards this set is supposed to hold, and a one-line description of why.

    A vintage set is built on an expanding window, so all 88 shards is the wrong expectation for 14 of
    the 15 vintages and would report their intentionally absent later quarters as missing. That turns the
    MISSING gate into noise, which is worse than having no gate: a real absence would sit unnoticed in a
    list of expected ones.

    The window is read from the set's own `_meta.json` when it records one, and otherwise inferred from
    the entry's cutoff. `extra_years` overrides both.
    """
    allshards = E.shard_keys()
    meta = read_set_meta(entry.model_id)
    if extra_years is None and meta.get("expected_shards"):
        want = [s for s in allshards if s in set(meta["expected_shards"])]
        return want, f"{len(want)} shards recorded in the set's own _meta.json"
    if entry.kind != R.VINTAGE or entry.cutoff is None:
        return allshards, f"all {len(allshards)} shards, since {entry.model_id} is {entry.kind}"
    # **Default to the wider window, not the narrower one.** When the sidecar predates the
    # `extra_years` field this fallback decides what "complete" means, and defaulting to 0 made the
    # audit agree with whatever happened to be on disk: it passed the 2022 and 2023 vintages with
    # MISSING = 0 while both were four shards short of the window they were meant to cover. An audit
    # that is unsure must expect more and report a gap, never expect less and pass.
    ey = extra_years if extra_years is not None else meta.get("extra_years")
    if ey is None:
        ey = DEFAULT_EXTRA_YEARS
    want = E.shards_up_to(entry.cutoff, allshards, ey)
    return want, (f"{len(want)} shards ending on or before the cutoff {entry.cutoff} plus "
                  f"{ey} year(s)")


def read_set_meta(model_id: str) -> dict:
    f = E.meta_path(model_id)
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except Exception:      # noqa: BLE001 -- a damaged sidecar must not stop the audit
        return {}


def run(model_id: str, sample_rows: int = 20_000, out=None,
        extra_years: int | None = None) -> pd.DataFrame:
    entry = R.load().by_name(model_id)
    if not E.shard_keys():
        raise SystemExit(f"no chunk shards under {P.CHUNKS_DIR}; build the chunk layer first")
    shards, why = expected_shards(entry, extra_years)
    print(f"  expecting {why}")
    rows = [audit_shard(s, entry, sample_rows) for s in shards]
    df = pd.DataFrame(rows)
    strays = sorted(p.name for p in model_dir(model_id).glob("*.tmp")) if \
        model_dir(model_id).exists() else []
    if strays:
        # A temp file beside the parts means a writer died mid-write. Harmless to the data, and worth
        # printing because it says a task failed without saying so anywhere else.
        print(f"  NOTE {len(strays)} stray temp file(s) from an interrupted write: {strays[:5]}")
    out = out or (P.PROCESSED_DIR / f"embedding_audit_{model_id}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def report(df: pd.DataFrame, entry: R.Entry) -> int:
    """Print the reconciliation and return the process exit code."""
    tot_chunks = int(df["n_chunks"].sum())
    tot_emb = int(df["n_embedded"].sum())
    tot_missing = int(df["n_missing"].sum())
    gb = df["bytes"].sum() / 1e9

    print(f"{entry.model_id}: dim {entry.dim}, cutoff {entry.training_cutoff}")
    print(f"  shards          {len(df)} with a part file: {int(df['part_exists'].sum())}")
    print(f"  chunks expected {tot_chunks:,}")
    print(f"  chunks embedded {tot_emb:,}")
    print(f"  MISSING         {tot_missing:,}   <- the gate, target zero")
    print(f"  on disk         {gb:.1f} GB")
    counts = df["status"].value_counts().to_dict()
    print(f"  status          {counts}")

    bad = df[~df["status"].isin(("OK", "OK_WITH_EXTRA"))]
    if not bad.empty:
        print("\n  shards needing attention:")
        for _, r in bad.iterrows():
            print(f"    {r['shard']}  {r['status']}  missing {r['n_missing']:,} of {r['n_chunks']:,}")
    return 0 if tot_missing == 0 and bad.empty else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="chrono-bert-v1-20241231")
    ap.add_argument("--extra-years", type=int, default=None,
                    help="override the expanding window: expect shards up to the cutoff plus this many "
                         "years. Defaults to whatever the set's _meta.json recorded")
    ap.add_argument("--sample-rows", type=int, default=20_000,
                    help="rows per shard for the norm and finiteness pass; 0 reads all of them")
    args = ap.parse_args(argv)

    entry = R.load().by_name(args.model)
    df = run(args.model, args.sample_rows or 10**9, extra_years=args.extra_years)
    code = report(df, entry)
    append_manifest({"stage": STAGE, "model_id": args.model, "n_shards": len(df),
                     "n_chunks": int(df["n_chunks"].sum()),
                     "n_embedded": int(df["n_embedded"].sum()),
                     "n_missing": int(df["n_missing"].sum()),
                     "bytes": int(df["bytes"].sum()), "exit_code": code,
                     "git_sha": git_sha(), "ts": utc_now()})
    return code


if __name__ == "__main__":
    sys.exit(main())
