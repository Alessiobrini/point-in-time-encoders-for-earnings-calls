"""Call-level embeddings for the premise panel, each from a vintage that never saw the future.

One vector per call, the mean of its chunk embeddings, L2 normalized. Mean pooling over chunks
rather than concatenation because the number of chunks varies ninefold across calls and a predictor
cannot take a ragged input.

**The vintage is resolved per call, not chosen once.** A call on 2017-04-25 is embedded by the 2016
checkpoint, because that is the newest model that saw no text after the call. Using one late vintage
for the whole panel is the look-ahead this project exists to prevent, and it would inflate exactly
the number the premise test is trying to measure. `registry.for_date` enforces it, and it now needs a
family because three families share every cutoff.

**Or one vintage embeds every call (`--model`), which is D11's design.** At a decision date one
model embeds the whole history, so a regression fitted on the result sees one embedding space. Per-call
resolution mixes up to ten spaces in one design matrix. ChronoBERT tolerates that because each of its
checkpoints continues the previous one, but our retrievers are pretrained from scratch independently,
so their spaces are unrelated and a ridge fitted across them would partly fit which vintage produced a
row. The single model must have a cutoff no later than the regression's training cut, which
`predictive_premise.run_models` checks.

`--encode` embeds the chunks here instead of reading the corpus-wide parts, which exist only for the
ChronoBERT vintages. The panel is about 1,500 calls, so this is minutes on a GPU, against the full Phase
5 pass it would otherwise wait for.

Run on the cluster, where the embeddings are:
    PYTHONPATH=src python3 scripts/call_embeddings.py --family chrono-bert-v1
    PYTHONPATH=src python3 scripts/call_embeddings.py --model ecalls-ret-20211231 --encode
"""
import argparse
import json

import numpy as np
import polars as pl

from ecalls.chunk import registry as R
from ecalls.config import paths as P
from ecalls.embed import encode as E
from ecalls.ingest.manifest import git_sha, utc_now

PANEL = P.PROCESSED_DIR / "premise_panel.parquet"


def call_vector(vecs: np.ndarray) -> list[float]:
    """The mean of a call's chunk vectors, L2 normalized, as a float32 list."""
    v = vecs.mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n if n else v).astype(np.float32).tolist()


def pool_by_call(V: np.ndarray, kd: np.ndarray, section: np.ndarray | None = None) -> dict:
    """Call vectors from chunk vectors in one sort, instead of one scan of every chunk per call.

    Returns {keydevid: (n_chunks, all, qa, presentation)}, the section means None when a call has no
    chunk of that section. The scan this replaces was quadratic and would not finish on the full
    universe, about four million chunks across 150,000 calls.
    """
    order = np.argsort(kd, kind="stable")
    kd_s = kd[order]
    keys, starts = np.unique(kd_s, return_index=True)
    ends = np.append(starts[1:], len(kd_s))
    out = {}
    for k, a, b in zip(keys, starts, ends):
        idx = order[a:b]
        qa = pres = None
        if section is not None:
            sec = section[idx]
            q, pr = idx[sec == "qa"], idx[sec == "presentation"]
            qa = call_vector(V[q]) if len(q) else None
            pres = call_vector(V[pr]) if len(pr) else None
        out[int(k)] = (int(len(idx)), call_vector(V[idx]), qa, pres)
    return out


def encode_calls(panel: pl.DataFrame, reg, device=None, batch_size: int = 64,
                 sections: bool = False) -> tuple[list, int]:
    """Embed every chunk of each call with the model the call is assigned, then pool to the call.

    One encoder load per model and one read per shard. Pooling and the length cap come from the
    registry entry, as in the benchmark, so a CLS-trained model is not mean-pooled by accident. With
    `sections`, each call also gets its Q&A-only and presentation-only means from the same pass (D28).
    """
    rows, missing = [], 0
    for mid in sorted(panel["model_id"].unique().to_list()):
        entry = reg.by_name(mid)
        tok, model, dev = E.load_encoder(entry, device)
        sub = panel.filter(pl.col("model_id") == mid)
        # One shard at a time: a call lives in one quarterly shard, so pooling per shard gives the
        # same vectors as pooling everything, and memory stays at one shard's chunks rather than the
        # whole universe's, which on D28's universe is millions of texts and their vectors.
        n_chunks = 0
        for shard in sorted(sub["shard"].unique().to_list()):
            want = set(sub.filter(pl.col("shard") == shard)["keydevid"].to_list())
            chunks = (pl.read_parquet(P.chunks_shard(shard), columns=["chunk_id", "keydevid", "text", "section"])
                      .filter(pl.col("keydevid").is_in(want)))
            missing += len(want - set(chunks["keydevid"].to_list()))
            if not chunks.height:
                continue
            n_chunks += chunks.height
            V = E.encode_texts(chunks["text"].to_list(), tok, model, dev, batch_size=batch_size,
                               max_length=E.effective_max_length(entry), pooling=entry.pooling)
            kd = chunks["keydevid"].to_numpy()
            sec = chunks["section"].to_numpy() if sections else None
            for k, (n, v_all, v_qa, v_pres) in pool_by_call(V, kd, sec).items():
                row = {"keydevid": k, "model_id": mid, "n_chunks": n, "embedding": v_all}
                if sections:
                    row["embedding_qa"], row["embedding_pres"] = v_qa, v_pres
                rows.append(row)
            print(f"  {mid} {shard}: {len(want):,} calls, {chunks.height:,} chunks", flush=True)
        print(f"  {mid}: {sub.height:,} calls, {n_chunks:,} chunks", flush=True)
        del model
    return rows, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--family", default=None, help="resolve a vintage per call from this family")
    g.add_argument("--model", default=None, help="one vintage embeds every call (D11)")
    ap.add_argument("--encode", action="store_true",
                    help="embed the chunks here rather than reading corpus-wide parts")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not args.family and not args.model:
        args.family = "chrono-bert-v1"

    panel = pl.read_parquet(PANEL)
    reg = R.load()
    label = args.model or args.family
    out_path = P.PROCESSED_DIR / (args.out or f"premise_embeddings_{label}.parquet")

    if args.model:
        entry = reg.by_name(args.model)
        if entry.kind != R.VINTAGE:
            raise SystemExit(f"{args.model} is {entry.kind}, so it has no cutoff to check")
        panel = panel.with_columns(model_id=pl.lit(args.model))
    else:
        # Which vintage each call needs, resolved from its own date.
        vintages = {}
        for d in panel["call_date"].unique().to_list():
            vintages[d] = reg.for_date(d, family=args.family).model_id
        panel = panel.with_columns(
            model_id=pl.col("call_date").map_elements(lambda d: vintages[d], return_dtype=pl.String))
    print(f"{panel.height:,} calls resolve to {panel['model_id'].n_unique()} vintages:")
    for mid, n in sorted(panel.group_by("model_id").len().iter_rows()):
        print(f"  {mid}  {n:,} calls")

    if args.encode:
        rows, missing = encode_calls(panel, reg, args.device)
    else:
        rows, missing = read_parts(panel)
    _write(rows, missing, panel, label, args, out_path)
    return 0


def read_parts(panel: pl.DataFrame) -> tuple[list, int]:
    rows, missing = [], 0
    # Grouped by (shard, model) because that pair is one part file, and reading it once per call
    # would reread a multi-hundred-megabyte file thousands of times.
    for (shard, mid), grp in panel.group_by(["shard", "model_id"]):
        want_calls = set(grp["keydevid"].to_list())
        chunks = pl.read_parquet(P.chunks_shard(shard), columns=["chunk_id", "keydevid"]).filter(
            pl.col("keydevid").is_in(want_calls))
        part = E.part_path(mid, shard)
        if not part.exists():
            missing += len(want_calls)
            print(f"  MISSING {part}")
            continue
        ids, vecs = E.read_vectors(part)
        index = {cid: i for i, cid in enumerate(ids)}
        for kd, sub in chunks.group_by("keydevid"):
            kd = kd[0] if isinstance(kd, tuple) else kd
            take = [index[c] for c in sub["chunk_id"].to_list() if c in index]
            if not take:
                missing += 1
                continue
            rows.append({"keydevid": int(kd), "model_id": mid, "n_chunks": len(take),
                         "embedding": call_vector(vecs[take])})
    return rows, missing


def _write(rows, missing, panel, label, args, out_path) -> None:
    df = pl.DataFrame(rows)
    df.write_parquet(out_path)
    meta = {"family": args.family, "model": args.model, "encoded_here": args.encode,
            "n_calls_requested": panel.height, "n_calls_embedded": df.height,
            "n_missing": missing, "dim": len(rows[0]["embedding"]) if rows else 0,
            "vintages_used": sorted(panel["model_id"].unique().to_list()),
            "git_sha": git_sha(), "built_at": utc_now()}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\n{df.height:,} of {panel.height:,} calls embedded, dim {meta['dim']}, "
          f"{missing} missing -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
