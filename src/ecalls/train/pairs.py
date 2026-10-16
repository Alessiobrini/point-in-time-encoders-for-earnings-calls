"""Contrastive training pairs for one vintage, restricted to its window and free of held-out calls.

This is the easy thing to get wrong: a pair built from a 2024 call cannot be used to train the 2015
vintage, and the pair construction code is separate from the corpus filter. So both filters are applied
here and both are recorded in the sidecar.

**The extraction rule is imported from `diagnostics.bow_vs_dense` rather than restated.** That module
defines what counts as a question and an answer for the retrieval benchmark, and Phase 4 steers by the
number that benchmark produces. Two copies of the rule would let training pairs and evaluation pairs drift
apart, and the drift would show up as an unexplained gap between training loss and benchmark score. The
import direction is unusual, a training module depending on a diagnostic, and it is the price of one
definition.

Hard negatives come from the same call. Each call's other executive answers share the company, the
quarter, the speakers and the topic, so separating them requires reading the question rather than matching
vocabulary or guessing the firm. That is also the pool the benchmark scores against, which makes the
training signal and the evaluation the same task rather than two related ones.

Run:
    python -m ecalls.train.pairs --cutoff 2010-12-31
"""
import argparse
import json

import polars as pl

from ecalls.config import paths as P
from ecalls.diagnostics.bow_vs_dense import ANSWER_SPEAKER, MIN_CHARS, QUESTION_SPEAKER, qa_pairs
from ecalls.embed.encode import shards_up_to
from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.train import holdout as HO

PAIRS_DIR = P.PROCESSED_DIR / "pairs"
ALL_CALLS = 10**9        # qa_pairs samples only when the shard holds more calls than this


def pairs_dir(name: str):
    return PAIRS_DIR / name


def pairs_name(cutoff: str, holdout_only: bool = False) -> str:
    suffix = "-holdout" if holdout_only else ""
    return f"qa-pairs-{cutoff.replace('-', '')}{suffix}"


def build_pairs(cutoff: str, extra_years: int = 0, min_chars: int = MIN_CHARS,
                holdout_fraction: float = HO.DEFAULT_FRACTION, holdout_salt: str = HO.SALT,
                limit_shards: int | None = None, quiet: bool = False,
                holdout_only: bool = False) -> dict:
    """Write every question-answer pair inside the window, minus held-out calls.

    `holdout_only` inverts the split and keeps only the held-out calls, which is the evaluation set.
    It is the same structure the benchmark scores, so the contrastive stage can be monitored on the
    number Phase 4 steers by without waiting for a separate evaluation pass.
    """
    if holdout_only and not holdout_fraction:
        raise ValueError(
            "holdout_only with holdout_fraction=0 would keep every call and call it the held-out "
            "set, so training and evaluation would share every pair.")
    shards = shards_up_to(cutoff, extra_years=extra_years)
    if limit_shards:
        shards = shards[:limit_shards]
    if not shards:
        raise SystemExit(f"no chunk shard ends on or before {cutoff}")

    name = pairs_name(cutoff, holdout_only)
    d = pairs_dir(name)
    d.mkdir(parents=True, exist_ok=True)

    frames = []
    n_held = 0
    for s in shards:
        q, a = qa_pairs(s, ALL_CALLS, min_chars=min_chars)
        if q.height == 0:
            continue
        if holdout_fraction:
            calls = q["keydevid"].unique().to_list()
            keep = {k for k in calls
                    if HO.is_holdout(k, holdout_fraction, holdout_salt) == holdout_only}
            n_held += len(calls) - len(keep)
            q = q.filter(pl.col("keydevid").is_in(keep))
            a = a.filter(pl.col("keydevid").is_in(keep))
        if q.height == 0:
            continue
        # The answer text joined onto each query, so the trainer reads one row per pair. The pool for
        # in-call negatives is recoverable by grouping the answers frame on keydevid.
        merged = q.join(a.select(["answer_id", "text"]).rename({"text": "answer_text"}),
                        on="answer_id", how="inner")
        if merged.height != q.height:
            raise ValueError(
                f"{s}: {q.height} queries but {merged.height} joined to an answer. Every query's gold "
                "answer must be present, or the pair is unusable and the count is silently short.")
        frames.append(merged.with_columns(shard=pl.lit(s)))
        if not quiet:
            print(f"  {s}: {merged.height:,} pairs", flush=True)

    if not frames:
        raise SystemExit(f"no pairs survived the filters for {cutoff}")
    out = pl.concat(frames).sort(["keydevid", "query_id"])   # deterministic
    path = d / "pairs.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    out.write_parquet(tmp)
    tmp.replace(path)

    meta = {
        "name": name,
        "cutoff": cutoff,
        "extra_years": extra_years,
        "n_pairs": out.height,
        "n_calls": out["keydevid"].n_unique(),
        "n_calls_held_out": n_held,
        "holdout_only": holdout_only,
        "min_chars": min_chars,
        "question_speaker": QUESTION_SPEAKER,
        "answer_speaker": ANSWER_SPEAKER,
        "extraction_rule": "diagnostics.bow_vs_dense.qa_pairs, imported so training and evaluation "
                           "cannot drift apart",
        "shards": shards,
        "n_shards": len(shards),
        "bytes": path.stat().st_size,
        **HO.describe(holdout_fraction, holdout_salt),
        "git_sha": git_sha(),
        "built_at": utc_now(),
    }
    (d / "_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_pairs(name: str):
    d = pairs_dir(name)
    f = d / "_meta.json"
    if not f.exists():
        raise FileNotFoundError(
            f"no pairs at {d}. Build them with "
            f"`python -m ecalls.train.pairs --cutoff <date>`.")
    return pl.read_parquet(d / "pairs.parquet"), json.loads(f.read_text())


def pool_sizes(df: pl.DataFrame) -> pl.DataFrame:
    """Per call, how many pairs and therefore how many in-call negatives a batch can draw."""
    return (df.group_by("keydevid").agg(pl.len().alias("n_pairs"))
            .sort("n_pairs", descending=True))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--extra-years", type=int, default=0,
                    help="0 for training: a vintage's pairs must not come from after its cutoff")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS)
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--holdout-only", action="store_true",
                    help="keep only the held-out calls, for evaluation instead of training")
    args = ap.parse_args(argv)

    m = build_pairs(args.cutoff, args.extra_years, args.min_chars, limit_shards=args.limit_shards,
                    holdout_only=args.holdout_only)
    print(f"\n{m['name']}")
    print(f"  {m['n_pairs']:,} pairs over {m['n_calls']:,} calls, {m['n_shards']} shards up to "
          f"{m['cutoff']}")
    print(f"  {m['n_calls_held_out']:,} calls held out at {m['holdout_fraction']:.0%}, "
          f"{m['bytes']/1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
