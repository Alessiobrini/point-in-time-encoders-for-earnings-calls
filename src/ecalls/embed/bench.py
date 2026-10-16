"""Measure embedding throughput under several batching configurations on one shard.

Why this exists as its own module rather than a one-off script: the batching choice is remeasured at
every phase, because Phase 5 embeds 17 expanding vintages over roughly 111 M chunk-embeddings and the
hardware on `scavenger-gpu` is whatever the preemption gods hand out. A configuration that was right
on an A5000 is not automatically right on a 2080 or a 6000 Ada.

It reports the **marginal** rate on a stratified sample rather than a cumulative rate over a
length-sorted shard. The cumulative figure on this corpus decays from 979 to 143 chunk/s across one
shard with nothing changing, purely because the short chunks are encoded first, and reading the first
number as the throughput overstates it by 7x.
"""
import argparse
import json
import time

import numpy as np
import polars as pl

from ecalls.chunk import registry as R
from ecalls.config import paths as P
from ecalls.embed import encode as E


def stratified_sample(texts: list[str], n: int, seed: int = 0) -> list[str]:
    """`n` texts spread evenly across the length distribution.

    A head slice is not a sample here: chunk length correlates with the section a chunk came from and
    with the era, so `texts[:n]` measures the throughput of whichever quarter's short prepared remarks
    happen to sort first.
    """
    if n >= len(texts):
        return list(texts)
    order = np.argsort([len(t) for t in texts], kind="stable")
    take = np.linspace(0, len(order) - 1, n).round().astype(int)
    rng = np.random.default_rng(seed)
    idx = order[take]
    rng.shuffle(idx)
    return [texts[i] for i in idx]


def run(model_id: str, shard: str, n: int, configs: list[dict], device=None,
        compute_dtype: str = "float16", require_gpu: bool = False) -> list[dict]:
    entry = R.load().by_name(model_id)
    tok, model, dev = E.load_encoder(entry, device, compute_dtype, require_gpu)
    texts = pl.read_parquet(P.chunks_shard(shard), columns=["text"])["text"].to_list()
    sample = stratified_sample(texts, n)
    chars = sum(len(t) for t in sample)

    E.encode_texts(sample[:64], tok, model, dev, batch_size=16)      # warm the kernels

    out = []
    for cfg in configs:
        t0 = time.time()
        E.encode_texts(sample, tok, model, dev, max_length=E.MAX_LENGTH, **cfg)
        el = time.time() - t0
        n_batches = len(E.plan_batches([len(t) for t in sample],
                                       cfg.get("batch_size", E.DEFAULT_BATCH),
                                       cfg.get("char_budget")))
        row = {**cfg, "device": dev, "compute_dtype": compute_dtype, "n": len(sample),
               "seconds": round(el, 2), "chunk_per_s": round(len(sample) / el, 1),
               "chars_per_s": round(chars / el), "n_batches": n_batches}
        out.append(row)
        print("RESULT " + json.dumps(row), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="chrono-bert-v1-20241231")
    ap.add_argument("--shard", default="2019Q3")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--compute-dtype", default="float16")
    ap.add_argument("--require-gpu", action="store_true",
                    help="fail rather than benchmarking CPU throughput by accident")
    args = ap.parse_args(argv)

    # Budgets are capped at what the allocated device can hold, because the first benchmark run died
    # of OOM on a 10.58 GB card at a fixed batch of 256 before producing a single number.
    auto = E.auto_char_budget(E.resolve_device(args.device)) or E.MAX_CHAR_BUDGET
    configs = [{"batch_size": 64}, {"batch_size": 256}]
    configs += [{"char_budget": b} for b in (131_072, 262_144, 524_288, 1_048_576) if b <= auto]
    configs += [{"char_budget": auto}] if auto not in (131_072, 262_144, 524_288, 1_048_576) else []
    rows = run(args.model, args.shard, args.n, configs, args.device, args.compute_dtype,
               args.require_gpu)
    best = max(rows, key=lambda r: r["chunk_per_s"])
    print(f"\nbest: {best['chunk_per_s']} chunk/s at "
          f"{ {k: v for k, v in best.items() if k in ('batch_size', 'char_budget')} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
