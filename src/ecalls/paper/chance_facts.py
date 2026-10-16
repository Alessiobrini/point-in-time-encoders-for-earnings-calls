"""The within-call recall@1 a random ranking would reach, per test quarter of Fig. 2.

    python -m ecalls.paper.chance_facts

A random ranking puts a question's own answer first with probability one over the size of its call's
answer pool, so the chance level of a quarter is the mean of that over its questions. The questions are
rebuilt exactly as the benchmark draws them (held-out calls, 400 calls, at most 3,000 questions,
evaluation seed 0). Writes `reports/paper/chance_facts.json`.
"""
import json

import numpy as np
import polars as pl

from ecalls.config import paths as P
from ecalls.diagnostics import bow_vs_dense as B
from ecalls.diagnostics import seed_summary as S

OUT = P.REPORTS_DIR / "paper" / "chance_facts.json"


def chance(queries: pl.DataFrame, answers: pl.DataFrame) -> float:
    """Mean over questions of one over the number of candidate answers in the question's call."""
    pool = answers.group_by("keydevid").len()
    q = queries.join(pool, on="keydevid", how="left")
    if q["len"].null_count():
        raise ValueError("a question's call has no answer pool")
    return float((1.0 / q["len"]).mean())


def main() -> int:
    out = {}
    for y in S.YEARS:
        shard = S.shard_of(y)
        q, a = B.qa_pairs(shard, 400, 0, holdout=True)
        if q.height > 3000:
            q = q.sample(n=3000, seed=0)
        out[shard] = {"chance_within": chance(q, a), "n_questions": q.height, "n_answers": a.height}
        print(shard, round(out[shard]["chance_within"], 4))
    out["mean_chance_within"] = float(np.mean([v["chance_within"] for v in out.values()]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
