"""Gate for the evaluation-seed array: seed 0 must reproduce the canonical comparison exactly.

`cluster/bench_seeds.sbatch` reruns the benchmark at evaluation seeds 0, 1 and 2 with the fusion scan
and per-question ranks added. Seed 0 draws the same calls and questions as the canonical files the
paper reads, `reports/cmp_<shard>_<model>_holdout.json`. TF-IDF and BM25 run on the CPU and must agree
to the last digit. The dense and hybrid scorers embed in float16 on whatever card the task drew, and a
different card can move a near-tied pair by one place, so they must agree within `DENSE_TOLERANCE`
of recall@1 and MRR. The scan's weight 0.5 must equal the unweighted hybrid of the same run exactly.
Anything else means the rerun is not measuring the same thing, and seeds 1 and 2 could not be read as
variation around the canonical number.

The expectation comes from the canonical files and the plan's 45 pairs, never from what the seed
directory happens to hold, so a missing output fails rather than shrinking the check.

    python3 scripts/check_seed_reproduction.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
EXACT = ("tfidf", "bm25")
NEAR = ("dense", "hybrid_bm25_dense")
DENSE_TOLERANCE = 0.002     # 6 questions in 3,000, a few near-ties reordered by a different card
POOLS = ("within_call", "cross_call")


def expected_pairs() -> list[tuple[str, str]]:
    """The 45 (shard, model) pairs of the canonical comparison: 15 quarters by 3 models."""
    out = []
    for year in range(2010, 2025):
        shard = f"{year + 1}Q2"
        for model in (f"ecalls-ret-{year}1231", "bge-small-en-v1.5", f"ecalls-abl-{year}1231"):
            out.append((shard, model))
    return out


DEVIATIONS: list[float] = []


def check(shard: str, model: str) -> list[str]:
    canon = REPORTS / f"cmp_{shard}_{model}_holdout.json"
    rerun = REPORTS / "seeds" / f"cmp_{shard}_{model}_holdout_s0.json"
    if not canon.exists():
        return [f"{canon.name}: canonical file missing"]
    if not rerun.exists():
        return [f"{rerun.name}: not written"]
    a, b = json.loads(canon.read_text()), json.loads(rerun.read_text())
    problems = []
    for key in ("n_queries", "n_candidates", "pool_median"):
        if a[key] != b[key]:
            problems.append(f"{shard} {model}: {key} {a[key]} against {b[key]}")
    for s in EXACT:
        for pool in POOLS:
            if a["scorers"][s][pool] != b["scorers"][s][pool]:
                problems.append(f"{shard} {model}: {s} {pool} differs, "
                                f"{a['scorers'][s][pool]['recall_at_1']} against "
                                f"{b['scorers'][s][pool]['recall_at_1']}")
    for s in NEAR:
        for pool in POOLS:
            for m in ("recall_at_1", "mrr"):
                d = abs(a["scorers"][s][pool][m] - b["scorers"][s][pool][m])
                DEVIATIONS.append(d)
                if d > DENSE_TOLERANCE:
                    problems.append(f"{shard} {model}: {s} {pool} {m} off by {d:.4f}")
    scan = b.get("fusion_scan", {}).get("0.5")
    if scan is None:
        problems.append(f"{shard} {model}: no fusion scan in the rerun")
    elif scan != b["scorers"]["hybrid_bm25_dense"]:
        problems.append(f"{shard} {model}: scan weight 0.5 does not equal the unweighted hybrid")
    return problems


def main() -> int:
    pairs = expected_pairs()
    problems = [p for shard, model in pairs for p in check(shard, model)]
    if problems:
        print(f"{len(problems)} problem(s) across {len(pairs)} pairs:")
        for p in problems[:40]:
            print("  " + p)
        return 1
    print(f"all {len(pairs)} seed-0 reruns reproduce the canonical comparison: lexical scorers "
          f"exactly, dense and hybrid within {max(DEVIATIONS):.4f} (tolerance {DENSE_TOLERANCE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
