"""Table 3: the steps between the references and our retrievers, each a paired difference in recall@1.

    python -m ecalls.paper.table_decomposition

Reads `reports/seeds/decomposition_summary.json` (seed_summary --decomposition) and, for the training
seed row, `training_seed_summary.json`. Each difference is model a minus model b on the same held-out
questions at evaluation seed 0, averaged over the quarters both were scored on, with a 95% interval
from resampling calls within each quarter.
"""
import json

from ecalls.paper import common as C

OUT = C.TAB_DIR / "decomposition.tex"
ROWS = [
    ("ours_minus_e5", "Ours, from scratch, minus e5-base"),
    ("ours_minus_finbert_clean", "Ours, from scratch, minus FinBERT"),
    ("task_stage_on_chronobert", "ChronoBERT + contrastive fine-tuning minus ChronoBERT"),
    ("dapt_minus_abl", "ChronoBERT + DAPT minus ChronoBERT + contrastive fine-tuning"),
    ("ours_minus_abl", "Ours, from scratch, minus ChronoBERT + contrastive fine-tuning"),
    ("chain_minus_ours", "Ours, chained, minus ours, from scratch"),
    ("twice_minus_ours2024", "Ours, twice the tokens, minus ours, from scratch"),
]


def seed_range(path) -> float:
    r = json.loads(path.read_text())["per_training_seed"]
    vals = [r[s]["ours"]["mean_recall_at_1"] for s in sorted(r)]
    return max(vals) - min(vals)


def seed_range_vintage(path) -> float:
    """The largest range across training seeds of one vintage's recall, over the vintages reseeded."""
    r = json.loads(path.read_text())["per_quarter"]
    return max(max(v.values()) - min(v.values()) for k, v in r.items() if k.startswith("ours_"))


def render(d: dict, seeds: float, seeds_vintage: float) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
             r"\caption{Paired differences in recall@1 between retrievers, on the same held-out questions, "
             r"averaged over the $Q$ test quarters both were scored on. Within is within-call "
             r"and Across is cross-call recall@1. The 95\% interval of the within-call difference comes from "
             r"resampling calls within each quarter. Each retriever is the vintage available at the time, and ChronoBERT has no contrastive fine-tuning. "
             r"FinBERT is compared from 2020Q2, which its corpus cannot contain, DAPT on the 2010, 2017 and 2024 vintages, and the chain from its 2011 vintage. "
             r"The two token budgets compare 2024 vintages on every quarter. The last two rows have no $Q$. They give the range across three seeds of contrastive fine-tuning, "
             r"with pretraining fixed, of our from-scratch recall averaged over the 2010, 2017 and 2024 vintages, and the largest range of one vintage.}",
             r"\label{tab:decomposition}",
             r"\begin{tabular}{>{\raggedright\arraybackslash}p{31mm} r r c r}", r"\toprule",
             r"Difference & $Q$ & Within & 95\% interval & Across \\", r"\midrule"]
    for key, label in ROWS:
        w, x = d[key]["within"], d[key]["cross"]
        lo, hi = w["ci95"]
        cells = [label, str(w["n_quarters"]), C.num(w["mean"], 3, signed=True),
                 f"[{C.num(lo, 3, signed=True)}, {C.num(hi, 3, signed=True)}]", C.num(x["mean"], 3, signed=True)]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(f"Training seed, range & n/a & {C.num(seeds, 3)} & & " + r"\\")
    lines.append(f"Training seed, range per vintage & n/a & {C.num(seeds_vintage, 3)} & & " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    d = json.loads((C.SEEDS / "decomposition_summary.json").read_text())
    f = C.SEEDS / "training_seed_summary.json"
    tex = render(d, seed_range(f), seed_range_vintage(f))
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
