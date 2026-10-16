"""Table 3: the training settings of each stage, read from the run records.

    python -m ecalls.paper.table_settings

Each column reads the `_run.json` of every vintage of its stage and refuses to render if the vintages
disagree on a setting, so the table states one value only where one value was used. The optimizer and
the schedule, which the code fixes for every stage (`train/pretrain.py`, `train/contrastive.py`), are
in the caption.
"""
import json

from ecalls.config import paths as P
from ecalls.paper import common as C

OUT = C.TAB_DIR / "settings.tex"
STAGES = {"scratch": "ecalls-mlm-*", "chain": "ecalls-chain-mlm-*", "dapt": "ecalls-dapt-2*",
          "ctr": "ecalls-*"}


def runs(pattern: str) -> list[dict]:
    rs = [json.loads(p.read_text()) for p in sorted(P.run_records_dir().glob(f"{pattern}/_run.json"))]
    if not rs:
        raise FileNotFoundError(f"no run records match {pattern}")
    return rs


def one(rs: list[dict], key: str):
    vals = {json.dumps(r.get(key)) for r in rs}
    if len(vals) != 1:
        raise ValueError(f"vintages disagree on {key}: {sorted(vals)}")
    return rs[0].get(key)


def sci(x: float) -> str:
    m, e = f"{x:.0e}".split("e")
    return rf"${m}\times 10^{{{int(e)}}}$"


def render() -> str:
    sc, ch, da = runs(STAGES["scratch"]), runs(STAGES["chain"]), runs(STAGES["dapt"])
    ct = [r for r in runs(STAGES["ctr"]) if r.get("stage") == "contrastive"]
    if len(ct) < 15:
        raise ValueError(f"only {len(ct)} contrastive run records found")
    if one(sc, "peak_lr") != one(ch, "peak_lr"):
        raise ValueError("the chain's segments and the from-scratch vintages use different peak rates")
    pct = lambda r: f"{round(100 * one(r, 'warmup_steps') / one(r, 'total_steps'))}\\%"
    rows = [
        ("Objective", rf"masking, {round(100 * one(sc, 'mask_ratio'))}\%",
         rf"masking, {round(100 * one(da, 'mask_ratio'))}\%", rf"InfoNCE, $\tau={one(ct, 'temperature')}$"),
        ("Length", rf"{one(sc, 'token_budget') / 1e9:.1f}B tokens", rf"{one(da, 'token_budget') / 1e9:.1f}B tokens",
         f"{one(ct, 'total_steps'):,} steps"),
        ("Batch", f"{one(sc, 'tokens_per_step'):,} tokens", f"{one(da, 'tokens_per_step'):,} tokens",
         rf"{one(ct, 'calls_per_batch')} calls $\times$ up to {one(ct, 'pairs_per_call')} pairs"),
        ("Sequence", f"{one(sc, 'context')} tokens", f"{one(da, 'context')} tokens",
         f"{one(ct, 'max_length_query')} and {one(ct, 'max_length_answer')} tokens"),
        ("Peak rate", sci(one(sc, "peak_lr")), sci(one(da, "peak_lr")), sci(one(ct, "peak_lr"))),
        ("Warmup", pct(sc), pct(da), pct(ct)),
    ]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Training settings. Pretraining covers our from-scratch vintages, and each chained "
             r"segment uses the same settings for one pass over its year. DAPT continues a ChronoBERT "
             r"vintage on our window. Contrastive fine-tuning is the same for every recipe, with questions cut at 256 tokens "
             r"and answers at 512, and $\tau$ is the InfoNCE temperature. Every stage uses AdamW with $\beta=(0.9, 0.98)$, "
             r"$\epsilon=10^{-6}$ and weight decay 0.01, gradients clipped at norm 1, a linear warmup over the "
             r"stated share of steps, and a cosine decay to a tenth of the peak learning rate.}",
             r"\label{tab:settings}",
             r"\begin{tabular}{l l l l}", r"\toprule",
             r"Setting & Pretraining & DAPT & Fine-tuning \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
