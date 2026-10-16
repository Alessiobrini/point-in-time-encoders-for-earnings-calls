"""Fig. 2: recall@1 by test quarter, within-call (a) and cross-call (b), for the point-in-time vintage.

    python -m ecalls.paper.fig_recall

Each point is one quarter's held-out benchmark at evaluation seed 0: vintage Y scored on quarter
(Y+1)Q2. Values come from the per-quarter benchmark files under reports/seeds/. BM25 does not depend
on the encoder, so it is read from our vintage's file after checking that every file of the quarter
gives the same value. Our families are in color and the references in gray, with a marker each.
"""
import json

from ecalls.diagnostics import seed_summary as S
from ecalls.paper import common as C
from ecalls.paper.figsize import set_size

OUT = C.FIG_DIR / "recall_by_quarter.pdf"
FLOAT = C.FIG_DIR / "recall_by_quarter.tex"
CAPTION = (r"Recall@1 on held-out calls by test quarter, for the vintage available at the time, so vintage $Y$ "
           r"is scored on the second quarter of year $Y+1$. Each quarter has 3,000 analyst questions from 400 sampled "
           r"calls. Panel (a): each question ranked against the executive answers in its own call. Panel (b): each question ranked "
           r"against every answer of the sampled calls. Colored lines are retrievers with our contrastive fine-tuning, "
           r"gray lines are references, and diamonds mark the three DAPT vintages (2010, 2017, 2024). "
           r"FinBERT's corpus includes call transcripts through 2019. Every retriever is scored on the same "
           r"random sample of calls and questions. bge-base, raw ChronoBERT and bge-large, the strongest general retriever across calls, are left out for legibility.")
YEARS = list(S.YEARS)                         # vintage years 2010 to 2024, quarters 2011Q2 to 2025Q2
# Categorical slots 1 to 4 of the dataviz reference palette, validated for adjacent pairs (light mode).
OURS = [("Ours, from scratch", lambda y: f"ecalls-ret-{y}1231", "#2a78d6", "o"),
        ("Ours, chained", lambda y: S.pit_vintage(S.CHAIN, y), "#eb6834", "s"),
        ("ChronoBERT + contrastive fine-tuning", lambda y: f"ecalls-abl-{y}1231", "#1baf7a", "^")]
DAPT = ("ChronoBERT + DAPT", lambda y: f"ecalls-dapt-ret-{y}1231", "#eda100", "D", (2010, 2017, 2024))
REFS = [("e5-base", lambda y: "e5-base-v2", "#6b6b6b", "v", "-"),
        ("bge-small", lambda y: "bge-small-en-v1.5", "#9a9a9a", "<", "--"),
        ("BM25", None, "#4a4a4a", "x", ":"),
        ("FinBERT", lambda y: "finbert-pretrain", "#b5b5b5", ">", "-.")]
KEYS = {"within": "within_call", "cross": "cross_call"}


def recall(model: str, year: int, scope: str, scorer: str = "dense") -> float | None:
    f = C.SEEDS / f"cmp_{S.shard_of(year)}_{model}_holdout_s0.json"
    if not f.exists():
        return None
    return float(json.loads(f.read_text())["scorers"][scorer][KEYS[scope]]["recall_at_1"])


def bm25(year: int, scope: str) -> float:
    vals = {m: recall(m, year, scope, "bm25") for m in (f"ecalls-ret-{year}1231", "e5-base-v2", "bge-small-en-v1.5")}
    got = {round(v, 10) for v in vals.values() if v is not None}
    if len(got) != 1:
        raise ValueError(f"{S.shard_of(year)}: BM25 differs across files {vals}, so the questions differ")
    return got.pop()


def series(scope: str) -> dict[str, list[float | None]]:
    out = {name: [recall(f(y), y, scope) for y in YEARS] for name, f, *_ in OURS}
    out[DAPT[0]] = [recall(DAPT[1](y), y, scope) if y in DAPT[4] else None for y in YEARS]
    for name, f, *_ in REFS:
        out[name] = [bm25(y, scope) if f is None else recall(f(y), y, scope) for y in YEARS]
    return out


def main() -> int:
    plt = C.style()
    w, h = set_size(C.COLUMN_PT, subplots=(2, 1))
    fig, axes = plt.subplots(2, 1, figsize=(w, h * 0.70), sharex=True)
    x = [y + 1 for y in YEARS]                 # the test quarter's year
    for ax, scope, panel in zip(axes, ("within", "cross"), ("(a)", "(b)")):
        s = series(scope)
        for name, _, color, marker, ls in REFS:
            ax.plot(x, s[name], color=color, marker=marker, ls=ls, ms=3, lw=0.9, label=name)
        for name, _, color, marker in OURS:
            ax.plot(x, s[name], color=color, marker=marker, ms=3, lw=1.2, label=name)
        pts = [(xi, v) for xi, v in zip(x, s[DAPT[0]]) if v is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="none", color=DAPT[2], marker=DAPT[3], ms=4,
                label=DAPT[0])
        ax.set_ylabel("Recall@1")
        ax.set_title(panel, loc="left")
        ax.grid(axis="y", lw=0.3, color="#dddddd")
    axes[-1].set_xlabel("Test quarter (Q2 of the year shown)")
    axes[-1].set_xticks(x[::2])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=3.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.215, 1, 1), h_pad=0.6, pad=0.5)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)                           # no tight bbox, so the PDF keeps the column width
    FLOAT.write_text("\n".join([r"\begin{figure}[t]", r"\centering",
                                  r"\includegraphics[width=\columnwidth]{figures/recall_by_quarter.pdf}",
                                  r"\caption{" + CAPTION + "}", r"\label{fig:recall}", r"\end{figure}", ""]))
    print(f"wrote {OUT} and {FLOAT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
