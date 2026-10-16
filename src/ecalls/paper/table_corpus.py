"""Table 1: the corpus by window, from `reports/paper/corpus_facts.json` (ecalls.paper.corpus_facts).

    python -m ecalls.paper.table_corpus
"""
import json

from ecalls.paper import common as C

OUT = C.TAB_DIR / "corpus.tex"


ROWS = [("Calls (k)", "calls", 1e3, 1), ("Held out (k)", "held_out_calls", 1e3, 1),
        ("Chunks (M)", "chunks", 1e6, 2), ("Tokens (B)", "tokens", 1e9, 2), ("Pairs (M)", "pairs", 1e6, 2)]


def render(d: dict) -> str:
    """Every cutoff as a column, since there is one vintage per year and the windows grow by a year."""
    years = sorted(d["by_cutoff"])
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.2pt}",
             r"\caption{The corpus by cutoff year: the calls dated on or before 31 December of each cutoff year. Calls "
             r"and chunks include the held-out calls, while the tokens for pretraining and the question-answer pairs "
             r"for contrastive fine-tuning exclude them. Calls are in thousands (k), chunks and pairs in "
             r"millions (M), and tokens in billions (B).}",
             r"\label{tab:corpus}",
             r"\begin{tabular}{l" + " r" * len(years) + "}", r"\toprule",
             " & ".join(["Cutoff", *years]) + r" \\", r"\midrule"]
    for label, key, scale, digits in ROWS:
        cells = [label, *[C.num(d["by_cutoff"][y][key] / scale, digits) for y in years]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def main() -> int:
    d = json.loads((C.P.REPORTS_DIR / "paper" / "corpus_facts.json").read_text())
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(d))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
