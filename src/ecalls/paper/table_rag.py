"""Table 6: look-ahead through the retriever inside an LLM pipeline (D32).

    python -m ecalls.paper.table_rag

Reads `reports/seeds/rag_summary.json` (`python -m ecalls.diagnostics.rag_lookahead summary`). Each
premium is the 2024 retriever's arm minus the point-in-time retriever's arm, per question, with the
same reader in both, averaged over quarters, with a 95% interval from resampling calls.
"""
import json

from ecalls.paper import common as C

OUT = C.TAB_DIR / "rag.tex"
ROWS = [("chain", "chain", "Ours, chained"),
        ("scratch", "scratch", "Ours, from scratch"),
        ("chain_nogold", None, "Ours, chained, own answer removed")]


def excludes_zero(ci) -> bool:
    return ci[0] > 0 or ci[1] < 0


def render(d: dict) -> str:
    lines = [r"\begin{table}[!tb]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{1.5pt}",
             r"\caption{Look-ahead through the retriever, for a language model that answers from what the retriever returns. For 1,000 held-out "
             r"analyst questions in each of the 14 quarters 2011Q2 to 2024Q2, a retriever returns the 3 "
             r"closest answers from the quarter's sampled held-out calls, and ChronoGPT-Instruct of the "
             r"cutoff year of the vintage available at the time reads them with the question. The score is "
             r"its mean log-probability per token, in nats, of the executive's real answer. Own answer is the share of "
             r"questions whose real answer is among the 3 excerpts, under the vintage available at the "
             r"time, vintage $Y$, and the 2024 vintage. The premium is the 2024 retriever's score minus that of the "
             r"vintage available at the time, with 95\% intervals from "
             r"resampling calls. The last row is the same test with each question's real answer removed from the candidates. "
             r"Pos.\ counts the quarters with a positive premium, and bold premia have an interval that excludes zero.}",
             r"\label{tab:rag}",
             r"\begin{tabular}{>{\raggedright\arraybackslash}p{19.5mm} r r r c r}", r"\toprule",
             r" & \multicolumn{2}{c}{Own answer (\%)} & & & \\", r"\cmidrule(lr){2-3}",
             r"Family & Vintage $Y$ & 2024 & Premium & 95\% interval & Pos. \\", r"\midrule"]
    for key, gold_key, label in ROWS:
        e = d["comparisons"][key]
        lo, hi = e["ci95"]
        if gold_key:
            g = d["gold_in_top"][gold_key]
            shares = [C.num(100 * g["pit"], 1), C.num(100 * g["late"], 1)]
        else:
            shares = ["n/a", "n/a"]
        cells = [label, *shares, C.num(e["mean"], 4, signed=True, bold=excludes_zero(e["ci95"])),
                 f"[{C.num(lo, 4, signed=True)}, {C.num(hi, 4, signed=True)}]",
                 f"{e['positive']}/{e['n_quarters']}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    d = json.loads((C.SEEDS / "rag_summary.json").read_text())
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(d))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
