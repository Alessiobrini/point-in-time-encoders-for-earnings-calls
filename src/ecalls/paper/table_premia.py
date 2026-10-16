"""Table 5: the look-ahead premium across uses of the vectors, for the 2024 and the same-year vintage.

    python -m ecalls.paper.table_premia

Every cell is a later vintage's score minus that of the vintage available at the time, paired by
quarter. Retrieval: within-call recall@1 on held-out questions (`retrieval_lookahead.json`, intervals
from resampling calls). Nearest neighbors: IC of the mean outcome of the 50 most similar earlier calls
of other firms (`knn_*.json`). Forecasts: the change in quarterly rank IC of text plus price baselines
(`*_results.json` through 2024Q4 for the 2024 vintage, `*_adjacent.json` for the same-year one), with
moving-block bootstrap intervals over quarters. A value in bold has a 95% interval that excludes zero.
The held-out split of the earnings-surprise premium is cited in the text from `sue_holdout_split.json`
through values.tex.
"""
import json

from ecalls.paper import common as C

OUT = C.TAB_DIR / "premia.tex"
FAMILIES = [("ecalls-ret", "Ours, from scratch"), ("ecalls-chain-ret", "Ours, chained"),
            ("ecalls-abl", "ChronoBERT + contrastive fine-tuning"), ("chrono-bert-v1", "ChronoBERT")]
FORECASTS = ("ret1d", "o2c", "ret60", "sue", "av")


def j(name: str) -> dict:
    return json.loads(((C.SEEDS if name.startswith("retrieval") else C.PORTFOLIO) / f"{name}.json").read_text())


def fmt(e: dict | None, digits: int = 4) -> str:
    if not e:
        return "n/a"
    lo, hi = e["ci95"]
    return C.num(e["mean"], digits, signed=True, bold=lo > 0 or hi < 0)


def row(fam: str, kind: str, retr: dict, knn1: dict, knns: dict, res: dict, adj: dict) -> list[str]:
    cells = [fmt(retr.get(fam, {}).get(kind))]
    for k in (knn1, knns):
        cells.append(fmt(k.get(f"{fam}|{kind}|other|k50")))
    for t in FORECASTS:
        if kind == "leaky":
            e = res[t]["premium"].get(f"{fam}+baselines", {}).get("leaky_minus_pit_through_2024")
        else:
            e = adj[t]["premium"].get(f"{fam}+baselines")
        cells.append(fmt(e))
    return cells


def render() -> str:
    retr = j("retrieval_lookahead")["families"]
    knn1, knns = j("knn_ret1d_all")["premium"], j("knn_sue_sp500")["premium"]
    res = {t: j(f"{t}_results") for t in FORECASTS}
    adj = {t: j(f"{t}_adjacent") for t in FORECASTS}
    head = ("", "Retrieval", "Next day", "Surprise", "Next day", "Session after", "Next quarter", "Surprise",
            "Volatility")
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Look-ahead premium across uses of the vectors: a later vintage's score minus that of "
             r"the vintage available at the time, averaged over test quarters. Retrieval is within-call "
             r"recall@1 on held-out questions over the 14 quarters 2011Q2 to 2024Q2, and every other column "
             r"averages the 48 quarters 2013Q1 to 2024Q4. Nearest neighbors is the forecast of Panel B of Table~\ref{tab:forecasts}. Forecasts are the change in "
             r"quarterly rank IC of text plus each target's baselines, for three return targets of Table~\ref{tab:forecasts}, "
             r"the next standardized earnings surprise and the abnormal volatility after the call, both on S\&P 500 calls. In Panel A the later vintage is the 2024 vintage, and in Panel B "
             r"the vintage of the test year itself. The retrieval premium of raw ChronoBERT, which has no contrastive stage, is not computed. Bold values have a 95\% interval that excludes zero, from resampling calls for retrieval and from a moving-block bootstrap over quarters for the other columns.}",
             r"\label{tab:premia}",
             r"\begin{tabular}{l r r r r r r r r}", r"\toprule",
             r" & & \multicolumn{2}{c}{Nearest neighbors} & \multicolumn{5}{c}{Forecast with baselines} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-9}",
             " & ".join(head) + r" \\", r"\midrule"]
    for kind, panel in (("leaky", "Panel A: the 2024 vintage"), ("adjacent", "Panel B: the same-year vintage")):
        lines.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{panel}}}}} \\")
        for fam, label in FAMILIES:
            lines.append(" & ".join([label] + row(fam, kind, retr, knn1, knns, res, adj)) + r" \\")
        if kind == "leaky":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def main() -> int:
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
