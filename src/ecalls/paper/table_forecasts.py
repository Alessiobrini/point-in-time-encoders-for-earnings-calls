"""Table 4: what call text adds to return forecasts, by horizon, and the nearest-neighbor forecast.

    python -m ecalls.paper.table_forecasts

Panel A reads `reports/portfolio/{ret1d,ret5d,o2c,ret60}_results.json` (D28 and its amendment): the
baselines' quarterly rank IC, then each text source's increment over the baselines (text plus
baselines minus baselines, paired by quarter, t from the quarterly differences), the decile long-short
Sharpe ratio with baselines and text, and the next-day IC of the text alone. Panel B reads the D30
nearest-neighbor files: the mean outcome of the 50 most similar earlier calls of other firms, point in
time, next-day return on the full universe and earnings surprise on the S&P 500 panel.
"""
import json

from ecalls.paper import common as C

OUT = C.TAB_DIR / "forecasts.tex"
HORIZONS = ("ret1d", "ret5d", "o2c", "ret60")
SOURCES = [("tone", "Dictionary tone"), ("tfidf", "TF-IDF"), ("ecalls-ret", "Ours, from scratch"),
           ("ecalls-chain-ret", "Ours, chained"), ("ecalls-abl", "ChronoBERT + contrastive fine-tuning"),
           ("chrono-bert-v1", "ChronoBERT")]   # the encoders in the order of Panel B and Table VI
KNN = [("ecalls-ret", "Ours, from scratch"), ("ecalls-chain-ret", "Ours, chained"),
       ("ecalls-abl", "ChronoBERT + contrastive fine-tuning"), ("chrono-bert-v1", "ChronoBERT")]


def load(t: str) -> dict:
    return json.loads((C.PORTFOLIO / f"{t}_results.json").read_text())


def ic_t(mean: float, t: float, signed: bool = True, bold: bool = False) -> str:
    return f"{C.num(mean, 3, signed=signed, bold=bold)} ({C.num(t, 1)})"


def best_rows(col: list[float | None]) -> set[int]:
    """Rows holding the column's largest value at the precision printed, so a tie bolds both."""
    vals = [round(v, 3) for v in col if v is not None]
    return {i for i, v in enumerate(col) if v is not None and round(v, 3) == max(vals)} if vals else set()


def sharpe_value(r: dict, spec: str) -> float | None:
    d = r.get("daily_long_short", {}).get(spec)
    return d["sharpe_all_days"] if d else None


def panel_a() -> list[str]:
    """Bold marks the largest value in each column: among the text rows for an increment or the IC of
    the text alone, and among every row, baselines included, for a Sharpe ratio."""
    rs = {t: load(t) for t in HORIZONS}
    inc = [[rs[t]["increment"][f"{key}+baselines minus baselines"] for t in HORIZONS] for key, _ in SOURCES]
    alone = [rs["ret1d"]["specs"][key] for key, _ in SOURCES]
    sharpes = [[sharpe_value(rs["ret1d"], s), sharpe_value(rs["o2c"], s)]
               for s in ["baselines"] + [f"{key}+baselines" for key, _ in SOURCES]]
    best_inc = [best_rows([row[j]["mean"] for row in inc]) for j in range(len(HORIZONS))]
    best_sh = [best_rows([row[j] for row in sharpes]) for j in range(2)]
    best_alone = best_rows([a["ic_mean"] for a in alone])

    def sh(i: int, j: int) -> str:
        v = sharpes[i][j]
        return C.num(v, 2, bold=i in best_sh[j]) if v is not None else ""

    base = [ic_t(rs[t]["specs"]["baselines"]["ic_mean"], rs[t]["specs"]["baselines"]["ic_t"], signed=False)
            for t in HORIZONS]
    lines = [" & ".join(["Price baselines, IC"] + base + [sh(0, 0), sh(0, 1), ""]) + r" \\",
             r"\addlinespace", r"\multicolumn{8}{l}{\emph{Text added to the baselines: increment in IC}} \\"]
    for i, (_, label) in enumerate(SOURCES):
        cells = [ic_t(e["mean"], e["t"], bold=i in best_inc[j]) for j, e in enumerate(inc[i])]
        cells += [sh(i + 1, 0), sh(i + 1, 1),
                  ic_t(alone[i]["ic_mean"], alone[i]["ic_t"], signed=False, bold=i in best_alone)]
        lines.append(" & ".join([label] + cells) + r" \\")
    return lines


def panel_b() -> list[str]:
    nd = json.loads((C.PORTFOLIO / "knn_ret1d_all.json").read_text())["specs"]
    su = json.loads((C.PORTFOLIO / "knn_sue_sp500.json").read_text())["specs"]
    rows = [(label, nd[f"{key}|pit|other|k50"], su[f"{key}|pit|other|k50"]) for key, label in KNN]
    best = [best_rows([r[j]["ic_mean"] for r in rows]) for j in (1, 2)]
    return [f"{label} & {ic_t(a['ic_mean'], a['ic_t'], False, i in best[0])} & "
            f"{ic_t(b['ic_mean'], b['ic_t'], False, i in best[1])} " + r"& & & & & \\"
            for i, (label, a, b) in enumerate(rows)]


def render() -> str:
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Call text in forecasts of returns and earnings surprises, with every vector from the vintage available at the time. "
             r"\textbf{Panel A}: quarterly rank information coefficient (IC) of a ridge forecast refit each January on the price baselines "
             r"and one text source, with the $t$ statistic over the test quarters from 2013 to 2025 in parentheses. The first row is the IC "
             r"of the baselines alone, and each other row is the increment in IC when that text source is added. Next day and 5 days are "
             r"the returns after entry at the first close at least an hour after the call starts, and next quarter is the return over "
             r"trading days $+2$ to $+61$, each minus the market's. Session after is the raw open-to-close return of the first session "
             r"after a call held before 08:30 or from 16:00 New York time. Sharpe ratios are annualized levels before trading costs, for "
             r"the baselines alone in the first row and with text in the others. They come from an equal-weighted decile long-short of "
             r"the calls entering each day, at the close or, for the session after, at the open, held one day on days with at least 20 "
             r"such calls. A day without a position counts as zero return. Text alone is the next-day IC without the baselines. "
             r"\textbf{Panel B}: IC of the mean outcome of the 50 most similar earlier calls of other firms, for the next-day return and, "
             r"on S\&P 500 calls, the next standardized earnings surprise. The largest value in each column is in bold, among the text "
             r"rows for an IC and among all rows for a Sharpe ratio.}",
             r"\label{tab:forecasts}",
             r"\begin{tabular}{l r r r r r r r}", r"\toprule",
             r" & \multicolumn{4}{c}{IC, or increment in IC ($t$)} & \multicolumn{2}{c}{Sharpe ratio} & Text alone \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
             r"Panel A & Next day & 5 days & Session after & Next quarter & Next day & Session after & Next day \\",
             r"\midrule"]
    lines += panel_a()
    lines += [r"\midrule", r"Panel B, IC & Next day & Surprise & & & & & \\", r"\midrule"]
    lines += panel_b()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def main() -> int:
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
