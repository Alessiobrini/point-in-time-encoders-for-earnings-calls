"""Fig. 3: the look-ahead premium in retrieval (a) and staleness on a fixed quarter (b).

    python -m ecalls.paper.fig_lookahead

(a) Each premium is a later vintage's recall@1 minus that of the vintage available at the time, on the
same held-out questions, averaged over the 14 quarters 2011Q2 to 2024Q2, with the 95% interval from
resampling calls (`retrieval_lookahead.json`). The chain's same-year premium is also split by the replay
control (`replay_summary.json`) into extra training on text already read and later text.
(b) Every vintage on 2025Q2, a quarter after every cutoff, minus its family's 2024 vintage
(`staleness_summary.json`).
"""
import json

from ecalls.paper import common as C
from ecalls.paper.figsize import set_size

OUT = C.FIG_DIR / "lookahead.pdf"
FLOAT = C.FIG_DIR / "lookahead.tex"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"        # the families' colors in Fig. 2
CAPTION = (r"Look-ahead through the encoder. \textbf{Panel (a)}: the premium of a later vintage over the "
           r"vintage available at the time, in within-call recall@1 on the same held-out questions, averaged "
           r"over the 14 test quarters 2011Q2 to 2024Q2, with 95\% intervals from resampling calls. The same "
           r"year is vintage $Y+1$ on the test quarter of vintage $Y$, and 2024 is the 2024 vintage on every "
           r"quarter. Hollow markers split the chain's same-year premium into extra training (replay minus "
           r"vintage $Y$) and later text (vintage $Y+1$ minus replay), where the replay control continues chained vintage $Y$ on year $Y$ for as many tokens "
           r"as the next segment reads. \textbf{Panel (b)}: the recall@1 "
           r"on 2025Q2, a quarter after every cutoff, of each vintage of our two families, which share their 2010 vintage, with shaded 95\% "
           r"intervals.")


def premia() -> list[tuple[str, dict, str]]:
    la = json.loads((C.SEEDS / "retrieval_lookahead.json").read_text())["families"]
    rp = json.loads((C.SEEDS / "replay_summary.json").read_text())
    return [
        ("Ours, from scratch, same year", la["ecalls-ret"]["adjacent"], BLUE),
        ("Ours, from scratch, 2024", la["ecalls-ret"]["leaky"], BLUE),
        ("ChronoBERT + contrastive fine-tuning, same year", la["ecalls-abl"]["adjacent"], AQUA),
        ("ChronoBERT + contrastive fine-tuning, 2024", la["ecalls-abl"]["leaky"], AQUA),
        ("Ours, chained, same year", la["ecalls-chain-ret"]["adjacent"], ORANGE),
        ("Ours, chained, same year: extra training", rp["compute_rpl_minus_chain"], ORANGE),
        ("Ours, chained, same year: later text", rp["later_text_chain_next_minus_rpl"], ORANGE),
        ("Ours, chained, 2024", la["ecalls-chain-ret"]["leaky"], ORANGE),
    ]


def main() -> int:
    plt = C.style()
    w, h = set_size(C.TEXT_PT, fraction=1.0, subplots=(1, 2))
    fig, (a, b) = plt.subplots(1, 2, figsize=(w, h * 0.66), gridspec_kw={"width_ratios": [1.15, 1]})
    rows = premia()
    for i, (label, e, color) in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = e["ci95"]
        hollow = ": " in label
        a.plot([lo, hi], [y, y], color=color, lw=1.2)
        a.plot([e["mean"]], [y], marker="o", ms=4.5, color=color, mfc="white" if hollow else color, mew=1.2)
    a.axvline(0, color="#555555", lw=0.6)
    a.set_xticks([-0.005, 0.0, 0.005, 0.010])
    a.set_xticklabels(["\u22120.005", "0", "0.005", "0.010"])
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([r[0] for r in rows][::-1])
    a.set_xlabel("Later vintage minus vintage available at the time (recall@1)")
    a.set_title("(a)", loc="left")
    a.grid(axis="x", lw=0.3, color="#dddddd")

    st = json.loads((C.SEEDS / "staleness_summary.json").read_text())["families"]
    for fam, label, color, marker in (("ecalls-ret", "Ours, from scratch", BLUE, "o"),
                                      ("ecalls-chain-ret", "Ours, chained", ORANGE, "s")):
        # Levels, so the two families share one axis: the chain's 2010 member is the from-scratch
        # 2010 retriever, and the 2024 vintages are the only point where the families part.
        pv = st[fam]["per_vintage"]
        years = sorted(int(k) for k in pv)
        mean = [pv[str(k)]["recall_at_1"] for k in years] + [st[fam]["latest_recall_at_1"]]
        lo = [pv[str(k)]["level_ci95"][0] for k in years] + [st[fam]["latest_level_ci95"][0]]
        hi = [pv[str(k)]["level_ci95"][1] for k in years] + [st[fam]["latest_level_ci95"][1]]
        years = years + [2024]
        b.fill_between(years, lo, hi, color=color, alpha=0.12, lw=0)
        b.plot(years, mean, color=color, marker=marker, ms=3, lw=1.2, label=label)
    b.set_xticks([2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024])
    b.set_xlabel("Vintage (training cutoff, end of year)")
    b.set_ylabel("Recall@1 on 2025Q2")
    b.set_title("(b)", loc="left")
    b.legend(frameon=False, loc="upper left")
    b.grid(axis="y", lw=0.3, color="#dddddd")
    fig.tight_layout(pad=0.2, w_pad=1.5)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    FLOAT.write_text("\n".join([r"\begin{figure*}[t]", r"\centering",
                                r"\includegraphics[width=\textwidth]{figures/lookahead.pdf}",
                                r"\caption{" + CAPTION + "}", r"\label{fig:lookahead}", r"\end{figure*}", ""]))
    print(f"wrote {OUT} and {FLOAT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
