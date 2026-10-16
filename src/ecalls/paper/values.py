"""paper/values.tex: every headline number the prose cites, as a macro generated from the result files.

    python -m ecalls.paper.values

Each macro keeps the full-precision value and rounds at display with siunitx, so the text cites
\\ChainSameYear rather than digits and a rerun updates the abstract, body and captions together.
A count is printed whole (round-mode=none), never with decimals.
"""
import json

from ecalls.paper import common as C

OUT = C.PAPER_DIR / "values.tex"


def j(path) -> dict:
    return json.loads(path.read_text())


def val(v: float, places: int) -> str:
    return rf"\num[round-mode=places,round-precision={places}]{{{v!r}}}"


def count(v: int) -> str:
    return rf"\num[round-mode=none,group-separator={{,}}]{{{int(v)}}}"


def macros() -> dict[str, str]:
    s, pf, pp = C.SEEDS, C.PORTFOLIO, C.P.REPORTS_DIR / "paper"
    m: dict[str, str] = {}
    ref, refx = j(s / "references_summary.json"), j(s / "references_summary_cross.json")
    m["RecallOurs"] = val(ref["mean_recall_at_1"]["ours"], 3)
    m["RecallChain"] = val(ref["mean_recall_at_1"]["chain"], 3)
    m["RecallAbl"] = val(ref["mean_recall_at_1"]["abl"], 3)
    m["RecallEfive"] = val(ref["mean_recall_at_1"]["e5-base-v2"], 3)
    m["RecallBgeSmall"] = val(ref["mean_recall_at_1"]["bge-small-en-v1.5"], 3)
    m["RecallFinbert"] = val(ref["mean_recall_at_1"]["finbert-pretrain"], 3)
    m["OursMinusEfive"] = val(ref["differences"]["ours_minus_e5-base-v2"]["mean"], 3)
    m["OursMinusEfiveCross"] = val(refx["differences"]["ours_minus_e5-base-v2"]["mean"], 3)
    # across calls the strongest general retriever is not e5-base, so the margin is also given over it
    general = ("bge-small-en-v1.5", "bge-base-en-v1.5", "bge-large-en-v1.5", "e5-base-v2")
    best = max(general, key=lambda g: refx["mean_recall_at_1"][g])
    assert best == "bge-large-en-v1.5", best   # the text names it
    m["OursMinusBestCross"] = val(refx["differences"][f"ours_minus_{best}"]["mean"], 3)
    from ecalls.paper.table_decomposition import seed_range_vintage
    m["SeedRangeVintage"] = val(seed_range_vintage(s / "training_seed_summary.json"), 3)
    m["CrossOurs"] = val(refx["mean_recall_at_1"]["ours"], 3)
    m["ChanceWithin"] = val(j(pp / "chance_facts.json")["mean_chance_within"], 2)
    m["CrossEfive"] = val(refx["mean_recall_at_1"]["e5-base-v2"], 3)

    fu = j(s / "seed_summary.json")["fusion"]
    for fam, tag in (("ours", "Ours"), ("bge", "BgeSmall")):
        curve = {float(w): r for w, r in fu[fam].items()}
        best = max(curve, key=curve.get)
        m[f"FusionGain{tag}"] = val(curve[best] - curve[0.0], 3)
        m[f"FusionBestW{tag}"] = val(best, 1)
    m["FusionEqualLossOurs"] = val({float(w): r for w, r in fu["ours"].items()}[0.0]
                                   - {float(w): r for w, r in fu["ours"].items()}[0.5], 3)
    la = j(s / "retrieval_lookahead.json")["families"]
    for fam, tag in (("ecalls-ret", "Scratch"), ("ecalls-chain-ret", "Chain"), ("ecalls-abl", "Abl")):
        for kind, k in (("adjacent", "SameYear"), ("leaky", "Late")):
            e = la[fam][kind]
            m[f"{tag}{k}"] = val(e["mean"], 4)
            m[f"{tag}{k}Lo"], m[f"{tag}{k}Hi"] = val(e["ci95"][0], 4), val(e["ci95"][1], 4)
            m[f"{tag}{k}Pos"] = count(e["positive"])
    rp = j(s / "replay_summary.json")
    for key, tag in (("compute_rpl_minus_chain", "ReplayCompute"), ("later_text_chain_next_minus_rpl", "ReplayLater")):
        e = rp[key]
        m[tag], m[tag + "Lo"], m[tag + "Hi"] = val(e["mean"], 4), val(e["ci95"][0], 4), val(e["ci95"][1], 4)
        m[tag + "Wins"] = count(e["wins"])
    bd = j(s / "budget_summary.json")
    m["TwiceGain"] = val(bd["twice_minus_ours2024"]["mean"], 3)
    m["TwiceMinusChain"] = val(bd["twice_minus_chain2024"]["mean"], 4)
    st = j(s / "staleness_summary.json")["families"]
    m["StaleSlopeChain"] = val(-st["ecalls-chain-ret"]["slope_per_year"], 4)
    m["StaleSlopeScratch"] = val(-st["ecalls-ret"]["slope_per_year"], 4)
    m["StaleChainTwentyTen"] = val(-st["ecalls-chain-ret"]["per_vintage"]["2010"]["minus_2024"], 3)
    # the from-scratch 2023 vintage and the twice-budget 2024 vintage on the same quarter, 2025Q2
    m["StaleScratchTwentyThree"] = val(st["ecalls-ret"]["per_vintage"]["2023"]["recall_at_1"], 3)
    tw = j(s / "cmp_2025Q2_ecalls-2x-ret-20241231_holdout_s0.json")["scorers"]["dense"]["within_call"]
    m["StaleTwiceTwentyFour"] = val(tw["recall_at_1"], 3)
    rg = j(s / "rag_summary.json")
    for key, tag in (("chain", "RagChain"), ("scratch", "RagScratch"), ("chain_nogold", "RagNoGold")):
        e = rg["comparisons"][key]
        m[tag], m[tag + "Lo"], m[tag + "Hi"] = val(e["mean"], 4), val(e["ci95"][0], 4), val(e["ci95"][1], 4)
        m[tag + "Pos"] = count(e["positive"])
    m["RagVsClosed"] = val(rg["comparisons"]["retrieval_vs_closed"]["mean"], 2)
    m["RagClosedLevel"] = val(-rg["levels"]["closed"], 2)         # nats per token, sign in the text
    m["RagPitLevel"] = val(-rg["levels"]["chain_pit"], 2)
    # D34: the chained premium by how closely the firm's later calls restate each test answer
    rs = j(s / "restatement_summary.json")["families"]["chain"]
    m["RestateLeakyLow"] = val(rs["leaky"]["low"]["mean"], 4)
    m["RestateLeakyLowLo"], m["RestateLeakyLowHi"] = (val(x, 4) for x in rs["leaky"]["low"]["ci95"])
    m["RestateLeakyGap"] = val(rs["leaky"]["high_minus_low"]["mean"], 4)
    m["RestateLeakyGapLo"], m["RestateLeakyGapHi"] = (val(x, 4) for x in rs["leaky"]["high_minus_low"]["ci95"])
    m["RestateAdjGap"] = val(rs["adjacent"]["high_minus_low"]["mean"], 4)
    m["RestateAdjGapLo"], m["RestateAdjGapHi"] = (val(x, 4) for x in rs["adjacent"]["high_minus_low"]["ci95"])
    m["RagGoldChainPit"] = val(100 * rg["gold_in_top"]["chain"]["pit"], 1)
    m["RagGoldChainLate"] = val(100 * rg["gold_in_top"]["chain"]["late"], 1)

    r1 = j(pf / "ret1d_results.json")
    for fam, tag in (("ecalls-ret", "Ours"), ("ecalls-chain-ret", "Chain"), ("ecalls-abl", "Abl"),
                     ("chrono-bert-v1", "Chrono"), ("tone", "Tone")):
        e = r1["increment"][f"{fam}+baselines minus baselines"]
        m[f"NextDayInc{tag}"], m[f"NextDayInc{tag}T"] = val(e["mean"], 3), val(e["t"], 1)
    ro = j(pf / "o2c_results.json")
    for fam, tag in (("ecalls-ret", "Ours"), ("tone", "Tone"), ("ecalls-abl", "Abl")):
        e = ro["increment"][f"{fam}+baselines minus baselines"]
        m[f"OtcInc{tag}"], m[f"OtcInc{tag}T"] = val(e["mean"], 3), val(e["t"], 1)
    base = r1["specs"]["baselines"]
    m["NextDayBaseIC"], m["NextDayBaseICT"] = val(base["ic_mean"], 3), val(base["ic_t"], 1)
    # The mean-squared-error counterpart of the IC, in percent: out-of-sample R^2 against the training mean.
    m["NextDayRtwoBase"] = val(100 * base["r2_oos"], 3)
    m["NextDayRtwoOurs"] = val(100 * r1["specs"]["ecalls-ret+baselines"]["r2_oos"], 3)
    dls = r1["daily_long_short"]["baselines"]
    m["NextDaySharpeBase"] = val(dls["sharpe_all_days"], 2)
    m["NextDayShareTraded"] = val(100 * dls["share_days_traded"], 0)
    e = r1["premium"]["ecalls-chain-ret+baselines"]["leaky_minus_pit_through_2024"]
    m["ChainNextDayLate"], m["ChainNextDayLateT"] = val(e["mean"], 4), val(e["t"], 1)
    for f, tag in (("knn_ret1d_all", "NextDay"), ("knn_sue_sp500", "Sue")):
        k = j(pf / f"{f}.json")["specs"]["ecalls-ret|pit|other|k50"]
        m[f"Knn{tag}Ours"], m[f"Knn{tag}OursT"] = val(k["ic_mean"], 3), val(k["ic_t"], 1)
    # Every forecast premium, with the baselines and with the text alone, and how many pass a
    # Bonferroni bound over all of them. The family is every premium the forecast files hold, so the
    # text-alone ones the text cites are inside it rather than added beside it.
    from scipy.stats import norm
    cells, block, names = [], [], []
    for t in ("ret1d", "ret5d", "o2c", "ret60", "sue", "av"):
        for name, getter in ((f"{t}_results", lambda v: v.get("leaky_minus_pit_through_2024")),
                             (f"{t}_adjacent", lambda v: v)):
            f = pf / f"{name}.json"
            if not f.exists():
                continue
            for fam, v in j(f).get("premium", {}).items():
                if "identity" in fam or "-pres" in fam or "-qa" in fam:
                    continue
                c = getter(v) if isinstance(v, dict) else None
                if c and "t" in c:
                    cells.append(c["t"])
                    names.append((name, fam))
                    # the t a 4-quarter block bootstrap implies, from its 95% interval
                    block.append(abs(c["mean"]) / ((c["ci95"][1] - c["ci95"][0]) / 3.92))
    bound = float(norm.ppf(1 - 0.025 / len(cells)))
    m["NForecastPremia"] = count(len(cells))
    m["NBonferroni"] = count(sum(abs(x) > bound for x in cells))
    m["NBonferroniBlock"] = count(sum(x > bound for x in block))
    m["BonferroniBound"] = val(bound, 2)
    m["BonferroniBlockMax"] = val(max(block), 2)
    # The text names the premia that pass under the block bootstrap, so it must be these two.
    passing = sorted(names[i] for i, x in enumerate(block) if x > bound)
    assert passing == [("sue_results", "ecalls-chain-ret"), ("sue_results", "ecalls-ret")], passing
    fundamental = all(names[i][0].startswith(("sue", "av")) for i, x in enumerate(cells) if abs(x) > bound)
    assert fundamental, "a premium on a return target passes the bound, so the text is wrong"
    sue = j(pf / "sue_results.json")["premium"]
    m["SueLateChrono"] = val(sue["chrono-bert-v1+baselines"]["leaky_minus_pit_through_2024"]["mean"], 4)
    m["SueLateChronoT"] = val(sue["chrono-bert-v1+baselines"]["leaky_minus_pit_through_2024"]["t"], 1)
    av = j(pf / "av_results.json")["premium"]["ecalls-chain-ret+baselines"]["leaky_minus_pit_through_2024"]
    m["AvLateChain"], m["AvLateChainT"] = val(av["mean"], 4), val(av["t"], 1)
    m["SueLateOurs"] = val(sue["ecalls-ret+baselines"]["leaky_minus_pit_through_2024"]["mean"], 4)
    m["SueLateOursT"] = val(sue["ecalls-ret+baselines"]["leaky_minus_pit_through_2024"]["t"], 1)
    m["SueLateOursAlone"] = val(sue["ecalls-ret"]["leaky_minus_pit_through_2024"]["mean"], 3)
    m["SueLateOursAloneT"] = val(sue["ecalls-ret"]["leaky_minus_pit_through_2024"]["t"], 1)
    # the text-alone surprise premia sit outside the Bonferroni family above; the t their block interval implies
    for fam, tag in (("ecalls-ret", "Ours"), ("ecalls-chain-ret", "Chain")):
        c = sue[fam]["leaky_minus_pit_through_2024"]
        m[f"SueLate{tag}AloneBlockT"] = val(abs(c["mean"]) / ((c["ci95"][1] - c["ci95"][0]) / 3.92), 1)

    hs = j(pf / "sue_holdout_split.json")["split"]
    for fam, tag in (("ecalls-ret", "Ours"), ("ecalls-chain-ret", "Chain")):
        for part, k in (("held_out", "Held"), ("trained", "Trained")):
            m[f"SueAlone{tag}{k}"] = val(hs[fam][part]["mean"], 4)
        m[f"SueAlone{tag}HeldLo"] = val(hs[fam]["held_out"]["ci95"][0], 4)
        m[f"SueAlone{tag}HeldHi"] = val(hs[fam]["held_out"]["ci95"][1], 4)
    cf = j(pp / "corpus_facts.json")
    m["NCallsIndex"] = count(cf["total_calls"])
    m["NPairs"] = val(cf["by_cutoff"]["2024"]["pairs"] / 1e6, 1)
    m["TokensTwentyTen"] = val(cf["by_cutoff"]["2010"]["tokens"] / 1e9, 1)
    mf = j(pp / "model_facts.json")["models"]
    m["ParamsOurs"] = val(mf["ecalls-ret-20241231"]["params"] / 1e6, 1)
    m["ParamsAbl"] = val(mf["ecalls-abl-20241231"]["params"] / 1e6, 1)
    runs = C.P.run_records_dir()
    for name, tag in (("ecalls-mlm-20241231", "PassesStd"), ("ecalls-mlm2x-20241231", "PassesTwice")):
        f = runs / name / "_run.json"
        if f.exists():
            m[tag] = val(j(f)["epochs_equivalent"], 2)

    dec = s / "decomposition_summary.json"
    if dec.exists():
        d = j(dec)
        m["TaskStageGain"] = val(d["task_stage_on_chronobert"]["within"]["mean"], 3)
        m["RecallChrono"] = val(ref["mean_recall_at_1"]["abl"] - d["task_stage_on_chronobert"]["within"]["mean"], 3)
        m["DaptMinusAbl"] = val(d["dapt_minus_abl"]["within"]["mean"], 3)
        m["OursMinusAbl"] = val(-d["ours_minus_abl"]["within"]["mean"], 3)
        m["ChainMinusScratch"] = val(d["chain_minus_ours"]["within"]["mean"], 4)
    return m


def main() -> int:
    m = macros()
    lines = ["% Generated by `python -m ecalls.paper.values`. Do not edit by hand.", ""]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} with {len(m)} macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
