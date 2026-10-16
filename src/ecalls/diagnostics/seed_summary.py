"""Intervals for the retrieval comparison, and the BM25 weight curve, from the evaluation-seed array.

Reads what `cluster/bench_seeds.sbatch` writes: one JSON per (quarter, model, evaluation seed) under
`reports/seeds/` and the per-question ranks under `DATA_ROOT/processed/bench_ranks/`.

**The bootstrap resamples calls, not questions.** The 3,000 questions in a quarter come from about 400
calls, and questions in one call share a company, a quarter and a pool of candidate answers, so they
are not independent. Resampling questions would treat them as if they were and give intervals that are
too narrow. Calls are drawn with replacement inside each quarter, every question of a drawn call comes
with it, and the statistic is recomputed. The three models in a quarter answer the same questions in
the same order, so a difference between two models is paired question by question, which the
resampling preserves by drawing one set of calls for both.

    python -m ecalls.diagnostics.seed_summary [--boot 2000]
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from ecalls.config import paths as P
from ecalls.ingest.manifest import git_sha, utc_now

SEEDS_DIR = P.REPORTS_DIR / "seeds"
RANKS_DIR = P.PROCESSED_DIR / "bench_ranks"
YEARS = range(2010, 2025)
FAMILIES = {"ours": "ecalls-ret-{y}1231", "bge": "bge-small-en-v1.5", "abl": "ecalls-abl-{y}1231"}
DIFFS = (("ours", "bge"), ("abl", "bge"), ("ours", "abl"))


def shard_of(year: int) -> str:
    return f"{year + 1}Q2"


def load_ranks(year: int, family: str, seed: int, scorer: str = "dense") -> tuple[np.ndarray, np.ndarray]:
    model = FAMILIES[family].format(y=year)
    z = np.load(RANKS_DIR / f"{shard_of(year)}_{model}_s{seed}.npz")
    return z["keydevid"], z[f"{scorer}__within"]


def call_bootstrap(groups: list[tuple[np.ndarray, np.ndarray]], n_boot: int = 2000,
                   seed: int = 0) -> np.ndarray:
    """Bootstrap draws of the mean over quarters of a per-question statistic.

    `groups` holds one (call id per question, value per question) pair per quarter. Each draw resamples
    the calls of every quarter with replacement, takes the mean value over the drawn calls' questions,
    then averages the quarters with equal weight, which is how the headline mean is computed.
    """
    rng = np.random.default_rng(seed)
    prepared = []
    for calls, vals in groups:
        uniq, inv = np.unique(calls, return_inverse=True)
        sums = np.bincount(inv, weights=vals, minlength=len(uniq))
        counts = np.bincount(inv, minlength=len(uniq)).astype(float)
        prepared.append((sums, counts))
    out = np.empty(n_boot)
    for b in range(n_boot):
        means = []
        for sums, counts in prepared:
            idx = rng.integers(0, len(sums), len(sums))
            means.append(sums[idx].sum() / counts[idx].sum())
        out[b] = np.mean(means)
    return out


def interval(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def summarize(seeds=(0, 1, 2), n_boot: int = 2000) -> dict:
    out = {"seeds": list(seeds), "n_boot": n_boot, "per_seed": {}, "pooled": {}, "fusion": {}}
    pooled_groups = {d: [] for d in DIFFS}
    for s in seeds:
        hits = {f: [] for f in FAMILIES}
        groups = {d: [] for d in DIFFS}
        for y in YEARS:
            r = {}
            calls = None
            for f in FAMILIES:
                c, rk = load_ranks(y, f, s)
                if calls is None:
                    calls = c
                elif not np.array_equal(calls, c):
                    raise ValueError(f"{shard_of(y)} seed {s}: {f} answered different questions, so "
                                     "the paired difference is undefined")
                r[f] = (rk == 1).astype(float)
                hits[f].append(r[f].mean())
            for a, b in DIFFS:
                groups[(a, b)].append((calls, r[a] - r[b]))
                pooled_groups[(a, b)].append((np.array([f"{s}-{x}" for x in calls]), r[a] - r[b]))
        entry = {"mean_recall_at_1": {f: float(np.mean(v)) for f, v in hits.items()}}
        for a, b in DIFFS:
            diff = np.array(hits[a]) - np.array(hits[b])
            draws = call_bootstrap(groups[(a, b)], n_boot, seed=s)
            entry[f"{a}_minus_{b}"] = {"mean": float(diff.mean()), "sd_over_quarters": float(diff.std(ddof=1)),
                                       "wins": int((diff > 0).sum()), "ci95": interval(draws)}
        out["per_seed"][str(s)] = entry

    # Pooled over evaluation seeds: a call drawn under two seeds is two observations of the same call,
    # so call ids are prefixed by seed only to keep the groups apart; each quarter-seed is one group.
    for a, b in DIFFS:
        draws = call_bootstrap(pooled_groups[(a, b)], n_boot, seed=99)
        means = [out["per_seed"][str(s)][f"{a}_minus_{b}"]["mean"] for s in seeds]
        out["pooled"][f"{a}_minus_{b}"] = {"mean": float(np.mean(means)), "ci95": interval(draws),
                                          "range_over_seeds": [float(min(means)), float(max(means))]}

    # The BM25 weight curve, mean within-call recall@1 over quarters and seeds, per family.
    for f in FAMILIES:
        curve = {}
        for s in seeds:
            for y in YEARS:
                model = FAMILIES[f].format(y=y)
                res = json.loads((SEEDS_DIR / f"cmp_{shard_of(y)}_{model}_holdout_s{s}.json").read_text())
                for w, v in res["fusion_scan"].items():
                    curve.setdefault(w, []).append(v["within_call"]["recall_at_1"])
        out["fusion"][f] = {w: float(np.mean(v)) for w, v in curve.items()}
    out.update(git_sha=git_sha(), ts=utc_now())
    return out


TRAIN_YEARS = (2010, 2017, 2024)
TRAINED = {"ours": {0: "ecalls-ret-{y}1231", 1: "ecalls-seed1-ret-{y}1231", 2: "ecalls-seed2-ret-{y}1231"},
           "abl": {0: "ecalls-abl-{y}1231", 1: "ecalls-seed1-abl-{y}1231", 2: "ecalls-seed2-abl-{y}1231"}}


WITHIN, CROSS = "dense__within", "dense__cross"


def _hits(year: int, model: str, eval_seed: int = 0, key: str = WITHIN) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-question hits at rank 1. `key` picks within-call (the question ranked against its own call's
    answers) or cross-call (against every answer in the quarter's sampled calls)."""
    f = RANKS_DIR / f"{shard_of(year)}_{model}_s{eval_seed}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z["keydevid"], (z[key] == 1).astype(float)


def training_seeds(n_boot: int = 2000) -> dict:
    """The comparison under retrained contrastive stages, on three vintages at evaluation seed 0.

    Every replicate is scored on the same questions as the seed-0 models, so the differences stay
    paired and the evaluation draw is held fixed. Only the training seed varies. A replicate whose ranks
    are not on disk yet is reported as missing rather than replaced by seed 0.
    """
    out = {"years": list(TRAIN_YEARS), "per_training_seed": {}, "per_quarter": {}}
    bge = {y: _hits(y, FAMILIES["bge"]) for y in TRAIN_YEARS}
    for t in (0, 1, 2):
        models = {fam: {y: _hits(y, TRAINED[fam][t].format(y=y)) for y in TRAIN_YEARS} for fam in TRAINED}
        entry = {}
        for fam in TRAINED:
            if any(v is None for v in models[fam].values()):
                entry[fam] = "missing"
                continue
            entry[fam] = {"mean_recall_at_1": float(np.mean([models[fam][y][1].mean() for y in TRAIN_YEARS]))}
            for y in TRAIN_YEARS:
                out["per_quarter"].setdefault(f"{fam}_{y}", {})[str(t)] = float(models[fam][y][1].mean())
        for a, b, A, B in (("ours", "bge", models["ours"], bge), ("abl", "bge", models["abl"], bge),
                           ("ours", "abl", models["ours"], models["abl"])):
            if any(v is None for v in list(A.values()) + list(B.values())):
                continue
            groups = []
            for y in TRAIN_YEARS:
                if not np.array_equal(A[y][0], B[y][0]):
                    raise ValueError(f"{shard_of(y)}: {a} and {b} answered different questions")
                groups.append((A[y][0], A[y][1] - B[y][1]))
            diffs = np.array([g[1].mean() for g in groups])
            entry[f"{a}_minus_{b}"] = {"mean": float(diffs.mean()), "wins": int((diffs > 0).sum()),
                                       "ci95": interval(call_bootstrap(groups, n_boot, seed=t))}
        out["per_training_seed"][str(t)] = entry
    return out


def dapt_summary(n_boot: int = 2000) -> dict:
    """The domain-adaptive retriever against the ablation, ours and bge-small, per available vintage.

    The comparison L15 asks for: DAPT and the ablation share architecture, starting weights and
    contrastive stage, and differ only in whether the finance corpus was seen. Paired on the same
    questions at evaluation seed 0. The ablation's training-seed spread is reported beside it, so a
    difference can be read against what retraining alone moves.
    """
    out = {"per_year": {}}
    for y in TRAIN_YEARS:
        d = _hits(y, f"ecalls-dapt-ret-{y}1231")
        if d is None:
            out["per_year"][str(y)] = "missing"
            continue
        e = {"dapt": float(d[1].mean())}
        for name, model in (("abl", f"ecalls-abl-{y}1231"), ("ours", f"ecalls-ret-{y}1231"),
                            ("bge", FAMILIES["bge"])):
            o = _hits(y, model)
            if not np.array_equal(d[0], o[0]):
                raise ValueError(f"{shard_of(y)}: dapt and {name} answered different questions")
            diff = d[1] - o[1]
            e[name] = float(o[1].mean())
            e[f"dapt_minus_{name}"] = {"mean": float(diff.mean()),
                                       "ci95": interval(call_bootstrap([(d[0], diff)], n_boot, seed=y))}
        seeds = [_hits(y, TRAINED["abl"][t].format(y=y)) for t in (0, 1, 2)]
        if all(x is not None for x in seeds):
            e["abl_training_seed_range"] = [float(min(x[1].mean() for x in seeds)),
                                            float(max(x[1].mean() for x in seeds))]
        out["per_year"][str(y)] = e
    return out


LOOKAHEAD_YEARS = range(2010, 2024)     # quarters 2011Q2 to 2024Q2, the ones with a later vintage
LOOKAHEAD_FAMILIES = ("ecalls-ret", "ecalls-abl", "ecalls-chain-ret")
CHAIN = "ecalls-chain-ret"


def pit_vintage(family: str, year: int) -> str:
    """The point-in-time vintage of a family. The chain's 2010 member is the from-scratch 2010
    retriever, since segment 2011 continues ecalls-mlm-20101231 (D29)."""
    model = "ecalls-ret-20101231" if family == CHAIN and year == 2010 else f"{family}-{year}1231"
    from ecalls.chunk import registry as R
    R.require_point_in_time(model, dt.date(year + 1, 4, 1))   # quarter (Y+1)Q2 starts on 1 April
    return model


def later_vintage(family: str, year: int, kind: str) -> str:
    """The vintage that replaces point-in-time vintage `year`: the 2024 one, or the same-year one."""
    if kind == "leaky":
        return f"{family}-20241231"
    if kind == "adjacent":
        return f"{family}-{year + 1}1231"
    raise ValueError(f"kind is leaky or adjacent, got {kind!r}")


def lookahead_summary(n_boot: int = 2000) -> dict:
    """The look-ahead premium in retrieval: a later vintage against the point-in-time one, per quarter.

    For quarter (Y+1)Q2 the point-in-time retriever is vintage Y. The leaky one is the 2024 vintage and
    the adjacent one is vintage Y+1, the JFE paper's cleanest test. All are scored on the same held-out
    questions at evaluation seed 0, so the difference is paired question by question and the bootstrap
    resamples calls. Held-out calls were never trained on by any vintage, so a later vintage can only
    know the rest of the quarter's calls, the language after it, and what followed.
    """
    out = {"years": list(LOOKAHEAD_YEARS), "families": {}}
    for fam in LOOKAHEAD_FAMILIES:
        fam_out = {}
        for kind in ("leaky", "adjacent"):
            groups, per_q, missing = [], {}, []
            for y in LOOKAHEAD_YEARS:
                pit, later = _hits(y, pit_vintage(fam, y)), _hits(y, later_vintage(fam, y, kind))
                if pit is None or later is None:
                    missing.append(shard_of(y))
                    continue
                if not np.array_equal(pit[0], later[0]):
                    raise ValueError(f"{shard_of(y)}: the two vintages answered different questions")
                diff = later[1] - pit[1]
                groups.append((pit[0], diff))
                per_q[shard_of(y)] = {"pit": float(pit[1].mean()), "later": float(later[1].mean()),
                                      "diff": float(diff.mean())}
            entry = {"per_quarter": per_q, "missing": missing}
            if groups:
                diffs = np.array([g[1].mean() for g in groups])
                entry.update(mean=float(diffs.mean()), positive=int((diffs > 0).sum()),
                             n_quarters=len(groups), ci95=interval(call_bootstrap(groups, n_boot)))
            fam_out[kind] = entry
        out["families"][fam] = fam_out
    return out


SEED_LOOKAHEAD = {"ecalls-ret": ("ecalls-ret", "ecalls-seed1-ret", "ecalls-seed2-ret"),
                  "ecalls-abl": ("ecalls-abl", "ecalls-seed1-abl", "ecalls-seed2-abl")}


def seed_lookahead_summary(n_boot: int = 2000) -> dict:
    """The 2024 look-ahead premium once per training seed of the 2024 vintage (D33).

    The point-in-time vintages stay the seed-0 ones, since each quarter has its own and their noise
    averages over quarters. The 2024 vintage is one model for all 14 quarters, so its draw does not
    average out, and this replaces it with each of its contrastive-stage replicates in turn. Pairing,
    sign and bootstrap are those of `lookahead_summary`.
    """
    out = {"years": list(LOOKAHEAD_YEARS), "families": {}}
    for fam, replicates in SEED_LOOKAHEAD.items():
        fam_out = {}
        for t, rep in enumerate(replicates):
            groups, per_q, missing = [], {}, []
            for y in LOOKAHEAD_YEARS:
                pit, later = _hits(y, pit_vintage(fam, y)), _hits(y, f"{rep}-20241231")
                if pit is None or later is None:
                    missing.append(shard_of(y))
                    continue
                if not np.array_equal(pit[0], later[0]):
                    raise ValueError(f"{shard_of(y)}: the two vintages answered different questions")
                diff = later[1] - pit[1]
                groups.append((pit[0], diff))
                per_q[shard_of(y)] = float(diff.mean())
            entry = {"model": f"{rep}-20241231", "per_quarter": per_q, "missing": missing}
            if groups:
                diffs = np.array([g[1].mean() for g in groups])
                entry.update(mean=float(diffs.mean()), positive=int((diffs > 0).sum()),
                             n_quarters=len(groups), ci95=interval(call_bootstrap(groups, n_boot)))
            fam_out[str(t)] = entry
        out["families"][fam] = fam_out
    return out


def chain_summary(n_boot: int = 2000) -> dict:
    """The chained family against the from-scratch one and the ablation, quarter by quarter (D29).

    Same held-out questions at evaluation seed 0, so every difference is paired question by question.
    The 2010 quarter is left out of the differences against ours, because the two families share that
    model.
    """
    out = {"per_quarter": {}, "missing": []}
    for other in ("ecalls-ret", "ecalls-abl"):
        groups = []
        for y in range(2011, 2025):
            c, o = _hits(y, pit_vintage(CHAIN, y)), _hits(y, f"{other}-{y}1231")
            if c is None or o is None:
                out["missing"].append(f"{shard_of(y)} {other}")
                continue
            if not np.array_equal(c[0], o[0]):
                raise ValueError(f"{shard_of(y)}: chain and {other} answered different questions")
            groups.append((c[0], c[1] - o[1]))
            q = out["per_quarter"].setdefault(shard_of(y), {"chain": float(c[1].mean())})
            q[other] = float(o[1].mean())
        if groups:
            diffs = np.array([g[1].mean() for g in groups])
            out[f"chain_minus_{other}"] = {"mean": float(diffs.mean()), "wins": int((diffs > 0).sum()),
                                           "n_quarters": len(groups),
                                           "ci95": interval(call_bootstrap(groups, n_boot))}
    return out


REFERENCES = ("bge-small-en-v1.5", "bge-base-en-v1.5", "bge-large-en-v1.5", "e5-base-v2", "finbert-pretrain")
# FinBERT's corpus holds Seeking Alpha call transcripts through 2019, which likely include our held-out
# calls, so its comparison is also reported on the quarters from 2020Q2 (vintage years from 2019).
CLEAN_FROM = {"finbert-pretrain": 2019}


def references_summary(n_boot: int = 2000, key: str = WITHIN) -> dict:
    """Ours, the chain and the ablation against each reference retriever, on the 15 quarters at
    evaluation seed 0, paired question by question with calls resampled. `key` selects within-call or
    cross-call recall. A reference in CLEAN_FROM also gets the comparison on its clean quarters."""
    ours = {"ours": lambda y: f"ecalls-ret-{y}1231", "chain": lambda y: pit_vintage(CHAIN, y),
            "abl": lambda y: f"ecalls-abl-{y}1231"}
    out = {"mean_recall_at_1": {}, "differences": {}, "missing": []}
    for ref in REFERENCES:
        hits = {y: _hits(y, ref, key=key) for y in YEARS}
        if any(h is None for h in hits.values()):
            out["missing"].append(ref)
            continue
        spans = {"": list(YEARS)}
        if ref in CLEAN_FROM:
            spans["_clean"] = [y for y in YEARS if y >= CLEAN_FROM[ref]]
        for tag, years in spans.items():
            out["mean_recall_at_1"][ref + tag] = float(np.mean([hits[y][1].mean() for y in years]))
            for name, f in ours.items():
                groups = []
                for y in years:
                    o = _hits(y, f(y), key=key)
                    if o is None:
                        break
                    if not np.array_equal(o[0], hits[y][0]):
                        raise ValueError(f"{shard_of(y)}: {name} and {ref} answered different questions")
                    groups.append((o[0], o[1] - hits[y][1]))
                if len(groups) == len(years):
                    diffs = np.array([g[1].mean() for g in groups])
                    out["differences"][f"{name}_minus_{ref}{tag}"] = {
                        "mean": float(diffs.mean()), "wins": int((diffs > 0).sum()), "n_quarters": len(groups),
                        "ci95": interval(call_bootstrap(groups, n_boot))}
    for name, f in ours.items():
        own = [_hits(y, f(y), key=key) for y in YEARS]
        if all(h is not None for h in own):
            out["mean_recall_at_1"][name] = float(np.mean([h[1].mean() for h in own]))
    return out


TWICE = "ecalls-2x-ret-20241231"


def _paired(pairs: list[tuple[int, str, str]], n_boot: int, key: str = WITHIN) -> dict:
    """Mean over quarters of model a minus model b, paired question by question, calls resampled."""
    groups, per_q, missing = [], {}, []
    for y, a, b in pairs:
        ha, hb = _hits(y, a, key=key), _hits(y, b, key=key)
        if ha is None or hb is None:
            missing.append(shard_of(y))
            continue
        if not np.array_equal(ha[0], hb[0]):
            raise ValueError(f"{shard_of(y)}: {a} and {b} answered different questions")
        groups.append((ha[0], ha[1] - hb[1]))
        per_q[shard_of(y)] = {"a": float(ha[1].mean()), "b": float(hb[1].mean())}
    out = {"per_quarter": per_q, "missing": missing}
    if groups:
        diffs = np.array([g[1].mean() for g in groups])
        out.update(mean=float(diffs.mean()), wins=int((diffs > 0).sum()), n_quarters=len(groups),
                   ci95=interval(call_bootstrap(groups, n_boot)))
    return out


MODELS_DIR = P.PROCESSED_DIR / "models"


def chain_tokens(models_dir: Path | None = None) -> dict[int, int]:
    """Pretraining tokens a chained vintage has seen in total, from the run records: the 2010 base plus
    every segment up to its year. The 2010 entry is the from-scratch base itself."""
    d = models_dir or P.run_records_dir()
    seen = {2010: json.loads((d / "ecalls-mlm-20101231" / "_run.json").read_text())["tokens_seen"]}
    for y in range(2011, 2025):
        seen[y] = seen[y - 1] + json.loads((d / f"ecalls-chain-mlm-{y}1231" / "_run.json").read_text())["tokens_seen"]
    return seen


def compute_prediction(per_doubling: float, tokens: dict[int, int]) -> dict:
    """What the chain's look-ahead premiums would be if recall rose by `per_doubling` per doubling of
    pretraining tokens and nothing else differed. One measured doubling, at 1.2B tokens from scratch,
    sets the rate, so this is a reading of that one point and not a fitted curve."""
    adj = {shard_of(y): per_doubling * float(np.log2(tokens[y + 1] / tokens[y])) for y in LOOKAHEAD_YEARS}
    leaky = {shard_of(y): per_doubling * float(np.log2(tokens[2024] / tokens[y])) for y in LOOKAHEAD_YEARS}
    ratio = [tokens[y + 1] / tokens[y] for y in LOOKAHEAD_YEARS]
    return {"per_doubling": per_doubling, "adjacent": adj, "leaky": leaky,
            "adjacent_mean": float(np.mean(list(adj.values()))), "leaky_mean": float(np.mean(list(leaky.values()))),
            "adjacent_token_ratio": [float(min(ratio)), float(max(ratio))],
            "tokens": {str(y): int(t) for y, t in tokens.items()}}


def budget_summary(n_boot: int = 2000, models_dir: Path | None = None) -> dict:
    """The twice-budget check of D29: the 2024 vintage from scratch at 2.4B tokens against 1.2B.

    The two share the window, the tokenizer and the task stage, so their difference on the 15 quarters
    is what doubling the pretraining compute buys. Set beside it: the chained 2024 vintage, which saw
    about 3.5B tokens, the twice-budget vintage's own look-ahead premium against the point-in-time
    vintages, and the chain's premiums against what its extra tokens alone would predict.
    """
    ours24, chain24 = "ecalls-ret-20241231", "ecalls-chain-ret-20241231"
    out = {
        "twice_minus_ours2024": _paired([(y, TWICE, ours24) for y in YEARS], n_boot),
        "chain2024_minus_ours2024": _paired([(y, chain24, ours24) for y in YEARS], n_boot),
        "twice_minus_chain2024": _paired([(y, TWICE, chain24) for y in YEARS], n_boot),
        "twice_minus_abl2024": _paired([(y, TWICE, "ecalls-abl-20241231") for y in YEARS], n_boot),
        "twice_minus_pit": _paired([(y, TWICE, pit_vintage("ecalls-ret", y)) for y in LOOKAHEAD_YEARS],
                                   n_boot),
        "chain_adjacent": _paired([(y, later_vintage(CHAIN, y, "adjacent"), pit_vintage(CHAIN, y))
                                   for y in LOOKAHEAD_YEARS], n_boot),
        "chain_leaky": _paired([(y, later_vintage(CHAIN, y, "leaky"), pit_vintage(CHAIN, y))
                                for y in LOOKAHEAD_YEARS], n_boot),
    }
    rate = out["twice_minus_ours2024"].get("mean")
    try:
        out["chain_compute_prediction"] = compute_prediction(rate, chain_tokens(models_dir)) if rate is not None else None
    except FileNotFoundError as e:
        out["chain_compute_prediction"] = f"missing run record: {e.filename}"
    return out


REPLAY = "ecalls-rpl-ret"


def replay_summary(n_boot: int = 2000) -> dict:
    """The compute-matched control of D31, on quarter (Y+1)Q2 for Y = 2010 to 2023.

    `rpl-ret-Y` is chained vintage Y continued on its own last year for as many tokens as segment Y+1
    read. Against chained vintage Y it measures what those extra tokens buy at the chain's own scale.
    Chained vintage Y+1 against it measures what later text adds at equal compute, which is the test.
    Every difference is paired question by question, so the two parts add up to the chain's same-year
    premium in every quarter, and the summary refuses to report if they do not.
    """
    def rpl(y):
        return f"{REPLAY}-{y}1231"

    compute = _paired([(y, rpl(y), pit_vintage(CHAIN, y)) for y in LOOKAHEAD_YEARS], n_boot)
    later = _paired([(y, later_vintage(CHAIN, y, "adjacent"), rpl(y)) for y in LOOKAHEAD_YEARS], n_boot)
    total = _paired([(y, later_vintage(CHAIN, y, "adjacent"), pit_vintage(CHAIN, y))
                     for y in LOOKAHEAD_YEARS], n_boot)
    for q, t in total["per_quarter"].items():
        if q in compute["per_quarter"] and q in later["per_quarter"]:
            c, l = compute["per_quarter"][q], later["per_quarter"][q]
            parts = (c["a"] - c["b"]) + (l["a"] - l["b"])
            if not np.isclose(parts, t["a"] - t["b"], atol=1e-12):
                raise ValueError(f"{q}: compute plus later text is {parts}, the premium is {t['a'] - t['b']}")
    return {"compute_rpl_minus_chain": compute, "later_text_chain_next_minus_rpl": later,
            "chain_adjacent": total}


PRETRAIN_SEEDS = (1, 2)


def pretrain_seed_names(seed: int) -> dict:
    """The names of one pretraining-seed replicate of the chain (D34), by role.

    Seed 0 is the chain the paper reports, whose 2010 member is the from-scratch 2010 retriever. A
    replicate reseeds that base too, so its 2010 member is `ecalls-pseed<S>-ret-20101231`, and every
    other name carries the same prefix. A missed prefix would pair a replicate with the seed-0 chain,
    so the names are built in one place.
    """
    if seed == 0:
        return {"pit": lambda y: pit_vintage(CHAIN, y), "chain": lambda y: f"{CHAIN}-{y}1231",
                "rpl": lambda y: f"{REPLAY}-{y}1231"}
    p = f"ecalls-pseed{seed}-"

    def pit(y):
        model = f"{p}ret-20101231" if y == 2010 else f"{p}chain-ret-{y}1231"
        from ecalls.chunk import registry as R
        R.require_point_in_time(model, dt.date(y + 1, 4, 1))
        return model
    return {"pit": pit, "chain": lambda y: f"{p}chain-ret-{y}1231", "rpl": lambda y: f"{p}rpl-ret-{y}1231"}


def pretrain_seed_summary(n_boot: int = 2000, seeds=(0, *PRETRAIN_SEEDS)) -> dict:
    """The chain's look-ahead premia and replay split once per pretraining seed (D34).

    The contrastive-seed check of D33 holds pretraining fixed. Here the whole chain is retrained from a
    reseeded base, so each seed has its own point-in-time vintages, same-year and 2024 vintages and
    replays, and every difference pairs models of one seed, question by question, as in
    `lookahead_summary` and `replay_summary`. The replay identity is checked per quarter.
    """
    out = {"years": list(LOOKAHEAD_YEARS), "seeds": {}}
    for s in seeds:
        nm = pretrain_seed_names(s)
        adjacent = _paired([(y, nm["chain"](y + 1), nm["pit"](y)) for y in LOOKAHEAD_YEARS], n_boot)
        leaky = _paired([(y, nm["chain"](2024), nm["pit"](y)) for y in LOOKAHEAD_YEARS], n_boot)
        compute = _paired([(y, nm["rpl"](y), nm["pit"](y)) for y in LOOKAHEAD_YEARS], n_boot)
        later = _paired([(y, nm["chain"](y + 1), nm["rpl"](y)) for y in LOOKAHEAD_YEARS], n_boot)
        for q, t in adjacent["per_quarter"].items():
            if q in compute["per_quarter"] and q in later["per_quarter"]:
                c, l = compute["per_quarter"][q], later["per_quarter"][q]
                if not np.isclose((c["a"] - c["b"]) + (l["a"] - l["b"]), t["a"] - t["b"], atol=1e-12):
                    raise ValueError(f"seed {s}, {q}: the replay split does not add up to the premium")
        out["seeds"][str(s)] = {"adjacent": adjacent, "leaky": leaky,
                                "compute_rpl_minus_chain": compute,
                                "later_text_chain_next_minus_rpl": later}
    return out


STALENESS_QUARTER_YEAR = 2024          # quarter 2025Q2, after every cutoff


def staleness_summary(n_boot: int = 2000) -> dict:
    """Every vintage on one quarter after all cutoffs, against its family's 2024 vintage.

    PALM's question for retrievers: on a fixed window, does an older vintage score worse? Each
    vintage Y is paired question by question with the same family's 2024 vintage on 2025Q2, so the
    difference is what Y's staleness (2024 minus Y years) costs. The slope of that difference on
    staleness is fitted by least squares over the vintages.
    """
    y0 = STALENESS_QUARTER_YEAR
    out = {"quarter": shard_of(y0), "families": {}}
    for fam in ("ecalls-ret", CHAIN):
        latest = _hits(y0, f"{fam}-20241231")
        rows, missing = {}, []
        for y in LOOKAHEAD_YEARS:
            h = _hits(y0, pit_vintage(fam, y))
            if h is None or latest is None:
                missing.append(y)
                continue
            if not np.array_equal(h[0], latest[0]):
                raise ValueError(f"{fam} {y}: answered different questions from the 2024 vintage")
            draws = call_bootstrap([(h[0], h[1] - latest[1])], n_boot)
            rows[str(y)] = {"recall_at_1": float(h[1].mean()), "minus_2024": float((h[1] - latest[1]).mean()),
                            "ci95": interval(draws), "staleness": 2024 - y}
        entry = {"per_vintage": rows, "missing": missing,
                 "latest_recall_at_1": float(latest[1].mean()) if latest is not None else None}
        if len(rows) >= 2:
            x = np.array([v["staleness"] for v in rows.values()], dtype=float)
            d = np.array([v["minus_2024"] for v in rows.values()])
            entry["slope_per_year"] = float(np.polyfit(x, d, 1)[0])
        out["families"][fam] = entry
    return out


DAPT_YEARS = (2010, 2017, 2024)
CLEAN_YEARS = [y for y in YEARS if y >= 2019]


def decomposition_summary(n_boot: int = 2000) -> dict:
    """Table 3: each step between the references and our retrievers as one paired difference, within
    call and across calls, on the quarters where both models were scored at evaluation seed 0."""
    rows = {
        "ours_minus_e5": [(y, f"ecalls-ret-{y}1231", "e5-base-v2") for y in YEARS],
        "ours_minus_finbert_clean": [(y, f"ecalls-ret-{y}1231", "finbert-pretrain") for y in CLEAN_YEARS],
        "task_stage_on_chronobert": [(y, f"ecalls-abl-{y}1231", f"chrono-bert-v1-{y}1231") for y in YEARS],
        "dapt_minus_abl": [(y, f"ecalls-dapt-ret-{y}1231", f"ecalls-abl-{y}1231") for y in DAPT_YEARS],
        "ours_minus_abl": [(y, f"ecalls-ret-{y}1231", f"ecalls-abl-{y}1231") for y in YEARS],
        "chain_minus_ours": [(y, pit_vintage(CHAIN, y), f"ecalls-ret-{y}1231") for y in YEARS if y > 2010],
        "twice_minus_ours2024": [(y, TWICE, "ecalls-ret-20241231") for y in YEARS],
    }
    out = {}
    for name, pairs in rows.items():
        out[name] = {}
        for scope, key in (("within", WITHIN), ("cross", CROSS)):
            r = _paired(pairs, n_boot, key=key)
            out[name][scope] = {k: v for k, v in r.items() if k != "per_quarter"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="seeds/seed_summary.json")
    ap.add_argument("--dapt", action="store_true", help="summarize the domain-adaptive retrievers")
    ap.add_argument("--training", action="store_true",
                    help="summarize the training-seed replicates instead of the evaluation seeds")
    ap.add_argument("--references", action="store_true", help="ours, the chain and the ablation against general retrievers")
    ap.add_argument("--cross", action="store_true", help="with --references: cross-call recall instead")
    ap.add_argument("--decomposition", action="store_true", help="the paired differences of Table 3")
    ap.add_argument("--chain", action="store_true", help="the chained family against ours and the ablation")
    ap.add_argument("--lookahead", action="store_true",
                    help="summarize the 2024 and same-year vintages against the point-in-time one")
    ap.add_argument("--budget", action="store_true", help="the twice-budget 2024 vintage (D29)")
    ap.add_argument("--replay", action="store_true", help="the compute-matched replay control (D31)")
    ap.add_argument("--staleness", action="store_true", help="every vintage on 2025Q2 against the 2024 one")
    ap.add_argument("--seed-lookahead", action="store_true",
                    help="the 2024 premium per training seed of the 2024 vintage (D33)")
    ap.add_argument("--pretrain-seeds", action="store_true",
                    help="the chain's premia and replay split per pretraining seed (D34)")
    args = ap.parse_args(argv)
    if args.pretrain_seeds:
        r = {**pretrain_seed_summary(args.boot), "git_sha": git_sha(), "ts": utc_now()}
        path = P.REPORTS_DIR / "seeds" / "pretrain_seed_summary.json"
        path.write_text(json.dumps(r, indent=2) + "\n")
        for s, e in r["seeds"].items():
            for k, v in e.items():
                if "mean" in v:
                    print(f"seed {s} {k:<34} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
                          f"wins {v['wins']}/{v['n_quarters']}  missing {v['missing']}")
                else:
                    print(f"seed {s} {k:<34} missing {v['missing']}")
        print(f"wrote {path}")
        return 0
    if args.seed_lookahead:
        r = {**seed_lookahead_summary(args.boot), "git_sha": git_sha(), "ts": utc_now()}
        path = P.REPORTS_DIR / "seeds" / "seed_lookahead_summary.json"
        path.write_text(json.dumps(r, indent=2) + "\n")
        for fam, by_seed in r["families"].items():
            for t, e in by_seed.items():
                if "mean" in e:
                    print(f"{fam:<11} seed {t}  {e['mean']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] "
                          f"positive {e['positive']}/{e['n_quarters']}  missing {e['missing']}")
                else:
                    print(f"{fam:<11} seed {t}  missing {e['missing']}")
        print(f"wrote {path}")
        return 0
    if args.decomposition:
        r = {**decomposition_summary(args.boot), "git_sha": git_sha(), "ts": utc_now()}
        path = P.REPORTS_DIR / "seeds" / "decomposition_summary.json"
        path.write_text(json.dumps(r, indent=2) + "\n")
        for k, v in r.items():
            if isinstance(v, dict):
                for scope, e in v.items():
                    if "mean" in e:
                        print(f"{k:<26} {scope:<6} {e['mean']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] "
                              f"wins {e['wins']}/{e['n_quarters']}  missing {e['missing']}")
                    else:
                        print(f"{k:<26} {scope:<6} missing {e['missing']}")
        print(f"wrote {path}")
        return 0
    if args.staleness:
        r = {**staleness_summary(args.boot), "git_sha": git_sha(), "ts": utc_now()}
        path = P.REPORTS_DIR / "seeds" / "staleness_summary.json"
        path.write_text(json.dumps(r, indent=2) + "\n")
        for fam, e in r["families"].items():
            print(f"{fam}: 2024 vintage {e['latest_recall_at_1']}, slope per year of staleness "
                  f"{e.get('slope_per_year')}, missing {e['missing']}")
            for y, v in e["per_vintage"].items():
                print(f"   {y}  {v['recall_at_1']:.4f}  minus 2024 {v['minus_2024']:+.4f} "
                      f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
        print(f"wrote {path}")
        return 0
    if args.replay:
        r = {**replay_summary(args.boot), "git_sha": git_sha(), "ts": utc_now()}
        path = P.REPORTS_DIR / "seeds" / "replay_summary.json"
        path.write_text(json.dumps(r, indent=2) + "\n")
        for k, v in r.items():
            if isinstance(v, dict) and "per_quarter" in v:
                if "mean" in v:
                    print(f"{k:<34} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
                          f"wins {v['wins']}/{v['n_quarters']}  missing {v['missing']}")
                else:
                    print(f"{k}: missing {v['missing']}")
        print(f"wrote {path}")
        return 0
    if args.budget:
        r = {**budget_summary(args.boot), "git_sha": git_sha(), "ts": utc_now()}
        path = P.REPORTS_DIR / "seeds" / "budget_summary.json"
        path.write_text(json.dumps(r, indent=2) + "\n")
        for k, v in r.items():
            if not isinstance(v, dict) or "per_quarter" not in v:
                continue
            if "mean" in v:
                print(f"{k:<22} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
                      f"wins {v['wins']}/{v['n_quarters']}  missing {v['missing']}")
            else:
                print(f"{k}: missing {v['missing']}")
        c = r["chain_compute_prediction"]
        if isinstance(c, dict):
            print(f"chain premium from tokens alone at {c['per_doubling']:+.4f} per doubling: "
                  f"adjacent {c['adjacent_mean']:+.4f}, leaky {c['leaky_mean']:+.4f}")
        print(f"wrote {path}")
        return 0
    if args.references:
        key = CROSS if args.cross else WITHIN
        r = {**references_summary(args.boot, key=key), "key": key, "git_sha": git_sha(), "ts": utc_now()}
        path = P.REPORTS_DIR / "seeds" / ("references_summary_cross.json" if args.cross else "references_summary.json")
        path.write_text(json.dumps(r, indent=2) + "\n")
        for k, v in r["mean_recall_at_1"].items():
            print(f"{k:<20} recall@1 {v:.4f}")
        for k, v in r["differences"].items():
            print(f"{k:<32} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] wins {v['wins']}/{v['n_quarters']}")
        print(f"missing: {r['missing']}")
        return 0
    if args.chain:
        r = {**chain_summary(args.boot), "git_sha": git_sha(), "ts": utc_now()}
        path = P.REPORTS_DIR / "seeds" / "chain_summary.json"
        path.write_text(json.dumps(r, indent=2) + "\n")
        for k in ("chain_minus_ecalls-ret", "chain_minus_ecalls-abl"):
            if k in r:
                v = r[k]
                print(f"{k:<26} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
                      f"wins {v['wins']}/{v['n_quarters']}")
        print(f"missing: {r['missing']}")
        print(f"wrote {path}")
        return 0
    if args.lookahead:
        r = {**lookahead_summary(args.boot), "git_sha": git_sha(), "ts": utc_now()}
        path = P.REPORTS_DIR / "seeds" / "retrieval_lookahead.json"
        path.write_text(json.dumps(r, indent=2) + "\n")
        for fam, e in r["families"].items():
            for kind, v in e.items():
                if "mean" not in v:
                    print(f"{fam} {kind}: missing {v['missing']}")
                    continue
                print(f"{fam:<11} {kind:<9} later minus point in time {v['mean']:+.4f} "
                      f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  positive {v['positive']}/{v['n_quarters']}"
                      f"  missing {v['missing']}")
        print(f"wrote {path}")
        return 0
    if args.dapt:
        r = {**dapt_summary(args.boot), "git_sha": git_sha(), "ts": utc_now()}
        path = P.REPORTS_DIR / "seeds" / "dapt_summary.json"
        path.write_text(json.dumps(r, indent=2) + "\n")
        for y, e in r["per_year"].items():
            if e == "missing":
                print(f"{y}: missing")
                continue
            print(f"{y}: dapt {e['dapt']:.4f}  abl {e['abl']:.4f}  ours {e['ours']:.4f}  bge {e['bge']:.4f}"
                  f"  abl seeds {e.get('abl_training_seed_range')}")
            for k in ("abl", "ours", "bge"):
                v = e[f"dapt_minus_{k}"]
                print(f"   dapt - {k} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
        print(f"wrote {path}")
        return 0
    if args.training:
        r = {**training_seeds(args.boot), "git_sha": git_sha(), "ts": utc_now()}
        path = P.REPORTS_DIR / "seeds" / "training_seed_summary.json"
        path.write_text(json.dumps(r, indent=2) + "\n")
        for t, e in r["per_training_seed"].items():
            print(f"training seed {t}: " + "  ".join(
                f"{k} {v['mean_recall_at_1']:.4f}" if isinstance(v, dict) and "mean_recall_at_1" in v
                else f"{k} {v}" for k, v in e.items() if k in TRAINED))
            for k, v in e.items():
                if "_minus_" in k:
                    print(f"   {k} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] wins {v['wins']}/3")
        print(f"wrote {path}")
        return 0
    r = summarize(n_boot=args.boot)
    path = P.REPORTS_DIR / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=2) + "\n")
    for s, e in r["per_seed"].items():
        print(f"seed {s}: " + "  ".join(f"{k} {v:.4f}" for k, v in e["mean_recall_at_1"].items()))
        for a, b in DIFFS:
            d = e[f"{a}_minus_{b}"]
            print(f"   {a}-{b} {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
                  f"wins {d['wins']}/15")
    for k, v in r["pooled"].items():
        print(f"pooled {k} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    for f, c in r["fusion"].items():
        best = max(c, key=c.get)
        print(f"fusion {f}: " + " ".join(f"{w}:{x:.4f}" for w, x in c.items()) + f"  best {best}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
