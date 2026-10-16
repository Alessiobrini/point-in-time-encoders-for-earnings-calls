"""Does the look-ahead premium come from the test firm restating its answers in later calls? (D34)

    python -m ecalls.diagnostics.restatement [--boot 2000]

A later vintage never reads the held-out test call, but it reads the same firm's later calls, which
can restate or update the answer to a test question. That is a direct leak of the answer. The rest of
the premium would come from later text in general. The reviewer of 2026-10-14 asked which it is.

Almost every test firm has later calls in a later vintage's window (98% at the 2024 vintage), so
splitting by whether it has any leaves an empty group. This splits instead by how closely the firm
restates each test answer: the largest TF-IDF cosine between the answer and any executive turn of the
firm's next `K_LATER` trainable calls inside the later vintage's window. Questions are cut into terciles
of that score within each quarter, and the premium of the later vintage over the point-in-time one is
computed per tercile, paired question by question, with calls resampled as everywhere else in the paper.

The questions are rebuilt with the benchmark's own sampling (`bow_vs_dense.qa_pairs`, 400 held-out
calls, seed 0, then 3,000 questions at seed 0) and checked call by call against the saved ranks, so a
score is attached to the question it was computed for. Only scores are written, never text.
"""
import argparse
import datetime as dt
import json

import numpy as np
import polars as pl

from ecalls.config import paths as P
from ecalls.diagnostics import bow_vs_dense as BD
from ecalls.diagnostics import seed_summary as S
from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.train import holdout as HO

K_LATER = 4
FAMILIES = {"chain": "ecalls-chain-ret", "scratch": "ecalls-ret"}


def later_calls(test: pl.DataFrame, calls: pl.DataFrame, end: dt.date, k: int = K_LATER) -> pl.DataFrame:
    """For each test call, the same firm's next `k` calls that a vintage with window end `end` read.

    `test` has keydevid, companyid and call_date, `calls` the same plus `holdout`. A later call counts
    when it is dated strictly after the test call and on or before `end`, and is not held out, since no
    vintage reads a held-out call. Returns one row per (test keydevid, later keydevid).
    """
    later = (calls.filter(~pl.col("holdout") & (pl.col("call_date") <= end))
             .select(pl.col("companyid"), pl.col("keydevid").alias("later"), pl.col("call_date").alias("later_date")))
    return (test.join(later, on="companyid")
            .filter(pl.col("later_date") > pl.col("call_date"))
            .sort(["keydevid", "later_date", "later"])
            .group_by("keydevid", maintain_order=True).head(k)
            .select("keydevid", "later"))


def max_cosine(answers: list[str], answer_call: np.ndarray, later_text: list[str], later_owner: np.ndarray) -> np.ndarray:
    """The largest TF-IDF cosine between each answer and any later text owned by the answer's call.

    `later_owner[j]` is the test call whose firm wrote later text j. An answer with no later text scores 0.
    """
    vec, _ = BD.tfidf_matrix(answers + later_text)
    A, L = vec.transform(answers), vec.transform(later_text)
    out = np.zeros(len(answers))
    if not later_text:
        return out
    for call in np.unique(answer_call):
        ai, li = np.flatnonzero(answer_call == call), np.flatnonzero(later_owner == call)
        if li.size:
            out[ai] = (A[ai] @ L[li].T).max(axis=1).toarray().ravel()
    return out


def terciles(score: np.ndarray) -> np.ndarray:
    """0, 1 or 2 by tercile of `score`, ties broken by position so the groups have equal size."""
    order = np.argsort(score, kind="stable")
    t = np.empty(len(score), dtype=int)
    t[order] = np.minimum(3 * np.arange(len(score)) // len(score), 2)
    return t


def quarter_scores(y: int, calls: pl.DataFrame, components_for) -> dict:
    """The questions of quarter (Y+1)Q2 with their restatement scores for both later vintages."""
    shard = S.shard_of(y)
    queries, answers = BD.qa_pairs(shard, 400, 0, holdout=True)
    if queries.height > 3000:
        queries = queries.sample(n=3000, seed=0)
    gold = answers.select(pl.col("answer_id"), pl.col("text").alias("answer_text"))
    q = (queries.select("keydevid", "answer_id").with_row_index("i")
         .join(gold, on="answer_id", how="left").sort("i")
         .join(calls.select("keydevid", "companyid", "call_date"), on="keydevid", how="left"))
    test = q.select("keydevid", "companyid", "call_date").unique()
    out = {"keydevid": q["keydevid"].to_numpy()}
    for kind, end in (("adjacent", dt.date(y + 1, 12, 31)), ("leaky", dt.date(2024, 12, 31))):
        pairs = later_calls(test, calls, end)
        comps = components_for(pairs["later"].unique().to_list())
        lt = pairs.join(comps, left_on="later", right_on="keydevid")
        out[kind] = max_cosine(q["answer_text"].to_list(), q["keydevid"].to_numpy(),
                               lt["text"].to_list(), lt["keydevid"].to_numpy())
        out[f"{kind}_share_with_later"] = float(np.isin(q["keydevid"].to_numpy(), pairs["keydevid"].to_numpy()).mean())
    return out


def components_loader(calls: pl.DataFrame):
    """Executive turns of the given calls, read only from the shards that hold them."""
    shard_of = dict(zip(calls["keydevid"].to_list(), calls["shard"].to_list()))

    def load(keys: list[int]) -> pl.DataFrame:
        frames = []
        by_shard: dict[str, list[int]] = {}
        for k in keys:
            by_shard.setdefault(shard_of[k], []).append(k)
        for shard, ks in sorted(by_shard.items()):
            frames.append(pl.read_parquet(P.components_shard(shard), columns=["keydevid", "speaker_type", "text"])
                          .filter(pl.col("keydevid").is_in(ks) & (pl.col("speaker_type") == "Executives"))
                          .select("keydevid", "text"))
        return pl.concat(frames) if frames else pl.DataFrame({"keydevid": [], "text": []},
                                                              schema={"keydevid": pl.Int64, "text": pl.Utf8})
    return load


def summarize(scores: dict[int, dict], n_boot: int = 2000) -> dict:
    """The premium per tercile of restatement, for each family and later vintage."""
    out = {"k_later": K_LATER, "families": {}}
    for fam, prefix in FAMILIES.items():
        fam_out = {}
        for kind in ("adjacent", "leaky"):
            per_t = {0: [], 1: [], 2: []}
            gap = []
            for y, sc in scores.items():
                pit_model = S.pit_vintage(prefix, y)
                pit, later = S._hits(y, pit_model), S._hits(y, S.later_vintage(prefix, y, kind))
                if pit is None or later is None:
                    continue
                if not (np.array_equal(pit[0], sc["keydevid"]) and np.array_equal(later[0], sc["keydevid"])):
                    raise ValueError(f"{S.shard_of(y)}: the rebuilt questions are not the scored ones")
                diff, t = later[1] - pit[1], terciles(sc[kind])
                for g in per_t:
                    per_t[g].append((sc["keydevid"][t == g], diff[t == g]))
                gap.append((sc["keydevid"], diff * ((t == 2).astype(float) * 3 - (t == 0).astype(float) * 3)))
            entry = {}
            for g, groups in per_t.items():
                if groups:
                    draws = S.call_bootstrap(groups, n_boot)
                    entry[["low", "middle", "high"][g]] = {
                        "mean": float(np.mean([d.mean() for _, d in groups])),
                        "ci95": S.interval(draws), "n_quarters": len(groups)}
            if gap:
                draws = S.call_bootstrap(gap, n_boot)
                entry["high_minus_low"] = {"mean": float(np.mean([d.mean() for _, d in gap])),
                                           "ci95": S.interval(draws)}
            fam_out[kind] = entry
        out["families"][fam] = fam_out
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    calls = (pl.read_parquet(P.PROCESSED_DIR / "calls.parquet", columns=["keydevid", "companyid", "call_date", "shard"])
             .with_columns(pl.col("call_date").dt.date(),
                           pl.col("keydevid").map_elements(HO.is_holdout, return_dtype=pl.Boolean).alias("holdout")))
    load = components_loader(calls)
    scores, shares = {}, {}
    for y in S.LOOKAHEAD_YEARS:
        scores[y] = quarter_scores(y, calls, load)
        shares[S.shard_of(y)] = {k: scores[y][f"{k}_share_with_later"] for k in ("adjacent", "leaky")}
        print(f"{S.shard_of(y)}: share with later calls {shares[S.shard_of(y)]}", flush=True)
    r = {**summarize(scores, args.boot), "share_with_later_calls": shares, "git_sha": git_sha(), "ts": utc_now()}
    path = P.REPORTS_DIR / "seeds" / "restatement_summary.json"
    path.write_text(json.dumps(r, indent=2) + "\n")
    for fam, e in r["families"].items():
        for kind, v in e.items():
            print(fam, kind, {g: (round(x["mean"], 4), [round(c, 4) for c in x["ci95"]]) for g, x in v.items()})
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
