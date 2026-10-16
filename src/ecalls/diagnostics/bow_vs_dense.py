"""Does a dense encoder beat bag-of-words at retrieval on this text? Phase 3 step 2, and it gates
whether a stronger retriever is worth buying (see DECISIONS D12).

**The benchmark is built from the corpus's own structure, so it needs no labels.** Inside the Q&A
section an `Analysts` turn is a question and the `Executives` turn that follows it in
`component_order` is its answer. That gives unambiguous ground truth for a retrieval task on finance
text, which is the thing FinMTEB has and no temporal corpus does.

Two pools, because they measure different things and only the first is hard:

- **within-call.** The candidates are every executive answer in the same call. Company, quarter,
  speakers and topic are held constant by construction, so a scorer cannot win on company
  fingerprint. This is the interesting number, and it is the direct answer to the identity confound
  measured in `reports/2026-09-26_identity-confound-diagnostic.md`, where within-company similarity
  was 0.4268 against 0.1741 across companies.
- **cross-call.** The candidates are answers drawn from other calls too, at a fixed pool size. Easier,
  and largely a company-identification task, which is why it is reported beside the first rather than
  instead of it.

The dense side embeds the individual question and answer components rather than reusing the corpus
embedding set. That is not an optimization, it is required: the chunk layer pairs each question with
its answer inside one chunk (`qa-pair-v2`), so a chunk embedding of the query already contains the
answer and would score a perfect result that means nothing.

Run:
    python -m ecalls.diagnostics.bow_vs_dense --shard 2019Q3 --n-calls 300
"""
import argparse
import json
import pathlib

import numpy as np
import polars as pl

from ecalls.chunk import registry as R
from ecalls.config import paths as P
from ecalls.ingest.manifest import git_sha, utc_now

QUESTION_SPEAKER = "Analysts"
ANSWER_SPEAKER = "Executives"
MIN_CHARS = 120        # a "thanks, next question" turn carries no retrievable content
MIN_POOL = 5           # a call with 4 answers makes recall@1 a coin flip rather than a measurement


# --- building the benchmark ---------------------------------------------------------------------

def qa_pairs(shard: str, n_calls: int, seed: int = 0, min_chars: int = MIN_CHARS,
             min_pool: int = MIN_POOL, holdout: bool | None = None
             ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Question components and answer components for a random sample of calls in one quarter.

    Returns (queries, answers). A query carries the `answer_id` of the executive turn that directly
    follows it, and `answers` is the candidate pool with one row per executive turn.

    `holdout` selects the population before sampling. `None` keeps every call, which is how the
    reference numbers in `reports/2026-09-27_bow-vs-dense-retrieval.md` were measured. `True` keeps
    only the held-out calls, which is the only population a model trained on this corpus may be
    scored on. `False` keeps only the trainable ones.

    **A model's score and a reference score have to come from the same population.** bge-small's
    0.5378 was measured with no filter, and comparing a vintage's held-out number against it would
    be comparing two different query sets. Passing `holdout=True` re-measures every scorer in `run`
    on the same calls, which is what makes the comparison mean anything.
    """
    df = pl.read_parquet(
        P.components_shard(shard),
        columns=["keydevid", "companyid", "component_order", "component_id", "section",
                 "speaker_type", "speaker_name", "text", "n_chars"],
    ).filter(pl.col("section") == "qa")

    if holdout is not None:
        from ecalls.train import holdout as HO
        keep_calls = {k for k in df["keydevid"].unique().to_list()
                      if HO.is_holdout(k) == holdout}
        df = df.filter(pl.col("keydevid").is_in(keep_calls))
        if df.height == 0:
            raise ValueError(
                f"{shard}: no call survives the holdout filter (holdout={holdout}). Scoring would "
                "report a number computed on nothing.")

    rng = np.random.default_rng(seed)
    calls = df["keydevid"].unique().to_numpy()
    if len(calls) > n_calls:
        calls = rng.choice(calls, size=n_calls, replace=False)
    df = df.filter(pl.col("keydevid").is_in(calls)).sort(["keydevid", "component_order"])

    # the answer that follows each turn within its own call
    df = df.with_columns(
        next_speaker=pl.col("speaker_type").shift(-1).over("keydevid"),
        next_id=pl.col("component_id").shift(-1).over("keydevid"),
        next_chars=pl.col("n_chars").shift(-1).over("keydevid"),
    )
    queries = df.filter(
        (pl.col("speaker_type") == QUESTION_SPEAKER)
        & (pl.col("next_speaker") == ANSWER_SPEAKER)
        & (pl.col("n_chars") >= min_chars)
        & (pl.col("next_chars") >= min_chars)
    ).select(["keydevid", "companyid", "component_id", "text", "next_id"]).rename(
        {"component_id": "query_id", "next_id": "answer_id"})

    answers = df.filter(
        (pl.col("speaker_type") == ANSWER_SPEAKER) & (pl.col("n_chars") >= min_chars)
    ).select(["keydevid", "companyid", "component_id", "text"]).rename(
        {"component_id": "answer_id"})

    # drop calls whose pool is too small for the metric to mean anything
    keep = set(answers.group_by("keydevid").len()
               .filter(pl.col("len") >= min_pool)["keydevid"].to_list())
    queries = queries.filter(pl.col("keydevid").is_in(keep))
    answers = answers.filter(pl.col("keydevid").is_in(keep))
    # and queries whose own answer survived the length filter, so every query is answerable
    queries = queries.filter(pl.col("answer_id").is_in(set(answers["answer_id"].to_list())))
    return queries, answers


# --- the scorers ---------------------------------------------------------------------------------

def tfidf_matrix(texts: list[str], **kw):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english", **kw)
    return vec, vec.fit_transform(texts)


def bm25_scores(q_tokens: list[list[str]], d_tokens: list[list[str]],
                k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Okapi BM25, implemented here rather than pulled in as a dependency.

    Bag-of-words is the baseline this whole diagnostic is built to take seriously, so it gets the
    scorer that actually wins retrieval benchmarks rather than only TF-IDF cosine.
    """
    from collections import Counter

    N = len(d_tokens)
    dl = np.array([len(d) for d in d_tokens], dtype=np.float64)
    avgdl = dl.mean() if N else 1.0
    df: Counter = Counter()
    tfs = []
    for d in d_tokens:
        c = Counter(d)
        tfs.append(c)
        df.update(c.keys())
    idf = {t: np.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    postings: dict[str, list[tuple[int, int]]] = {}
    for j, c in enumerate(tfs):
        for t, f in c.items():
            postings.setdefault(t, []).append((j, f))

    out = np.zeros((len(q_tokens), N))
    denom_norm = k1 * (1 - b + b * dl / avgdl)
    for i, q in enumerate(q_tokens):
        for t in set(q):
            if t not in postings:
                continue
            w = idf[t]
            for j, f in postings[t]:
                out[i, j] += w * f * (k1 + 1) / (f + denom_norm[j])
    return out


def tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def dense_matrix(queries: list[str], docs: list[str], model_id: str, batch_size: int = 32,
                 device=None) -> tuple[np.ndarray, np.ndarray]:
    """Query and document matrices for a registered model, honoring its pooling and query prefix.

    Queries and documents are embedded separately rather than as one concatenated list, because a
    retriever may be asymmetric: the BGE family is trained with an instruction on the query side only,
    and prefixing the documents too costs retrieval quality. Pooling comes from the registry entry, so
    a CLS-trained retriever is not silently mean-pooled.
    """
    from ecalls.embed import encode as E
    entry = R.load().by_name(model_id)
    tok, model, dev = E.load_encoder(entry, device)
    prefix = (entry.extra or {}).get("query_prefix", "")
    # E5 is trained with a prefix on both sides ("query: ", "passage: "), BGE on the query side only.
    dprefix = (entry.extra or {}).get("passage_prefix", "")
    max_len = E.effective_max_length(entry)
    kw = dict(batch_size=batch_size, max_length=max_len, pooling=entry.pooling)
    Q = E.encode_texts([prefix + t for t in queries], tok, model, dev, **kw)
    D = E.encode_texts([dprefix + t for t in docs], tok, model, dev, **kw)
    return Q, D


# --- the metric ----------------------------------------------------------------------------------

def rank_metrics(scores: np.ndarray, gold_col: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """recall@1, recall@5 and MRR from a query-by-candidate score matrix."""
    return metrics_from_ranks(rank_vector(scores, gold_col, mask))


def metrics_from_ranks(ranks: np.ndarray) -> dict:
    return {"n_queries": int(len(ranks)), "recall_at_1": float((ranks == 1).mean()),
            "recall_at_5": float((ranks <= 5).mean()), "mrr": float((1.0 / ranks).mean()),
            "median_rank": float(np.median(ranks))}


def rank_vector(scores: np.ndarray, gold_col: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """The gold candidate's rank for each query, 1 being best.

    `mask` marks the candidates each query is allowed to see, which is how the within-call pool is
    expressed without building a separate matrix per call. Ties are broken pessimistically: a gold
    document tied with k others ranks below all of them, so a scorer that returns a constant gets the
    worst possible score rather than an accidental win.
    """
    n = len(gold_col)
    rows = np.arange(n)
    if mask is not None:
        hidden = ~mask[rows, gold_col]
        if hidden.any():
            # `ties.sum(1) - 1` below assumes the gold is tied with itself. A masked-out gold breaks
            # that and the rank came back as 1, so an unanswerable query scored a perfect hit. Refuse
            # instead: the pool must contain the answer, and a caller that masks it has a bug.
            raise ValueError(
                f"{int(hidden.sum())} of {n} queries have their gold candidate masked out, so they "
                "are unanswerable. Fix the mask or drop those queries; scoring them would report a "
                "perfect rank for a query whose answer is not in the pool.")
    s = scores.astype(np.float64).copy()
    if mask is not None:
        s[~mask] = -np.inf
    gold = s[rows, gold_col]
    better = (s > gold[:, None])
    if mask is not None:
        better &= mask
    ranks = better.sum(1) + 1
    ties = (s == gold[:, None])
    if mask is not None:
        ties &= mask
    ranks = ranks + (ties.sum(1) - 1)          # pessimistic on ties
    return ranks.astype(np.int32)


def zscore_rows(S: np.ndarray) -> np.ndarray:
    """Standardize each query's score row, so two scorers on different scales can be added.

    BM25 scores are unbounded sums of term weights and cosine similarities live in [-1, 1], so adding
    them raw would be a BM25 ranking with rounding noise. A row whose scores are all equal, which
    happens when a query shares no term with any candidate, is returned as zeros rather than dividing
    by a zero standard deviation.
    """
    S = S.astype(np.float64)
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True)
    return np.divide(S - mu, sd, out=np.zeros_like(S), where=sd > 0)


# Weight on BM25 in the fusion scan. 0 is the dense model alone and 1 is BM25 alone, and 0.5 ranks
# identically to `hybrid_bm25_dense`, the unweighted sum, so the scan reproduces that scorer.
FUSION_WEIGHTS = tuple(round(0.1 * i, 1) for i in range(11))


def fused(bm_z: np.ndarray, dn_z: np.ndarray, w: float) -> np.ndarray:
    """BM25 and dense, each z-scored per query, mixed with weight `w` on BM25."""
    return w * bm_z + (1.0 - w) * dn_z


def within_call_mask(q_call: np.ndarray, a_call: np.ndarray) -> np.ndarray:
    return q_call[:, None] == a_call[None, :]


def run(shard: str, n_calls: int, model_id: str, seed: int = 0, device=None,
        max_queries: int = 3000, holdout: bool | None = None, keep_ranks: bool = False,
        fusion_scan: bool = False) -> dict:
    """One benchmark run. With `keep_ranks`, the per-question ranks and each question's call are
    returned under `_ranks`, because the call, not the question, is the independent unit a bootstrap
    has to resample: the 3,000 questions come from about 400 calls. With `fusion_scan`, the BM25 and
    dense mix is scored at every weight in `FUSION_WEIGHTS`, since the equal-weight sum alone cannot
    say whether fusion hurts a strong retriever or the weight was wrong for it."""
    queries, answers = qa_pairs(shard, n_calls, seed, holdout=holdout)
    if queries.height > max_queries:
        queries = queries.sample(n=max_queries, seed=seed)
    q_text = queries["text"].to_list()
    a_text = answers["text"].to_list()
    a_index = {aid: j for j, aid in enumerate(answers["answer_id"].to_list())}
    gold = np.array([a_index[x] for x in queries["answer_id"].to_list()])
    q_call = queries["keydevid"].to_numpy()
    a_call = answers["keydevid"].to_numpy()
    mask = within_call_mask(q_call, a_call)
    pool = mask.sum(1)

    print(f"{shard}: {queries.height:,} questions, {answers.height:,} candidate answers, "
          f"within-call pool median {np.median(pool):.0f} (min {pool.min()}, max {pool.max()})",
          flush=True)

    out = {"shard": shard, "n_calls_sampled": n_calls, "model_id": model_id, "seed": seed,
           "holdout": holdout, "n_queries": queries.height, "n_candidates": answers.height,
           "pool_median": float(np.median(pool)), "scorers": {}}

    # TF-IDF, fitted on the candidate pool and applied to the queries
    vec, A = tfidf_matrix(a_text)
    Q = vec.transform(q_text)
    tf = (Q @ A.T).toarray()

    # BM25
    bm = bm25_scores([tokenize(t) for t in q_text], [tokenize(t) for t in a_text])

    # dense
    print(f"  embedding questions and answers separately with {model_id}", flush=True)
    Vq, Va = dense_matrix(q_text, a_text, model_id, device=device)
    dn = Vq @ Va.T

    # A hybrid, because "which is better alone" is not the question a retriever design turns on.
    # Dense can lose outright and still carry signal the lexical scorer misses. Both sides are
    # z-scored per query before adding, so the sum is not dominated by whichever has the wider spread.
    bm_z, dn_z = zscore_rows(bm), zscore_rows(dn)
    hy = bm_z + dn_z

    ranks = {}
    for name, S in (("tfidf", tf), ("bm25", bm), ("dense", dn), ("hybrid_bm25_dense", hy)):
        rw, rc = rank_vector(S, gold, mask), rank_vector(S, gold, None)
        ranks[name] = (rw, rc)
        out["scorers"][name] = {"within_call": metrics_from_ranks(rw),
                                "cross_call": metrics_from_ranks(rc)}
        w = out["scorers"][name]["within_call"]
        c = out["scorers"][name]["cross_call"]
        print(f"  {name:6s} within-call R@1 {w['recall_at_1']:.4f} MRR {w['mrr']:.4f} | "
              f"cross-call R@1 {c['recall_at_1']:.4f} MRR {c['mrr']:.4f}", flush=True)

    if fusion_scan:
        out["fusion_scan"] = {}
        for w in FUSION_WEIGHTS:
            S = fused(bm_z, dn_z, w)
            rw, rc = rank_vector(S, gold, mask), rank_vector(S, gold, None)
            ranks[f"fused_w{w}"] = (rw, rc)
            out["fusion_scan"][str(w)] = {"within_call": metrics_from_ranks(rw),
                                          "cross_call": metrics_from_ranks(rc)}
        best = max(out["fusion_scan"], key=lambda k: out["fusion_scan"][k]["within_call"]["recall_at_1"])
        print(f"  fusion scan, within-call R@1 by weight on BM25: " + " ".join(
            f"{k}:{v['within_call']['recall_at_1']:.4f}" for k, v in out["fusion_scan"].items())
              + f"  (best {best})", flush=True)

    out["git_sha"] = git_sha()
    out["ts"] = utc_now()
    if keep_ranks:
        out["_ranks"] = {"keydevid": q_call, "ranks": ranks}
    return out


def save_ranks(res: dict, path: pathlib.Path) -> pathlib.Path:
    """Per-question ranks as a compressed npz beside nothing licensed: call ids and integers only."""
    r = res["_ranks"]
    arrays = {"keydevid": np.asarray(r["keydevid"])}
    for name, (rw, rc) in r["ranks"].items():
        arrays[f"{name}__within"] = rw
        arrays[f"{name}__cross"] = rc
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", default="2019Q3")
    ap.add_argument("--n-calls", type=int, default=300)
    ap.add_argument("--model", default="chrono-bert-v1-20241231")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-queries", type=int, default=3000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ranks-out", default=None,
                    help="also write per-question ranks here, under DATA_ROOT unless absolute")
    ap.add_argument("--fusion-scan", action="store_true",
                    help="score the BM25 and dense mix at every weight in FUSION_WEIGHTS")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--holdout-only", dest="holdout", action="store_const", const=True,
                   help="score only held-out calls, required for any model trained on this corpus")
    g.add_argument("--trainable-only", dest="holdout", action="store_const", const=False)
    ap.set_defaults(holdout=None)
    args = ap.parse_args(argv)

    res = run(args.shard, args.n_calls, args.model, args.seed, args.device, args.max_queries,
              holdout=args.holdout, keep_ranks=bool(args.ranks_out), fusion_scan=args.fusion_scan)
    if args.ranks_out:
        rp = pathlib.Path(args.ranks_out)
        rp = rp if rp.is_absolute() else P.PROCESSED_DIR / rp
        print(f"wrote {save_ranks(res, rp)}")
        res.pop("_ranks")
    tag = {True: "-holdout", False: "-trainable", None: ""}[args.holdout]
    # Under `reports/`, not `data/processed/`. This is a measurement written for people and it is
    # committed, while DATA_ROOT is licensed or regenerable and git-ignored. An absolute --out is
    # honored as given.
    name = args.out or f"bow_vs_dense_{args.shard}_{args.model}{tag}.json"
    out = pathlib.Path(name) if pathlib.Path(name).is_absolute() else P.REPORTS_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {out}")

    best = max(res["scorers"], key=lambda k: res["scorers"][k]["within_call"]["mrr"])
    print(f"best on the within-call pool: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
