"""Look-ahead through the retriever inside an LLM pipeline, with a point-in-time reader (D32).

    python -m ecalls.diagnostics.rag_lookahead score --year 2015 [--n-questions 1000] [--limit N]
    python -m ecalls.diagnostics.rag_lookahead summary

For quarter (Y+1)Q2, the benchmark's held-out questions are answered from the top 3 answers of the
quarter's 400 calls, retrieved once with the vintage available at the time and once with the 2024
vintage. ChronoGPT-Instruct of year Y reads the question and the three excerpts, and the score is its
mean log-probability per token of the executive's real answer. The premium is the 2024 arm minus the
point-in-time arm, per question, so the reader and the question are the same in both and only the
retriever differs. Outputs hold identifiers and scores only. Excerpts and answers stay where the
corpus is.
"""
import argparse
import json
import os

import numpy as np

from ecalls.config import paths as P
from ecalls.ingest.manifest import git_sha, utc_now

K = 3
N_QUESTIONS = 1000
QUESTION_CAP, EXCERPT_CAP, ANSWER_CAP = 200, 300, 300
YEARS = range(2010, 2024)
FAMILIES = {"chain": "ecalls-chain-ret", "scratch": "ecalls-ret"}
LATE = 2024
OUT_DIR = P.PROCESSED_DIR / "rag_lookahead"
SUMMARY = P.REPORTS_DIR / "seeds" / "rag_summary.json"
LLM_CACHE = os.environ.get("ECALLS_LLM_CACHE", ".llm-cache")


def shard_of(year: int) -> str:
    return f"{year + 1}Q2"


def vintage(family: str, year: int) -> str:
    """The point-in-time retriever of a family for cutoff `year`. The chain's 2010 member is the
    from-scratch 2010 retriever (D29), the same rule as `seed_summary.pit_vintage`."""
    model = "ecalls-ret-20101231" if family == "chain" and year == 2010 else f"{FAMILIES[family]}-{year}1231"
    from ecalls.chunk import registry as R
    R.require_point_in_time(model, f"{year + 1}-04-01")   # quarter (Y+1)Q2 starts on 1 April
    return model


def reader(year: int) -> str:
    """The reader of quarter (Y+1)Q2: ChronoGPT-Instruct with cutoff 31 December Y, the same cutoff as
    the point-in-time retriever, so the reader adds no later text of its own."""
    return f"manelalab/chrono-gpt-instruct-v1-{year}1231"


def top_k(Q: np.ndarray, D: np.ndarray, k: int = K, exclude: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k highest dot products per row, best first. `exclude` gives one column per row
    to leave out (the question's own answer in the secondary arm), or -1 for none."""
    S = Q @ D.T
    if exclude is not None:
        rows = np.flatnonzero(exclude >= 0)
        S[rows, exclude[rows]] = -np.inf
    idx = np.argpartition(-S, kth=min(k, S.shape[1] - 1), axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(S, idx, axis=1), axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


def prompt_text(question: str, excerpts: list[str]) -> str:
    """The reader's input. Excerpts arrive already cut to EXCERPT_CAP tokens."""
    if not excerpts:
        return f"An analyst asks on an earnings call: {question}\nAnswer as the executive."
    body = "\n".join(f"[{i + 1}] {e}" for i, e in enumerate(excerpts))
    return (f"Excerpts from earnings calls:\n{body}\n\nAn analyst asks: {question}\n"
            f"Answer as the executive.")


def cut(tok, text: str, n: int) -> str:
    """The first n tokens of a text, decoded back, so every cap is in the reader's own tokens."""
    ids = tok.encode(text)
    return text if len(ids) <= n else tok.decode(ids[:n])


def pair_ids(tok, question: str, excerpts: list[str], answer_ids: list[int]) -> tuple[list[int], int]:
    """The prompt and answer as one id sequence, and the prompt length."""
    from ecalls.diagnostics.chronogpt import template
    p = tok.encode(template(prompt_text(question, excerpts)))
    return p + answer_ids, len(p)


def answer_logprob(logits, ids, prompt_len: list[int], ans_len: list[int]) -> np.ndarray:
    """Mean log-probability per answer token, by teacher forcing, for a right-padded batch.

    The token at position t is predicted by the logits at t-1, so the answer of row b, at positions
    prompt_len[b] to prompt_len[b] + ans_len[b] - 1, is scored by the logits one position earlier.
    Padding sits after each answer and a causal model never sees it from inside the answer.
    """
    import torch
    out = np.empty(len(prompt_len))
    for b, (p, a) in enumerate(zip(prompt_len, ans_len)):
        tgt = ids[b, p:p + a]
        # the softmax over the answer positions only, since the full (B, T, vocab) one does not fit
        lp = torch.log_softmax(logits[b, p - 1:p + a - 1], dim=-1)
        out[b] = lp.gather(-1, tgt[:, None]).mean().item()
    return out


def score_sequences(model, seqs: list[tuple[list[int], int, int]], device: str,
                    token_budget: int = 6144) -> np.ndarray:
    """Scores for (ids, prompt_len, answer_len) triples, batched by length under a token budget."""
    import torch
    order = np.argsort([len(s[0]) for s in seqs])
    out = np.empty(len(seqs))
    i = 0
    while i < len(order):
        T = len(seqs[order[i]][0])
        j = i
        while j < len(order) and (j - i + 1) * max(T, len(seqs[order[j]][0])) <= token_budget:
            T = max(T, len(seqs[order[j]][0]))
            j += 1
        j = max(j, i + 1)
        batch = [seqs[o] for o in order[i:j]]
        T = max(len(s[0]) for s in batch)
        ids = torch.full((len(batch), T), 0, dtype=torch.long)
        for b, s in enumerate(batch):
            ids[b, :len(s[0])] = torch.tensor(s[0])
        ids = ids.to(device)
        logits = model(ids)
        vals = answer_logprob(logits, ids, [s[1] for s in batch], [s[2] for s in batch])
        out[order[i:j]] = vals
        i = j
    return out


def score_year(year: int, n_questions: int = N_QUESTIONS, limit: int | None = None,
               device: str = "cuda") -> dict:
    import time

    import torch
    from transformers import GPT2TokenizerFast

    from ecalls.diagnostics import bow_vs_dense as B
    from ecalls.diagnostics import chronogpt as CG

    shard = shard_of(year)
    queries, answers = B.qa_pairs(shard, 400, 0, holdout=True)
    if queries.height > 3000:
        queries = queries.sample(n=3000, seed=0)
    queries = queries.sample(n=min(n_questions, queries.height), seed=0)
    if limit:
        queries = queries.head(limit)
    q_text, a_text = queries["text"].to_list(), answers["text"].to_list()
    a_ids = answers["answer_id"].to_numpy()
    a_index = {int(a): j for j, a in enumerate(a_ids)}
    gold = np.array([a_index[int(x)] for x in queries["answer_id"].to_list()])

    # retrieval: two vintages per family, top 3 from the cross-call pool
    arms = {}
    for fam in FAMILIES:
        for kind, model_id in (("pit", vintage(fam, year)), ("late", f"{FAMILIES[fam]}-{LATE}1231")):
            Vq, Va = B.dense_matrix(q_text, a_text, model_id, device=device)
            arms[f"{fam}_{kind}"] = top_k(Vq, Va)
            if fam == "chain":
                arms[f"{fam}_{kind}_nogold"] = top_k(Vq, Va, exclude=gold.copy())
            torch.cuda.empty_cache()

    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2", cache_dir=LLM_CACHE)
    t0 = time.time()
    model = CG.load(reader(year), LLM_CACHE, device)
    print(f"{shard}: reader {reader(year)} loaded in {time.time() - t0:.0f}s", flush=True)

    qs = [cut(tok, t, QUESTION_CAP) for t in q_text]
    ans = [tok.encode(t)[:ANSWER_CAP] for t in (a_text[g] for g in gold)]
    ex_cache: dict[int, str] = {}

    def excerpt(j: int) -> str:
        if j not in ex_cache:
            ex_cache[j] = cut(tok, a_text[j], EXCERPT_CAP)
        return ex_cache[j]

    # one sequence per distinct (question, excerpt list); identical contexts get identical scores
    keys, seqs = {}, []
    def key_of(i: int, ctx: tuple) -> int:
        k = (i, ctx)
        if k not in keys:
            ids, p = pair_ids(tok, qs[i], [excerpt(j) for j in ctx], ans[i])
            if len(ids) > CG.CONTEXT:
                raise ValueError(f"{shard} question {i}: {len(ids)} tokens exceed the context")
            keys[k] = len(seqs)
            seqs.append((ids, p, len(ans[i])))
        return keys[k]

    n = len(qs)
    ref = {"closed": np.array([key_of(i, ()) for i in range(n)])}
    for name, idx in arms.items():
        ref[name] = np.array([key_of(i, tuple(int(j) for j in idx[i])) for i in range(n)])
    print(f"{shard}: {n} questions, {len(seqs)} distinct sequences, "
          f"max length {max(len(s[0]) for s in seqs)}", flush=True)

    t0 = time.time()
    scores = score_sequences(model, seqs, device)
    secs = time.time() - t0
    toks = sum(len(s[0]) for s in seqs)
    print(f"{shard}: scored in {secs:.0f}s, {toks / secs:,.0f} tokens/s", flush=True)
    if not np.isfinite(scores).all():
        raise ValueError(f"{shard}: non-finite scores")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {"keydevid": queries["keydevid"].to_numpy(), "query_id": queries["query_id"].to_numpy(),
              "gold_answer_id": a_ids[gold]}
    for name, r in ref.items():
        arrays[f"{name}__logprob"] = scores[r]
    for name, idx in arms.items():
        arrays[f"{name}__top"] = a_ids[idx]
        arrays[f"{name}__gold_in_top"] = (idx == gold[:, None]).any(1)
    path = OUT_DIR / f"{shard}{'_limit' + str(limit) if limit else ''}.npz"
    np.savez(path, **arrays)
    meta = {"shard": shard, "reader": reader(year), "n_questions": n, "n_requested": n_questions,
            "n_sequences": len(seqs),
            "seconds": secs, "tokens_per_second": toks / secs, "device": torch.cuda.get_device_name(0)
            if device == "cuda" else device, "git_sha": git_sha(), "utc": utc_now()}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {path}", flush=True)
    return meta


def premium(groups: list[tuple[np.ndarray, np.ndarray]], n_boot: int = 2000) -> dict:
    from ecalls.diagnostics import seed_summary as S
    diffs = np.array([g[1].mean() for g in groups])
    return {"mean": float(diffs.mean()), "positive": int((diffs > 0).sum()), "n_quarters": len(groups),
            "ci95": S.interval(S.call_bootstrap(groups, n_boot)),
            "per_quarter": [float(d) for d in diffs]}


def summary(n_boot: int = 2000) -> dict:
    out = {"years": [], "comparisons": {}, "levels": {}, "gold_in_top": {}}
    data = {}
    for y in YEARS:
        p = OUT_DIR / f"{shard_of(y)}.npz"
        if p.exists():
            data[y] = np.load(p)
    if not data:
        raise FileNotFoundError(f"no scores under {OUT_DIR}")
    out["years"] = sorted(data)
    out["missing"] = [shard_of(y) for y in YEARS if y not in data]
    for fam_arm in ("chain", "scratch", "chain_nogold"):
        late, pit = (f"chain_late_nogold", f"chain_pit_nogold") if fam_arm == "chain_nogold" else \
            (f"{fam_arm}_late", f"{fam_arm}_pit")
        groups = [(z["keydevid"], z[f"{late}__logprob"] - z[f"{pit}__logprob"]) for z in data.values()]
        out["comparisons"][fam_arm] = premium(groups, n_boot)
        out["gold_in_top"][fam_arm] = {
            "pit": float(np.mean([z[f"{pit}__gold_in_top"].mean() for z in data.values()])),
            "late": float(np.mean([z[f"{late}__gold_in_top"].mean() for z in data.values()]))}
    for name in ("closed", "chain_pit", "chain_late", "scratch_pit", "scratch_late"):
        out["levels"][name] = float(np.mean([z[f"{name}__logprob"].mean() for z in data.values()]))
    groups = [(z["keydevid"], z["chain_pit__logprob"] - z["closed__logprob"]) for z in data.values()]
    out["comparisons"]["retrieval_vs_closed"] = premium(groups, n_boot)
    out["git_sha"], out["utc"] = git_sha(), utc_now()
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("comparisons", "levels", "gold_in_top")}, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--year", type=int, required=True)
    s.add_argument("--n-questions", type=int, default=N_QUESTIONS)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--device", default="cuda")
    s.add_argument("--skip-existing", action="store_true")
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        if a.year not in YEARS:
            raise SystemExit(f"--year must be in {YEARS.start} to {YEARS.stop - 1}")
        target = OUT_DIR / f"{shard_of(a.year)}.npz"
        meta = target.with_suffix(".json")
        if a.skip_existing and not a.limit and target.exists() and meta.exists():
            m = json.loads(meta.read_text())
            # coverage, not existence: the file must hold the run this request asks for
            if m.get("n_requested") == a.n_questions and m.get("reader") == reader(a.year):
                print(f"{target} holds {m['n_questions']} questions for this request, skipping")
                return 0
        score_year(a.year, a.n_questions, a.limit, a.device)
    else:
        summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
