"""Phase 2, diagnostic 1: do call representations encode the company more than the quarter?

Yu (Ding), Liu and He (arXiv 2412.18029, ACL Findings 2025) report that "current transcript
representations predominantly capture ticker identity rather than offering financially meaningful
insights specific to each earnings", evidenced by same-ticker representations being far more similar
than cross-ticker ones. EARNINGSINONE (arXiv 2606.29734) replicates it at "0.80 vs 0.67 avg".

If that holds here, an embedding of a call is largely a company fingerprint, and any predictive model
built on it is partly just learning which company it is looking at. So this runs before anything
expensive.

Two representations, because the obvious confound is the test itself. A call names its own company
and its own executives dozens of times, so clustering by company is partly trivial. The masked
variant removes the company name, the speaker names and the ticker-like tokens, and the gap between
the two says how much of the identity signal is surface naming rather than content. That also sizes
the anonymization control that Glasserman and Lin (arXiv 2309.17322) found mattered more than
look-ahead bias itself.

Run:
    python -m ecalls.diagnostics.identity_confound
    python -m ecalls.diagnostics.identity_confound --companies 300 --calls-per 20 --section qa
"""
import argparse
import re
import sys

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ecalls.config import paths as P

RNG = np.random.default_rng(0)


def sample_calls(n_companies: int, calls_per: int) -> pd.DataFrame:
    """Companies deep enough for a within-company comparison, from the point-in-time panel."""
    c = pd.read_parquet(P.CALLS, columns=["keydevid", "companyid", "company_name", "call_date",
                                          "in_index_at_call"])
    c = c[c["in_index_at_call"]]
    per = c.groupby("companyid").size()
    eligible = per[per >= calls_per].index.to_numpy()
    pick = RNG.choice(eligible, size=min(n_companies, len(eligible)), replace=False)
    out = []
    for cid in pick:
        g = c[c["companyid"] == cid].sort_values("call_date")
        idx = np.linspace(0, len(g) - 1, calls_per).astype(int)   # spread across the firm's history
        out.append(g.iloc[np.unique(idx)])
    return pd.concat(out, ignore_index=True)


def fetch_components(samp: pd.DataFrame, section: str | None) -> pd.DataFrame:
    """Text and speaker names per call, read only from the quarterly shards that hold them.

    Reading the consolidated `components.parquet` instead runs out of memory: it is 5.4 GB sorted by
    (call_date, keydevid, component_order), so a `keydevid in (...)` predicate prunes almost no row
    groups and DuckDB scans the whole file to aggregate a few hundred calls. The shard for every call
    is already recorded in `calls.parquet`, so reading those few files and filtering in pyarrow moves
    tens of MB instead of gigabytes.
    """
    import pyarrow.parquet as pq

    shards = pd.read_parquet(P.CALLS, columns=["keydevid", "shard"])
    samp = samp.merge(shards, on="keydevid", how="left")
    wanted = set(samp["keydevid"].astype("int64"))

    frames = []
    for sh in sorted(samp["shard"].dropna().unique()):
        path = P.components_shard(sh)
        if not path.exists():
            continue
        t = pq.read_table(path, columns=["keydevid", "component_order", "section",
                                         "speaker_name", "text"],
                          filters=[("keydevid", "in", wanted)])
        if t.num_rows:
            frames.append(t.to_pandas())
    if not frames:
        return pd.DataFrame(columns=["keydevid", "text", "names"])
    comp = pd.concat(frames, ignore_index=True)
    if section:
        comp = comp[comp["section"] == section]
    comp = comp.sort_values(["keydevid", "component_order"])

    text = comp.groupby("keydevid")["text"].apply(lambda s_: " ".join(s_.astype(str)))
    names = comp.groupby("keydevid")["speaker_name"].apply(
        lambda s_: sorted({x for x in s_.dropna().astype(str) if x}))
    return pd.DataFrame({"keydevid": text.index, "text": text.to_numpy(),
                         "names": names.reindex(text.index).to_numpy()})


def mask_identity(text: str, company: str, speakers: list[str]) -> str:
    """Remove the naming that makes company identity trivially recoverable."""
    if not isinstance(text, str):
        return ""
    pats = []
    for name in [company] + list(speakers):
        if not isinstance(name, str) or len(name) < 3:
            continue
        for tok in re.split(r"[\s,\.]+", name):
            if len(tok) >= 4 and tok.lower() not in {"inc", "corp", "company", "group", "the",
                                                     "holdings", "international", "limited"}:
                pats.append(re.escape(tok))
    # Ticker-shaped tokens first, and lowercase placeholders. Running the name pass first with an
    # uppercase " NAME " placeholder let the ticker regex \b[A-Z]{2,5}\b match NAME itself and
    # rewrite every masked name as TICKER, collapsing the two into one token.
    text = re.sub(r"\b[A-Z]{2,5}\b", " __ticker__ ", text)
    if pats:
        text = re.sub(r"\b(" + "|".join(sorted(set(pats), key=len, reverse=True)) + r")\b",
                      " __name__ ", text, flags=re.IGNORECASE)
    return text


def similarity_gap(X, companyid: np.ndarray) -> dict:
    """Mean cosine similarity for pairs from the same company against pairs from different ones.

    TF-IDF rows are L2-normalized, so the cosine is a dot product. Sampling pairs rather than
    forming the full matrix keeps this cheap at a few thousand calls.
    """
    n = X.shape[0]
    same, diff = [], []
    order = RNG.permutation(n)
    by_co: dict[int, list[int]] = {}
    for i in order:
        by_co.setdefault(int(companyid[i]), []).append(int(i))
    for idxs in by_co.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                same.append(float(X[idxs[a]].multiply(X[idxs[b]]).sum()))
    # Draw until we have the requested number of cross-company pairs, rather than drawing that many
    # times and skipping the same-company hits. Skipping made the realized cross count fall short of
    # the within count, and with few companies it could return an empty list and a NaN mean.
    n_cross = min(len(same), 200_000)
    tries, cap = 0, n_cross * 100 + 1000
    while len(diff) < n_cross and tries < cap:
        tries += 1
        i, j = RNG.integers(0, n, 2)
        if companyid[i] == companyid[j]:
            continue
        diff.append(float(X[i].multiply(X[j]).sum()))
    if not diff:
        raise ValueError("no cross-company pairs available; the sample has one company")
    return {"within_mean": float(np.mean(same)), "within_n": len(same),
            "cross_mean": float(np.mean(diff)), "cross_n": len(diff),
            "gap": float(np.mean(same) - np.mean(diff))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", type=int, default=200)
    ap.add_argument("--calls-per", type=int, default=12)
    ap.add_argument("--section", default=None, choices=["presentation", "qa", "operator"])
    args = ap.parse_args(argv)

    samp = sample_calls(args.companies, args.calls_per)
    print(f"sample: {samp.companyid.nunique()} companies, {len(samp):,} calls"
          + (f", section={args.section}" if args.section else ", full call"), flush=True)

    got = fetch_components(samp, args.section)
    df = samp.merge(got, on="keydevid", how="inner")
    print(f"  text fetched for {len(df):,} calls, "
          f"{df.text.str.len().sum()/1e6:.0f} MB", flush=True)
    df["masked"] = [mask_identity(t, c, s if isinstance(s, (list, np.ndarray)) else [])
                    for t, c, s in zip(df.text, df.company_name, df.names)]

    results = {}
    for label, col in (("raw text", "text"), ("names masked", "masked")):
        vec = TfidfVectorizer(max_features=50_000, min_df=3, sublinear_tf=True,
                              stop_words="english")
        X = vec.fit_transform(df[col])
        r = similarity_gap(X, df["companyid"].to_numpy())
        results[label] = r
        print(f"\n=== {label} (TF-IDF, {X.shape[1]:,} features) ===")
        print(f"  within-company cosine  {r['within_mean']:.4f}  over {r['within_n']:,} pairs")
        print(f"  cross-company  cosine  {r['cross_mean']:.4f}  over {r['cross_n']:,} pairs")
        print(f"  gap                    {r['gap']:+.4f}")

    print("\n=== reference ===")
    print("  EARNINGSINONE reports 0.80 within vs 0.67 cross for neural embeddings, gap +0.13.")
    raw, msk = results["raw text"]["gap"], results["names masked"]["gap"]
    if raw > 0:
        print(f"  masking names removes {100*(raw-msk)/raw:.0f}% of the gap "
              f"({raw:+.4f} -> {msk:+.4f}), so the remainder is not surface naming.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
