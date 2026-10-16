"""Do call embeddings beat training-free volatility baselines? The premise test.

This is the deferred half of Phase 2 and the roadmap calls it "the risk that could invalidate the
premise". Yu (Ding), Liu and He (2025), *Same Company, Same Signal*, arXiv 2412.18029, ACL Findings,
doi 10.18653/v1/2025.findings-acl.946, report that two training-free baselines, post-earnings
volatility and same-ticker post-earnings volatility, **beat every transcript-based model** they
tested, and that transcript representations mostly encode ticker identity. If that reproduces here,
the predictive leg of this project needs rethinking before anything is built on it.

The design answers one question and refuses to flatter it.

**The split is by time, not at random.** A random split lets a model see a company's later calls while
predicting its earlier ones, and on a panel where the same forty tickers recur every quarter that is
most of the signal. Xue et al. (arXiv 2608.17223) measure random-split inflation at 1.1x to 6.5x on
financial text. Train is the early years, test the late ones, with a gap so no training call overlaps
a test call's measurement window.

**The baselines get the same information the model gets, and no less.** Each is computed from prices
available strictly before the call, so beating them has to come from the text.

**The target is measured after the call, with a gap.** Realized variance over the trading days
following the call, which is the quantity Yu/Ding use and the one an earnings call plausibly moves.

Run:
    python -m ecalls.diagnostics.predictive_premise --panel        # build and describe the panel
    python -m ecalls.diagnostics.predictive_premise --baselines    # score the training-free baselines
"""
import argparse
import json
import os

import numpy as np
import polars as pl

from ecalls.config import paths as P
from ecalls.ingest.manifest import git_sha, utc_now

# Realized variance by stock and day from the VOLARE data set, a separate download. Set its path here.
VOLARE = os.environ.get("ECALLS_VOLARE_CSV", "realized_variance_stocks.csv")
PANEL = P.PROCESSED_DIR / "premise_panel.parquet"

POST_DAYS = 5          # trading days after the call that make up the target
PRE_DAYS = 20          # trading days before the call that make up the pre-call baseline
GAP_DAYS = 1           # the call day itself is excluded from both sides
MIN_COVERAGE = 4       # a target needs at least this many of its POST_DAYS present


def load_volare(path: str = VOLARE) -> pl.DataFrame:
    df = pl.read_csv(path, columns=["date", "symbol", "rv5", "close_price"])
    return (df.with_columns(pl.col("date").str.to_date())
            .rename({"symbol": "ticker"})
            .filter(pl.col("rv5").is_not_null() & (pl.col("rv5") > 0))
            .sort(["ticker", "date"])
            # A trading-day index per ticker, so "the 5 days after" is 5 rows and never 5 calendar
            # days, which would silently shorten any window spanning a weekend or a holiday.
            .with_columns(t=pl.int_range(pl.len()).over("ticker")))


def build_panel(quiet: bool = False) -> dict:
    """One row per call: its target, its baselines, and the ids needed to attach an embedding."""
    import pandas as pd

    from ecalls.ingest import ticker_map as TM

    vol = load_volare()
    calls = pl.read_parquet(P.PROCESSED_DIR / "calls.parquet",
                            columns=["keydevid", "companyid", "permno", "call_date",
                                     "company_name", "shard"]).drop_nulls("permno")
    mapping = pl.read_parquet(TM.TICKER_MAP)

    # As of the call date, because a ticker is reassigned: 12 of the 40 tickers in this panel map to
    # more than one permno and V maps to five. A dictionary lookup would attach the wrong company.
    tagged = pl.from_pandas(
        TM.as_of(calls.to_pandas(), mapping.to_pandas(), date_col="call_date"))
    tagged = tagged.filter(pl.col("ticker").is_in(vol["ticker"].unique()))

    rows = []
    by_ticker = {t: g for t, g in vol.group_by("ticker")}
    for r in tagged.iter_rows(named=True):
        g = by_ticker.get((r["ticker"],))
        if g is None:
            continue
        d = r["call_date"]
        after = g.filter(pl.col("date") > d).head(POST_DAYS + GAP_DAYS - 1)
        before = g.filter(pl.col("date") < d).tail(PRE_DAYS)
        if after.height < MIN_COVERAGE or before.height < PRE_DAYS:
            continue
        rows.append({
            "keydevid": r["keydevid"], "ticker": r["ticker"], "permno": r["permno"],
            "call_date": d, "shard": r["shard"], "company_name": r["company_name"],
            "target_rv": float(after["rv5"].mean()),
            "pre_rv": float(before["rv5"].mean()),
            "pre_rv_5": float(before.tail(5)["rv5"].mean()),
            "n_post": after.height,
        })
    panel = pl.DataFrame(rows).sort(["ticker", "call_date"])

    # The same-ticker previous post-earnings realized variance, which is Yu/Ding's second baseline.
    # Shifted within ticker, so it uses only this company's own history before this call.
    panel = panel.with_columns(
        prev_post_rv=pl.col("target_rv").shift(1).over("ticker"),
        prev_call_date=pl.col("call_date").shift(1).over("ticker"))
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(PANEL)

    meta = {"n_calls": panel.height, "n_tickers": panel["ticker"].n_unique(),
            "date_min": str(panel["call_date"].min()), "date_max": str(panel["call_date"].max()),
            "post_days": POST_DAYS, "pre_days": PRE_DAYS, "gap_days": GAP_DAYS,
            "with_prev_post": int(panel["prev_post_rv"].is_not_null().sum()),
            "git_sha": git_sha(), "built_at": utc_now()}
    (PANEL.with_suffix(".json")).write_text(json.dumps(meta, indent=2) + "\n")
    if not quiet:
        print(f"{meta['n_calls']:,} calls over {meta['n_tickers']} tickers, "
              f"{meta['date_min']} to {meta['date_max']}")
        print(f"  {meta['with_prev_post']:,} have a previous call to use as a baseline")
    return meta


def time_split(panel: pl.DataFrame, test_fraction: float = 0.3):
    """Train on the early calls, test on the late ones, with a gap of one quarter between.

    The gap exists because a training call's target window can overlap a test call's features if the
    two are days apart, which is a leak no random split would show either.
    """
    dates = panel["call_date"].sort()
    cut = dates[int(len(dates) * (1 - test_fraction))]
    import datetime as dt
    gap_end = cut + dt.timedelta(days=90)
    # Sorted by date, with the call id breaking ties. The penalty is chosen by `TimeSeriesSplit`, which
    # takes row order as time order. The panel is built ticker by ticker, so without the sort the
    # "time-ordered" folds were ticker blocks, and the chosen penalty, hence every R2, moved with
    # whatever order a join happened to return. Found 2026-10-06 when the baselines, which use no
    # embeddings, scored 0.6515 next to one embedding file and 0.6625 next to another.
    order = ["call_date", "keydevid"]
    train = panel.filter(pl.col("call_date") < cut).sort(order)
    test = panel.filter(pl.col("call_date") >= gap_end).sort(order)
    return train, test, cut, gap_end


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Out-of-sample R squared against the training mean, plus rank correlation.

    Realized variance is right skewed, so everything is scored in logs and the rank metric is
    reported beside the R squared, since a single crisis quarter can otherwise dominate either one.
    """
    from scipy.stats import spearmanr
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    rho = float(spearmanr(y_true, y_pred).statistic)
    return {"r2": 1 - ss_res / ss_tot, "spearman": rho, "n": int(len(y_true)),
            "rmse": float(np.sqrt(ss_res / len(y_true)))}


def run_baselines(quiet: bool = False) -> dict:
    panel = pl.read_parquet(PANEL).drop_nulls("prev_post_rv")
    train, test, cut, gap_end = time_split(panel)
    y = np.log(test["target_rv"].to_numpy())

    preds = {
        "pre_call_rv_20d": np.log(test["pre_rv"].to_numpy()),
        "pre_call_rv_5d": np.log(test["pre_rv_5"].to_numpy()),
        "same_ticker_prev_post_rv": np.log(test["prev_post_rv"].to_numpy()),
        "train_mean": np.full(test.height, np.log(train["target_rv"].to_numpy()).mean()),
    }
    out = {"split_cut": str(cut), "test_starts": str(gap_end),
           "n_train": train.height, "n_test": test.height,
           "baselines": {k: score(y, v) for k, v in preds.items()},
           "git_sha": git_sha(), "ts": utc_now()}
    if not quiet:
        print(f"train {train.height:,} calls before {cut}, test {test.height:,} from {gap_end}")
        print(f"{'baseline':<28}{'R2':>8}{'Spearman':>10}{'RMSE':>8}")
        for k, v in out["baselines"].items():
            print(f"{k:<28}{v['r2']:>8.4f}{v['spearman']:>10.4f}{v['rmse']:>8.4f}")
    return out


# --- the model side ------------------------------------------------------------------------------

def _ridge(Xtr, ytr, Xte, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)):
    """Ridge with the penalty chosen by time-ordered cross-validation on the training set only.

    `TimeSeriesSplit`, not `RidgeCV`'s default leave-one-out: choosing the penalty on folds that
    include the future is the same leak the train/test split exists to prevent, one level down.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    best, best_err = None, np.inf
    for a in alphas:
        errs = []
        for i_tr, i_va in TimeSeriesSplit(n_splits=4).split(Xtr):
            m = Ridge(alpha=a).fit(Xtr[i_tr], ytr[i_tr])
            errs.append(float(((m.predict(Xtr[i_va]) - ytr[i_va]) ** 2).mean()))
        err = float(np.mean(errs))
        if err < best_err:
            best, best_err = a, err
    m = Ridge(alpha=best).fit(Xtr, ytr)
    return m.predict(Xte), best


def check_point_in_time(train: pl.DataFrame, test: pl.DataFrame, cut, cutoffs: dict) -> None:
    """Refuse an embedding made by a model newer than the moment it is used.

    A training row's features are used when the regression is fitted, at `cut`, so its model's
    cutoff must be on or before `cut`. A test row's features are used when its call happens, so its
    model's cutoff must be on or before the call date. Per-call resolution meets both by construction.
    One vintage for the whole panel (D11) meets them only if its cutoff precedes the cut, and a too-new
    model there would leak the test period into every row.
    """
    import datetime as dt

    from ecalls.chunk.registry import as_date

    cut = as_date(cut)
    bad = []
    for df, when in ((train, None), (test, "call")):
        for mid, d in zip(df["model_id"].to_list(), df["call_date"].to_list()):
            limit = as_date(d) if when == "call" else cut
            c = cutoffs.get(mid)
            if c is None or as_date(c) > limit:
                bad.append((mid, c, str(limit)))
    if bad:
        mid, c, limit = bad[0]
        raise ValueError(
            f"{len(bad)} rows are embedded by a model newer than the moment they are used, first "
            f"{mid} with cutoff {c} used at {limit}. That is look-ahead in the features.")


def run_models(embeddings: str = "premise_embeddings_chrono-bert-v1.parquet",
               quiet: bool = False) -> dict:
    """Embeddings against the baselines, and the two controls that say what a win would mean."""
    emb = pl.read_parquet(P.PROCESSED_DIR / embeddings)
    panel = pl.read_parquet(PANEL).drop_nulls("prev_post_rv").join(
        emb.select(["keydevid", "model_id", "embedding"]), on="keydevid", how="inner")
    train, test, cut, gap_end = time_split(panel)
    from ecalls.chunk import registry as R
    reg = R.load()
    cutoffs = {m: reg.by_name(m).training_cutoff for m in panel["model_id"].unique().to_list()}
    check_point_in_time(train, test, cut, cutoffs)

    # 768 text features against about a thousand training rows is close to p = n, where ridge does
    # badly for reasons that have nothing to do with whether the text carries signal. So the text is
    # also offered at a few compressed widths, with the projection fitted on the training rows only.
    # Without this, "text does not help" and "text was never given a fair chance" look identical.
    def pca_fit(df, k):
        from sklearn.decomposition import PCA
        V = np.vstack([np.asarray(x, dtype=np.float32) for x in df["embedding"].to_list()])
        return PCA(n_components=k, random_state=0).fit(V)

    def mat(df, kind, projections=None):
        V = np.vstack([np.asarray(x, dtype=np.float32) for x in df["embedding"].to_list()])
        base = np.column_stack([np.log(df["pre_rv"].to_numpy()),
                                np.log(df["pre_rv_5"].to_numpy()),
                                np.log(df["prev_post_rv"].to_numpy())])
        # Ticker identity as one-hot, which is Yu/Ding's claim made testable: if text alone scores
        # like identity alone, the text is mostly telling us which company it is.
        tickers = sorted(panel["ticker"].unique().to_list())
        ident = np.zeros((df.height, len(tickers)), dtype=np.float32)
        for i, t in enumerate(df["ticker"].to_list()):
            ident[i, tickers.index(t)] = 1.0
        options = {"text": V, "baselines": base, "text+baselines": np.column_stack([V, base]),
                   "identity": ident, "identity+baselines": np.column_stack([ident, base])}
        for k, pca in (projections or {}).items():
            Z = pca.transform(V)
            options[f"text{k}"] = Z
            options[f"text{k}+baselines"] = np.column_stack([Z, base])
            # The direct test of the identity claim: does the text add anything once the model
            # already knows the company and its recent volatility?
            options[f"text{k}+identity+baselines"] = np.column_stack([Z, ident, base])
        return options[kind]

    ytr = np.log(train["target_rv"].to_numpy())
    yte = np.log(test["target_rv"].to_numpy())
    out = {"embeddings": embeddings, "split_cut": str(cut), "test_starts": str(gap_end),
           "n_train": train.height, "n_test": test.height,
           "dim": len(train["embedding"][0]),
           "vintages_used": sorted(panel["model_id"].unique().to_list()),
           "models": {}, "baselines": {}}

    widths = (8, 32, 128)
    projections = {w: pca_fit(train, w) for w in widths}
    out["pca_widths"] = list(widths)
    out["pca_explained_variance"] = {w: float(projections[w].explained_variance_ratio_.sum())
                                     for w in widths}
    kinds = ["text", "baselines", "text+baselines", "identity", "identity+baselines"]
    for w in widths:
        kinds += [f"text{w}", f"text{w}+baselines", f"text{w}+identity+baselines"]
    for kind in kinds:
        pred, alpha = _ridge(mat(train, kind, projections), ytr, mat(test, kind, projections))
        out["models"][kind] = {**score(yte, pred), "alpha": alpha}

    # The training-free baselines on the same test rows, so every number here is comparable.
    for name, col in (("pre_call_rv_20d", "pre_rv"), ("pre_call_rv_5d", "pre_rv_5"),
                      ("same_ticker_prev_post_rv", "prev_post_rv")):
        out["baselines"][name] = score(yte, np.log(test[col].to_numpy()))

    out.update(git_sha=git_sha(), ts=utc_now())
    if not quiet:
        print(f"train {train.height:,} before {cut}, test {test.height:,} from {gap_end}, "
              f"dim {out['dim']}")
        print(f"\n{'training-free baseline':<28}{'R2':>8}{'Spearman':>10}")
        for k, v in out["baselines"].items():
            print(f"{k:<28}{v['r2']:>8.4f}{v['spearman']:>10.4f}")
        print(f"\n{'fitted on train only':<28}{'R2':>8}{'Spearman':>10}{'alpha':>9}")
        for k, v in out["models"].items():
            print(f"{k:<28}{v['r2']:>8.4f}{v['spearman']:>10.4f}{v['alpha']:>9.1f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", action="store_true", help="build the panel")
    ap.add_argument("--baselines", action="store_true", help="score the training-free baselines")
    ap.add_argument("--models", action="store_true",
                    help="score the embeddings against the baselines and the identity control")
    ap.add_argument("--embeddings", default="premise_embeddings_chrono-bert-v1.parquet")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.panel:
        build_panel()
    if args.baselines:
        r = run_baselines()
        if args.out:
            (P.REPORTS_DIR / args.out).write_text(json.dumps(r, indent=2) + "\n")
            print(f"\nwrote {P.REPORTS_DIR / args.out}")
    if args.models:
        r = run_models(args.embeddings)
        if args.out:
            (P.REPORTS_DIR / args.out).write_text(json.dumps(r, indent=2) + "\n")
            print(f"\nwrote {P.REPORTS_DIR / args.out}")
    if not (args.panel or args.baselines or args.models):
        ap.error("pass --panel, --baselines or --models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
