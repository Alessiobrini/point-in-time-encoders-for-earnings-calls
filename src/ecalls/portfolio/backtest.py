"""The portfolio test (D26): text against the literature's baselines, point in time against leaky.

Every setting here is fixed by D26 before any result: ridge refit each January from 2013 to 2025 on
calls whose return window had closed, 32 PCA components for an embedding, the penalty chosen by
time-ordered cross-validation on date-sorted training rows, the target winsorized at the training
rows' 1st and 99th percentiles, and every block standardized on the training rows.

**Point in time and leaky.** For test year t a point-in-time specification reads the embeddings made
by the family's vintage with cutoff t-1, which embedded every call it is used on (D11). A leaky one
reads the 2024 vintage's embeddings in every year. Leakage is requested only through `lookahead=True`
on a block, it is written into every output, and nothing here resolves a model by date, so the
registry's refusal of too-new models is not being bypassed silently: the experiment names the leak.

**One row set.** Each year's specifications are scored on the calls that have every feature, so two
specifications never differ because one of them dropped calls.

**Primary metric.** The rank correlation between prediction and realized target within each calendar
quarter, averaged over quarters. Portfolios are secondary because thirteen years of daily long-short
returns give a Sharpe ratio a standard error near 0.3.

    python -m ecalls.portfolio.backtest
"""
import argparse
import datetime as dt
import json
from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl

from ecalls.config import paths as P
from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.portfolio import panel as PN
from ecalls.portfolio import text_features as TF

EMB_DIR = P.PROCESSED_DIR / "pf_embeddings"
EMB_DIR_ALL = P.PROCESSED_DIR / "pf_embeddings_all"
VIEW_COLUMN = {"all": "embedding", "qa": "embedding_qa", "pres": "embedding_pres"}
OUT_DIR = P.REPORTS_DIR / "portfolio"
TEST_YEARS = range(2013, 2026)
PCA_DIM = 32
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
WINSOR = (0.01, 0.99)
TAIL = 0.10
NW_LAGS = 60
FAMILIES = ("ecalls-ret", "ecalls-abl", "chrono-bert-v1", "ecalls-chain-ret")   # the chain added 2026-10-10 (D29)
# The S&P 500 vectors of D26 were built for the first three families, and the chain's are added on
# 2026-10-10. A family enters a target only when its 2024 vintage has vectors for that panel.
SP500_FAMILIES = FAMILIES[:3]


def families_for(target: "Target") -> tuple:
    return tuple(f for f in FAMILIES
                 if f in SP500_FAMILIES or (target.emb_dir / f"{f}-{LEAKY_VINTAGE}1231.parquet").exists())
LEAKY_VINTAGE = 2024
BASE_COLS = ("ear", "mom", "rev", "logcap")


@dataclass(frozen=True)
class Target:
    """One forecasting target: its column, the date it becomes known, and its literature baselines.

    `known` decides which calls may train a refit: a call trains the January refit of year t only if
    its target was known before January 1 of t.
    """
    key: str
    column: str
    known: str
    base_cols: tuple
    portfolios: bool
    design: str
    results_file: str
    predictions_file: str
    panel: str = "sp500"            # "daily" is D28's next-day panel, "quarter" its D26-style panel, both full universe

    @property
    def emb_dir(self):
        return EMB_DIR if self.panel == "sp500" else EMB_DIR_ALL

    @property
    def feat_dir(self):
        return TF.FEAT_DIR if self.panel == "sp500" else TF.FEAT_DIR_ALL


DAILY_BASE = ("reaction", "rev5", "mom", "logcap")
TARGETS = {
    "ret": Target("ret", "target", "exit_date", BASE_COLS, True, "D26",
                  "backtest_results.json", "pf_predictions.parquet"),
    "av": Target("av", "av_target", "av_known_date",
                 ("lrv_week", "lrv_month", "lrv_year", "mkt_lrv_month", "ear", "abs_ear", "logcap"),
                 False, "D27", "av_results.json", "pf_predictions_av.parquet"),
    "sue": Target("sue", "sue_next", "sue_known_date", ("sue_cur", "ear", "mom", "logcap"),
                  False, "D27", "sue_results.json", "pf_predictions_sue.parquet"),
    # D28, the next-day test after the JFE design. The open-to-close target takes the overnight gap in
    # place of the reaction, which contains the session (D28 amendment).
    "ret1d": Target("ret1d", "target_1d", "known_1d", DAILY_BASE, False, "D28",
                    "ret1d_results.json", "pf_predictions_ret1d.parquet", "daily"),
    "ret5d": Target("ret5d", "target_5d", "known_5d", DAILY_BASE, False, "D28",
                    "ret5d_results.json", "pf_predictions_ret5d.parquet", "daily"),
    "o2c": Target("o2c", "target_o2c", "known_1d", ("gap", "rev5", "mom", "logcap"), False, "D28",
                  "o2c_results.json", "pf_predictions_o2c.parquet", "daily"),
    # D28 amendment of 2026-10-09: D26's quarter-horizon target and baselines on the full universe.
    "ret60": Target("ret60", "target", "exit_date", BASE_COLS, False, "D28 amendment",
                    "ret60_results.json", "pf_predictions_ret60.parquet", "quarter"),
}


@dataclass(frozen=True)
class Block:
    kind: str                       # base, ident, tone, tfidf, emb
    family: str | None = None
    lookahead: bool = False
    adjacent: bool = False          # vintage t for test year t, the JFE paper's same-year comparison
    view: str = "all"               # D28: the call vector from all chunks, Q&A only, or presentation only

    @property
    def name(self) -> str:
        if self.kind != "emb":
            return self.kind
        v = "" if self.view == "all" else f"-{self.view}"
        return f"{self.family}{v}{'-leaky' if self.lookahead else ''}{'-adjacent' if self.adjacent else ''}"


@dataclass(frozen=True)
class Spec:
    name: str
    blocks: tuple = field(default_factory=tuple)

    @property
    def lookahead(self) -> bool:
        return any(b.lookahead or b.adjacent for b in self.blocks)


def specifications(views: tuple = ("all",), leaky_views: tuple = ("all",), families: tuple = SP500_FAMILIES) -> list[Spec]:
    B, I = Block("base"), Block("ident")
    specs = [Spec("baselines", (B,)), Spec("identity", (I,)), Spec("identity+baselines", (I, B)),
             Spec("tone", (Block("tone"),)), Spec("tone+baselines", (Block("tone"), B)),
             Spec("tfidf", (Block("tfidf"),)), Spec("tfidf+baselines", (Block("tfidf"), B))]
    for fam in families:
        for view, leak in [(v, False) for v in views] + [(v, True) for v in leaky_views]:
            e = Block("emb", fam, leak, view=view)
            tag = e.name
            specs += [Spec(f"{tag}", (e,)), Spec(f"{tag}+baselines", (e, B)),
                      Spec(f"{tag}+identity+baselines", (e, I, B))]
    return specs


def lexical_specs() -> list[Spec]:
    """The specifications that read no embedding: an early read on a target before the vectors exist."""
    return [s for s in specifications() if not any(b.kind == "emb" for b in s.blocks)]


def vintage_for(family: str, year: int, lookahead: bool, adjacent: bool = False) -> str:
    """Vintage t-1 for test year t, or the 2024 vintage when leaky, or vintage t when adjacent."""
    v = year if adjacent else (LEAKY_VINTAGE if lookahead else year - 1)
    model = f"{family}-{v}1231"
    if not (adjacent or lookahead):
        from ecalls.chunk import registry as R
        R.require_point_in_time(model, f"{year}-01-01")
    return model


# --- features ------------------------------------------------------------------------------------

class Features:
    """Loads each feature source once per test year and hands out aligned matrices."""

    def __init__(self, year: int, keys: np.ndarray, emb_dir=None, feat_dir=None):
        self.year, self.keys = year, keys
        self.emb_dir = emb_dir or EMB_DIR
        self.feat_dir = feat_dir or TF.FEAT_DIR
        self._cache: dict[str, np.ndarray] = {}

    def _emb(self, model: str, view: str = "all") -> np.ndarray:
        key = f"{model}|{view}"
        if key not in self._cache:
            col = VIEW_COLUMN[view]
            df = pl.read_parquet(self.emb_dir / f"{model}.parquet", columns=["keydevid", col])
            self._cache[key] = _align(df, col, self.keys)
        return self._cache[key]

    def _tfidf(self) -> np.ndarray:
        if "tfidf" not in self._cache:
            df = pl.read_parquet(self.feat_dir / f"tfidf_{self.year}.parquet", columns=["keydevid", "tfidf"])
            self._cache["tfidf"] = _align(df, "tfidf", self.keys)
        return self._cache["tfidf"]

    def _tone(self) -> np.ndarray:
        if "tone" not in self._cache:
            col = f"tone_{self.year}"
            df = pl.read_parquet(self.feat_dir / "tone.parquet", columns=["keydevid", col])
            m = dict(zip(df["keydevid"].to_list(), df[col].to_list()))
            self._cache["tone"] = np.array([[m.get(k, np.nan)] for k in self.keys], dtype=float)
        return self._cache["tone"]

    def raw(self, block: Block) -> np.ndarray | None:
        """The block's raw matrix for every key, NaN rows where a source lacks the call."""
        if block.kind == "tone":
            return self._tone()
        if block.kind == "tfidf":
            return self._tfidf()
        if block.kind == "emb":
            return self._emb(vintage_for(block.family, self.year, block.lookahead, block.adjacent), block.view)
        return None


def _align(df: pl.DataFrame, col: str, keys: np.ndarray) -> np.ndarray:
    """Rows of a list column in the order of `keys`, NaN where a key is absent.

    A join and a fixed-width array rather than Python lists: a 768-dimensional file of 29,000 calls
    is 22 million floats, which as Python objects costs several hundred megabytes per file.
    """
    dim = int(df[col].drop_nulls().list.len().max())
    kdf = pl.DataFrame({"keydevid": keys, "_i": np.arange(len(keys))})
    j = kdf.join(df.unique("keydevid"), on="keydevid", how="left").sort("_i")
    out = np.full((len(keys), dim), np.nan, dtype=np.float32)
    present = j[col].is_not_null().to_numpy()
    if present.any():
        out[present] = j.filter(pl.Series(present))[col].cast(pl.Array(pl.Float32, dim)).to_numpy()
    return out


def design(block: Block, tr: pl.DataFrame, te: pl.DataFrame, raw_tr, raw_te,
           base_cols: tuple = BASE_COLS) -> tuple[np.ndarray, np.ndarray]:
    """Training and test matrices for one block, every statistic fitted on the training rows."""
    if block.kind == "base":
        Xtr = tr.select(base_cols).to_numpy().astype(float)
        Xte = te.select(base_cols).to_numpy().astype(float)
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xte = np.where(np.isnan(Xte), med, Xte)
        return Xtr, Xte
    if block.kind == "ident":
        levels = sorted(set(tr["sic2"].to_list()))
        idx = {s: j for j, s in enumerate(levels)}

        def onehot(df):
            X = np.zeros((df.height, len(levels)))
            for i, s in enumerate(df["sic2"].to_list()):
                if s in idx:
                    X[i, idx[s]] = 1.0
            return X
        return onehot(tr), onehot(te)
    if block.kind == "emb":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=PCA_DIM, random_state=0).fit(raw_tr)
        return pca.transform(raw_tr), pca.transform(raw_te)
    return raw_tr.astype(float), raw_te.astype(float)


# --- the regression ------------------------------------------------------------------------------

def ridge(Xtr, ytr, Xte):
    """Ridge with the penalty chosen by `TimeSeriesSplit` on rows already sorted by date.

    Returns test predictions, training predictions and the penalty. The rows must arrive sorted:
    the split reads row order as time order, which was wrong once in this project already.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    best, best_err = None, np.inf
    for a in ALPHAS:
        errs = []
        for i_tr, i_va in TimeSeriesSplit(n_splits=4).split(Xtr):
            m = Ridge(alpha=a).fit(Xtr[i_tr], ytr[i_tr])
            errs.append(float(((m.predict(Xtr[i_va]) - ytr[i_va]) ** 2).mean()))
        if np.mean(errs) < best_err:
            best, best_err = a, float(np.mean(errs))
    m = Ridge(alpha=best).fit(Xtr, ytr)
    return m.predict(Xte), m.predict(Xtr), best


def split_year(panel: pl.DataFrame, year: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """D26's split, both sides sorted by date and call id. A call trains the refit only once its
    target is known: `known_date` when the target sets one, the return window's end otherwise."""
    order = ["day0", "keydevid"]
    known = "known_date" if "known_date" in panel.columns else "exit_date"
    train = panel.filter(pl.col(known) < dt.date(year, 1, 1)).sort(order)
    test = panel.filter(pl.col("day0").dt.year() == year).sort(order)
    return train, test


def run_year(panel: pl.DataFrame, year: int, specs: list[Spec],
             base_cols: tuple = BASE_COLS, target: "Target | None" = None) -> tuple[pl.DataFrame, dict]:
    train, test = split_year(panel, year)
    keys = np.concatenate([train["keydevid"].to_numpy(), test["keydevid"].to_numpy()])
    F = Features(year, keys, target.emb_dir if target else None, target.feat_dir if target else None)
    # One row set: drop any call missing a feature some specification needs.
    blocks = {b for s in specs for b in s.blocks}
    ok = np.ones(len(keys), dtype=bool)
    raws = {}
    for b in blocks:
        r = F.raw(b)
        if r is not None:
            raws[b] = r
            ok &= ~np.isnan(r).any(axis=1)
    ntr = train.height
    ok_tr, ok_te = ok[:ntr], ok[ntr:]
    tr, te = train.filter(pl.Series(ok_tr)), test.filter(pl.Series(ok_te))
    y = tr["target"].to_numpy()
    train_mean = float(y.mean())
    lo, hi = np.quantile(y, WINSOR)
    y = np.clip(y, lo, hi)
    preds = {"keydevid": te["keydevid"].to_list()}
    info = {"year": year, "n_train": tr.height, "n_test": te.height, "train_mean": train_mean,
            "n_dropped_missing_feature": int((~ok).sum()), "specs": {}}
    for s in specs:
        Xtr_parts, Xte_parts = [], []
        for b in s.blocks:
            raw = raws.get(b)
            a, c = design(b, tr, te, raw[:ntr][ok_tr] if raw is not None else None,
                          raw[ntr:][ok_te] if raw is not None else None, base_cols)
            Xtr_parts.append(a)
            Xte_parts.append(c)
        p_te, p_tr, alpha = ridge(np.column_stack(Xtr_parts), y, np.column_stack(Xte_parts))
        preds[s.name] = p_te
        info["specs"][s.name] = {
            "alpha": alpha, "lookahead": s.lookahead,
            "vintages": [vintage_for(b.family, year, b.lookahead, b.adjacent)
                         for b in s.blocks if b.kind == "emb"],
            "long_cutoff": float(np.quantile(p_tr, 1 - TAIL)),
            "short_cutoff": float(np.quantile(p_tr, TAIL))}
    keep_cols = [c for c in ("keydevid", "permno", "day0", "target", "entry_idx", "exit_idx", "logcap",
                             "known_date") if c in te.columns]
    out = te.select(keep_cols)
    out = out.with_columns(train_mean=pl.lit(train_mean))
    return out.join(pl.DataFrame(preds), on="keydevid"), info


# --- metrics -------------------------------------------------------------------------------------

def quarterly_ic(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Spearman correlation of a prediction with the realized target within each calendar quarter."""
    from scipy.stats import spearmanr
    q = df.with_columns(q=pl.col("day0").dt.year() * 10 + pl.col("day0").dt.quarter())
    rows = []
    for (key,), g in q.group_by(["q"], maintain_order=True):
        if g.height >= 20:
            rows.append({"quarter": int(key),
                         "ic": float(spearmanr(g[col].to_numpy(), g["target"].to_numpy()).statistic),
                         "n": g.height})
    return pl.DataFrame(rows).sort("quarter")


def paired_difference(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, block: int = 4,
                      seed: int = 0) -> dict:
    """Mean of a - b over quarters, a t statistic, and a moving-block bootstrap interval."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, int(np.ceil(n / block)))])[:n]
        draws.append(d[idx].mean())
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"mean": float(d.mean()), "t": float(d.mean() / se) if se else float("nan"), "n": n,
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int = NW_LAGS) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West standard errors with Bartlett weights. X carries no constant."""
    X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = Xu[lag:].T @ Xu[:-lag]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.diag(V))


def daily_legs(df: pl.DataFrame, col: str, cutoffs: dict, R: dict, m: np.ndarray, T: int) -> dict:
    """Daily long and short returns, equal and value weighted, from each call's holding window.

    A call is long from day +2 to day +61 when its prediction is at or above its year's long cutoff,
    short at or below the short cutoff. A day without a stock return, after a delisting, earns the
    market, matching the target.
    """
    acc = {k: np.zeros(T) for k in ("l_ew", "l_n", "s_ew", "s_n", "l_vw", "l_w", "s_vw", "s_w")}
    for row in df.select("permno", "day0", "entry_idx", "exit_idx", "logcap", col).iter_rows():
        permno, day0, a, b, logcap, pred = row
        lc, sc = cutoffs[day0.year]
        side = "l" if pred >= lc else ("s" if pred <= sc else None)
        if side is None:
            continue
        r = R[permno][a: b + 1]
        r = np.where(np.isnan(r), m[a: b + 1], r)
        w = float(np.exp(logcap))
        acc[f"{side}_ew"][a: b + 1] += r
        acc[f"{side}_n"][a: b + 1] += 1
        acc[f"{side}_vw"][a: b + 1] += w * r
        acc[f"{side}_w"][a: b + 1] += w
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for side in ("l", "s"):
            out[f"{side}_ew"] = acc[f"{side}_ew"] / acc[f"{side}_n"]
            out[f"{side}_vw"] = acc[f"{side}_vw"] / acc[f"{side}_w"]
    out["n_long"], out["n_short"] = acc["l_n"], acc["s_n"]
    return out


def perf(r: np.ndarray, factors: np.ndarray, rf: np.ndarray | None = None) -> dict:
    """Annualized mean, volatility, Sharpe ratio and five-factor-plus-momentum alpha with NW errors.

    `rf` is subtracted for a long-only portfolio. A long-short portfolio is self-financing and is
    evaluated as is.
    """
    x = r - rf if rf is not None else r
    beta, se = newey_west_ols(x, factors)
    return {"mean_ann": float(x.mean() * 252), "vol_ann": float(x.std(ddof=1) * np.sqrt(252)),
            "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
            "alpha_ann": float(beta[0] * 252), "alpha_t": float(beta[0] / se[0]), "n_days": int(len(x))}


# --- benchmarks ----------------------------------------------------------------------------------

def sp500_benchmarks(calendar, R: dict, CAP: dict) -> dict:
    """Equally and value weighted S&P 500 from point-in-time membership spells."""
    spells = pl.read_parquet(P.UNIVERSE_DIR / "sp500_membership.parquet",
                             columns=["permno", "mbr_start", "mbr_end"])
    T = len(calendar)
    ew_sum, ew_n, vw_sum, vw_w = (np.zeros(T) for _ in range(4))
    for permno, start, end in spells.iter_rows():
        if int(permno) not in R:
            continue
        a = int(np.searchsorted(calendar, np.datetime64(start, "D")))
        b = int(np.searchsorted(calendar, np.datetime64(end, "D"), side="right")) - 1
        if b < a:
            continue
        r = R[int(permno)][a: b + 1]
        prev_cap = np.concatenate([[np.nan], CAP[int(permno)][:-1]])[a: b + 1]   # yesterday's cap
        ok = ~np.isnan(r)
        ew_sum[a: b + 1] += np.where(ok, r, 0.0)
        ew_n[a: b + 1] += ok
        okc = ok & ~np.isnan(prev_cap)
        vw_sum[a: b + 1] += np.where(okc, r * np.nan_to_num(prev_cap), 0.0)
        vw_w[a: b + 1] += np.where(okc, np.nan_to_num(prev_cap), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"sp500_ew": ew_sum / ew_n, "sp500_vw": vw_sum / vw_w}


# --- the run -------------------------------------------------------------------------------------

def load_panel(target: Target = TARGETS["ret"]) -> pl.DataFrame:
    """The event panel with `target` and `known_date` set for one forecasting target.

    Events whose target is missing or not finite are dropped for that target only.
    """
    from ecalls.portfolio import sue as SU
    if target.panel == "quarter":
        from ecalls.portfolio import panel as PN
        p = TF.load_panel(PN.PANEL_QUARTER).with_columns(known_date=pl.col("exit_date"))
        return p.filter(pl.col("target").is_not_null() & pl.col("target").is_finite())
    if target.panel == "daily":
        from ecalls.portfolio import panel_daily as PDL
        p = pl.read_parquet(PDL.PANEL_ALL).with_columns(day0=pl.col("entry_date").cast(pl.Date))
        p = p.rename({target.column: "target"}).with_columns(known_date=pl.col(target.known).cast(pl.Date))
        return p.filter(pl.col("target").is_not_null() & pl.col("target").is_finite()
                        & pl.col("known_date").is_not_null())
    p = TF.load_panel()
    ff = pl.read_parquet(P.RAW_DIR / "ff" / "ff_daily.parquet").sort("date")
    cal = ff["date"]
    p = p.with_columns(av_known_date=pl.col("av_known_idx").map_elements(lambda i: cal[i], return_dtype=pl.Date))
    if SU.SUE_PATH.exists():
        p = p.join(pl.read_parquet(SU.SUE_PATH).select("keydevid", "sue_cur", "sue_next", "sue_known_date"),
                   on="keydevid", how="left")
    if target.column != "target":
        p = p.drop("target").rename({target.column: "target"})
    p = p.with_columns(known_date=pl.col(target.known))
    return p.filter(pl.col("target").is_not_null() & pl.col("target").is_finite()
                    & pl.col("known_date").is_not_null())


def increments(ic: dict, specs: list[Spec]) -> dict:
    """D27's answer to whether the text is worth having: text plus X minus X, paired by quarter, for
    every text specification built on the baselines or on identity and the baselines."""
    names = {s.name for s in specs}
    out = {}
    for name in sorted(names - {"baselines", "identity", "identity+baselines"}):
        for ctl in ("identity+baselines", "baselines"):
            if name.endswith("+" + ctl) and name != ctl and ctl in names:
                if ctl == "baselines" and name.endswith("+identity+baselines"):
                    continue
                both = ic[name].join(ic[ctl], on="quarter", suffix="_ctl")
                out[f"{name} minus {ctl}"] = paired_difference(both["ic"], both["ic_ctl"])
                break
    return out


def daily_long_short(preds: pl.DataFrame, col: str, min_calls: int = 20) -> pl.DataFrame:
    """D28's portfolio, as in the JFE paper: each entry close, the calls entering then are ranked
    against each other, top decile long and bottom decile short, equally weighted, held one day.

    The target is the next-day return minus the market's, and the market cancels between the legs, so
    the long-short return is the difference of the two legs' mean targets. A day with fewer than
    `min_calls` entering calls is not traded. The return is dated on the day it is earned.
    """
    rows = []
    for (day,), g in preds.group_by(["day0"], maintain_order=True):
        if g.height < min_calls:
            continue
        x = g[col].to_numpy()
        lo, hi = np.quantile(x, [0.1, 0.9])
        y = g["target"].to_numpy()
        long, short = y[x >= hi], y[x <= lo]
        if len(long) and len(short):
            rows.append({"entry_date": day, "earned_on": g["known_date"][0],
                         "ls": float(long.mean() - short.mean()), "n_calls": g.height})
    return pl.DataFrame(rows).sort("earned_on") if rows else pl.DataFrame()


def daily_portfolio_stats(ls: pl.DataFrame) -> dict:
    """Sharpe ratio over every trading day of the span, zeros where nothing traded, and over traded
    days only, with the five-factor-plus-momentum alpha at 5 Newey-West lags (one-day holdings)."""
    ff = pl.read_parquet(P.RAW_DIR / "ff" / "ff_daily.parquet").sort("date")
    lo, hi = ls["earned_on"].min(), ls["earned_on"].max()
    span = ff.filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
    full = span.select("date").join(ls.select(pl.col("earned_on").alias("date"), "ls"), on="date", how="left")
    x = full["ls"].fill_null(0.0).to_numpy()
    fac = span.select("mktrf", "smb", "hml", "rmw", "cma", "umd").to_numpy()
    beta, se = newey_west_ols(x, fac, lags=5)
    t = ls["ls"].to_numpy()
    return {"sharpe_all_days": float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
            "sharpe_traded_days": float(t.mean() / t.std(ddof=1) * np.sqrt(252)),
            "mean_traded_day": float(t.mean()), "share_days_traded": float(len(t) / len(x)),
            "alpha_ann": float(beta[0] * 252), "alpha_t": float(beta[0] / se[0]), "n_days": int(len(x))}


def r2_oos(preds: pl.DataFrame, col: str) -> float:
    """Out-of-sample R squared against each refit's training mean, pooled over every test call."""
    y = preds["target"].to_numpy()
    sse = float(((y - preds[col].to_numpy()) ** 2).sum())
    sst = float(((y - preds["train_mean"].to_numpy()) ** 2).sum())
    return 1.0 - sse / sst


def run(years=TEST_YEARS, specs: list[Spec] | None = None, target: Target = TARGETS["ret"]) -> dict:
    if specs is None:
        # D28 compares the three call vectors point in time, and the leak on the all-chunk vector.
        specs = (specifications(families=families_for(target)) if target.panel == "sp500"
                 else specifications(("all", "qa", "pres"), families=families_for(target)))
    panel = load_panel(target)
    frames, infos = [], []
    for y in years:
        df, info = run_year(panel, y, specs, target.base_cols, target)
        frames.append(df)
        infos.append(info)
        print(f"{y}: {info['n_train']:,} train, {info['n_test']:,} test, "
              f"{info['n_dropped_missing_feature']} dropped for a missing feature", flush=True)
    preds = pl.concat(frames, how="diagonal")

    # Primary metric
    ic = {s.name: quarterly_ic(preds, s.name) for s in specs}
    results = {"design": target.design, "target": target.key, "specs": {}, "premium": {}, "years": infos}
    for s in specs:
        q = ic[s.name]
        results["specs"][s.name] = {"lookahead": s.lookahead, "ic_mean": float(q["ic"].mean()),
                                    "r2_oos": r2_oos(preds, s.name),
                                    "ic_t": float(q["ic"].mean() / (q["ic"].std() / np.sqrt(q.height))),
                                    "n_quarters": q.height,
                                    "ic_by_quarter": dict(zip(map(str, q["quarter"]), q["ic"]))}
    for fam in FAMILIES:
        if fam not in ic:
            continue                # a lexical-only run has no embedding to leak
        for suffix in ("", "+baselines", "+identity+baselines"):
            pit, leak = ic[f"{fam}{suffix}"], ic[f"{fam}-leaky{suffix}"]
            both = pit.join(leak, on="quarter", suffix="_leak")
            pre = both.filter(pl.col("quarter") < 20251)
            results["premium"][f"{fam}{suffix}"] = {
                "leaky_minus_pit_all_quarters": paired_difference(both["ic_leak"], both["ic"]),
                "leaky_minus_pit_through_2024": paired_difference(pre["ic_leak"], pre["ic"])}

    results["increment"] = increments(ic, specs)
    if target.panel == "daily":
        results["daily_long_short"] = {}
        for s_ in specs:
            ls = daily_long_short(preds, s_.name)
            if ls.height:
                results["daily_long_short"][s_.name] = daily_portfolio_stats(ls)
    if not target.portfolios:
        results.update(git_sha=git_sha(), ts=utc_now())
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        preds.write_parquet(P.PROCESSED_DIR / target.predictions_file)
        (OUT_DIR / target.results_file).write_text(json.dumps(results, indent=2, default=float) + "\n")
        return results

    # Portfolios, secondary
    calendar, m, R, CAP, _, _ = PN.load_returns()
    T = len(calendar)
    ff = pl.read_parquet(P.RAW_DIR / "ff" / "ff_daily.parquet").sort("date")
    fac = ff.select("mktrf", "smb", "hml", "rmw", "cma", "umd").to_numpy()
    rf = ff["rf"].to_numpy()
    first = int(preds["entry_idx"].min())
    last = int(preds["exit_idx"].max())
    span = slice(first, last + 1)
    cut = {y["year"]: None for y in infos}
    daily = {"date": calendar[span].astype("datetime64[D]").astype(str)}
    results["portfolios"] = {}
    for s in specs:
        cutoffs = {y["year"]: (y["specs"][s.name]["long_cutoff"], y["specs"][s.name]["short_cutoff"])
                   for y in infos}
        legs = daily_legs(preds, s.name, cutoffs, R, m, T)
        entry = {}
        for w in ("ew", "vw"):
            ls = np.nan_to_num(legs[f"l_{w}"][span] - legs[f"s_{w}"][span])
            lo = legs[f"l_{w}"][span]
            lo = np.where(np.isnan(lo), rf[span], lo)
            entry[f"long_short_{w}"] = perf(ls, fac[span])
            entry[f"long_{w}"] = perf(lo, fac[span], rf[span])
            daily[f"{s.name}|ls_{w}"] = ls
        entry["mean_names_long"] = float(np.mean(legs["n_long"][span]))
        entry["mean_names_short"] = float(np.mean(legs["n_short"][span]))
        entry["lookahead"] = s.lookahead
        results["portfolios"][s.name] = entry
    bm = sp500_benchmarks(calendar, R, CAP)
    allcalls = daily_legs(preds.with_columns(pl.lit(0.0).alias("_all")), "_all",
                          {y: (-np.inf, np.inf) for y in cut}, R, m, T)
    results["benchmarks"] = {
        "sp500_ew": perf(np.nan_to_num(bm["sp500_ew"][span]), fac[span], rf[span]),
        "sp500_vw": perf(np.nan_to_num(bm["sp500_vw"][span]), fac[span], rf[span]),
        "all_calls_ew": perf(np.nan_to_num(allcalls["l_ew"][span]), fac[span], rf[span]),
        "market": perf(fac[span][:, 0] + rf[span], fac[span], rf[span]),
    }
    results.update(git_sha=git_sha(), ts=utc_now())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds.write_parquet(P.PROCESSED_DIR / target.predictions_file)
    pl.DataFrame(daily).write_parquet(P.PROCESSED_DIR / "pf_daily_returns.parquet")
    (OUT_DIR / target.results_file).write_text(json.dumps(results, indent=2, default=float) + "\n")
    return results


def run_adjacent(target: Target) -> dict:
    """The adjacent-vintage premium, D27 amendment of 2026-10-08: vintage t against vintage t-1.

    He, Lv, Manela and Wu call this their cleanest test for look-ahead bias (JFE 2026, p. 11). Test years
    stop at 2024, the last year with a same-year vintage.
    """
    B, I = Block("base"), Block("ident")
    specs = []
    for fam in families_for(target):
        for adj in (False, True):
            e = Block("emb", fam, adjacent=adj)
            specs += [Spec(e.name, (e,)), Spec(f"{e.name}+baselines", (e, B)),
                      Spec(f"{e.name}+identity+baselines", (e, I, B))]
    panel = load_panel(target)
    frames, infos = [], []
    for y in range(TEST_YEARS.start, 2025):
        df, info = run_year(panel, y, specs, target.base_cols, target)
        frames.append(df)
        infos.append(info)
    preds = pl.concat(frames, how="diagonal")
    ic = {s.name: quarterly_ic(preds, s.name) for s in specs}
    out = {"design": "D27 amendment 2026-10-08, adjacent vintage", "target": target.key,
           "years": infos, "specs": {}, "premium": {}}
    for s in specs:
        q = ic[s.name]
        out["specs"][s.name] = {"lookahead": s.lookahead, "ic_mean": float(q["ic"].mean()),
                                "ic_t": float(q["ic"].mean() / (q["ic"].std() / np.sqrt(q.height))),
                                "n_quarters": q.height}
    for fam in families_for(target):
        for suffix in ("", "+baselines", "+identity+baselines"):
            both = ic[f"{fam}{suffix}"].join(ic[f"{fam}-adjacent{suffix}"], on="quarter", suffix="_adj")
            out["premium"][f"{fam}{suffix}"] = paired_difference(both["ic_adj"], both["ic"])
    out.update(git_sha=git_sha(), ts=utc_now())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{target.key}_adjacent.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=sorted(TARGETS), default="ret")
    ap.add_argument("--adjacent", action="store_true", help="the same-year vintage premium only")
    ap.add_argument("--lexical-only", action="store_true",
                    help="no embedding specifications, results written beside the full run's as *_lexical")
    args = ap.parse_args(argv)
    if args.adjacent:
        r = run_adjacent(TARGETS[args.target])
        for k, v in r["premium"].items():
            print(f"adjacent minus PIT {k:<36} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] t {v['t']:.2f}")
        return 0
    target, specs = TARGETS[args.target], None
    if args.lexical_only:
        tag = lambda f: f.replace(".json", "_lexical.json").replace(".parquet", "_lexical.parquet")
        target = replace(target, results_file=tag(target.results_file),
                         predictions_file=tag(target.predictions_file))
        specs = lexical_specs()
    r = run(target=target, specs=specs)
    print(f"\n{'specification':<40}{'IC':>8}{'t':>7}{'R2 oos':>9}{'LS ew SR':>10}{'alpha t':>9}")
    for name, v in r["specs"].items():
        p = r.get("portfolios", {}).get(name, {}).get("long_short_ew") or \
            {"sharpe": r.get("daily_long_short", {}).get(name, {}).get("sharpe_all_days", float("nan")),
             "alpha_t": r.get("daily_long_short", {}).get(name, {}).get("alpha_t", float("nan"))}
        print(f"{name:<40}{v['ic_mean']:>8.4f}{v['ic_t']:>7.2f}{v['r2_oos']:>9.4f}{p['sharpe']:>10.2f}{p['alpha_t']:>9.2f}")
    for k, v in r["increment"].items():
        print(f"increment {k:<52} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] t {v['t']:.2f}")
    for k, v in r["premium"].items():
        a = v["leaky_minus_pit_through_2024"]
        print(f"premium {k:<36} {a['mean']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] t {a['t']:.2f}")
    for k, v in r.get("benchmarks", {}).items():
        print(f"benchmark {k:<20} SR {v['sharpe']:.2f} alpha t {v['alpha_t']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
