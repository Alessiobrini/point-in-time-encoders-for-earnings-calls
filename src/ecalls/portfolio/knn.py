"""Nearest-neighbor forecasting from call vectors, retrieval inside a backtest with no LLM (D30).

A call's forecast is the mean target of the k earlier calls whose vectors are closest to its own. An
earlier call enters the pool only once its target was known, strictly before the test call's event
day, so the forecast uses nothing a trader could not have had. Every vector a test year uses comes
from one vintage: t-1 point in time, the 2024 vintage when leaky, vintage t when adjacent. The leaky
premium is then what the retrieval step gains from an encoder that has read the future.

The primary pool excludes the test call's own firm. Calls of one company are each other's nearest
neighbors (the identity diagnostic of 2026-09-26), so with its own firm in the pool the forecast
becomes the firm's history, and the encoder's contribution is the choice among other firms' calls.

    python -m ecalls.portfolio.knn --target ret1d --universe sp500
"""
import argparse
import json

import numpy as np
import polars as pl

from ecalls.config import paths as P
from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.portfolio import backtest as BT

K_PRIMARY = 50
K_ALSO = (10, 200)
FAMILIES = ("ecalls-ret", "ecalls-abl", "chrono-bert-v1", "ecalls-chain-ret")   # the chain added 2026-10-10 (D29)
OUT_DIR = P.REPORTS_DIR / "portfolio"


def knn_forecast(pool_vec: np.ndarray, pool_known: np.ndarray, pool_y: np.ndarray, pool_firm: np.ndarray,
                 q_vec: np.ndarray, q_day: np.ndarray, q_firm: np.ndarray, ks=(K_PRIMARY,),
                 exclude_same_firm: bool = True) -> dict:
    """Forecasts for each query at every k in `ks`, NaN where the eligible pool is empty.

    `pool_known` and `q_day` are datetime64[D]. Vectors are L2-normalized, so a dot product is the
    cosine. The pool is sorted by known date once, and queries sharing an event day share one eligible
    prefix, which keeps the work to one matrix product per day.
    """
    order = np.argsort(pool_known, kind="stable")
    pv, pk, py, pf = pool_vec[order], pool_known[order], pool_y[order], pool_firm[order]
    out = {k: np.full(len(q_vec), np.nan) for k in ks}
    kmax = max(ks)
    for day in np.unique(q_day):
        qi = np.flatnonzero(q_day == day)
        n = int(np.searchsorted(pk, day, side="left"))        # known strictly before the event day
        if n == 0:
            continue
        sims = q_vec[qi] @ pv[:n].T
        if exclude_same_firm:
            sims[q_firm[qi][:, None] == pf[:n][None, :]] = -np.inf
        m = min(kmax, n)
        top = np.argpartition(-sims, m - 1, axis=1)[:, :m] if m < n else np.tile(np.arange(n), (len(qi), 1))
        top_sims = np.take_along_axis(sims, top, axis=1)
        rank = np.argsort(-top_sims, axis=1, kind="stable")
        top, top_sims = np.take_along_axis(top, rank, axis=1), np.take_along_axis(top_sims, rank, axis=1)
        valid = np.isfinite(top_sims)
        for k in ks:
            v = valid[:, :k]
            ys = np.where(v, py[top[:, :k]], 0.0)
            cnt = v.sum(axis=1)
            out[k][qi] = np.where(cnt > 0, ys.sum(axis=1) / np.maximum(cnt, 1), np.nan)
    return out


def vintage(family: str, year: int, kind: str) -> str:
    return BT.vintage_for(family, year, lookahead=(kind == "leaky"), adjacent=(kind == "adjacent"))


def load_vectors(model: str, keys: np.ndarray, emb_dir) -> np.ndarray:
    df = pl.read_parquet(emb_dir / f"{model}.parquet", columns=["keydevid", "embedding"])
    return BT._align(df, "embedding", keys)


def run(target_key: str, universe: str, families=FAMILIES, years=BT.TEST_YEARS) -> dict:
    target = BT.TARGETS[target_key]
    panel = BT.load_panel(target)
    emb_dir = BT.EMB_DIR_ALL if universe == "all" else BT.EMB_DIR
    if universe == "sp500":
        # The S&P 500 calls are the ones the D26 vectors cover, the in-index calls with a permno.
        sp = pl.read_parquet(BT.EMB_DIR / "ecalls-ret-20241231.parquet", columns=["keydevid"])
        panel = panel.join(sp, on="keydevid", how="semi")
    panel = panel.sort(["day0", "keydevid"])
    ks = (K_PRIMARY, *K_ALSO)
    preds = []
    for year in years:
        train, test = BT.split_year(panel, year)
        # The pool for any test call of the year is every call known before its event day, which can
        # include calls of the same year, so it is drawn from the whole panel up to the year's end.
        pool = panel.filter((pl.col("day0") >= pl.date(2011, 1, 1)) & (pl.col("known_date") <= pl.date(year, 12, 31)))
        lo, hi = np.quantile(train["target"].to_numpy(), [BT.WINSOR[0], BT.WINSOR[1]])
        py = np.clip(pool["target"].to_numpy(), lo, hi)
        pk = pool["known_date"].to_numpy().astype("datetime64[D]")
        qd = test["day0"].to_numpy().astype("datetime64[D]")
        frame = {"keydevid": test["keydevid"], "day0": test["day0"], "target": test["target"]}
        for fam in families:
            for kind in ("pit", "leaky", "adjacent"):
                if kind == "adjacent" and year > BT.LEAKY_VINTAGE:
                    continue
                model = vintage(fam, year, kind)
                if not (emb_dir / f"{model}.parquet").exists():
                    continue
                pv = load_vectors(model, pool["keydevid"].to_numpy(), emb_dir)
                qv = load_vectors(model, test["keydevid"].to_numpy(), emb_dir)
                ok_p = ~np.isnan(pv).any(axis=1)
                ok_q = ~np.isnan(qv).any(axis=1)
                for excl in (True, False):
                    f = knn_forecast(pv[ok_p], pk[ok_p], py[ok_p], pool["permno"].to_numpy()[ok_p],
                                     np.nan_to_num(qv), qd, test["permno"].to_numpy(), ks, excl)
                    for k, v in f.items():
                        v[~ok_q] = np.nan
                        frame[f"{fam}|{kind}|{'other' if excl else 'all'}|k{k}"] = v
        preds.append(pl.DataFrame(frame))
        print(f"{year}: {test.height:,} test calls, pool up to {pool.height:,}", flush=True)
    df = pl.concat(preds, how="diagonal")
    cols = [c for c in df.columns if "|" in c]
    ic = {c: BT.quarterly_ic(df.filter(pl.col(c).is_not_nan()), c) for c in cols}
    out = {"design": "D30", "target": target_key, "universe": universe, "specs": {}, "premium": {}}
    for c in cols:
        q = ic[c]
        out["specs"][c] = {"ic_mean": float(q["ic"].mean()), "n_quarters": q.height,
                           "ic_t": float(q["ic"].mean() / (q["ic"].std() / np.sqrt(q.height)))}
    for c in cols:
        fam, kind, pool_kind, k = c.split("|")
        if kind == "pit":
            continue
        base = ic[f"{fam}|pit|{pool_kind}|{k}"]
        both = ic[c].join(base, on="quarter", suffix="_pit")
        both = both.filter(pl.col("quarter") < 20251)        # 2025 has no later vintage than its own
        out["premium"][c] = BT.paired_difference(both["ic"], both["ic_pit"])
    out.update(git_sha=git_sha(), ts=utc_now())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"knn_{target_key}_{universe}.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {path}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=("ret1d", "sue", "av"), default="ret1d")
    ap.add_argument("--universe", choices=("sp500", "all"), default="sp500")
    args = ap.parse_args(argv)
    r = run(args.target, args.universe)
    for c, v in r["specs"].items():
        if c.endswith(f"k{K_PRIMARY}"):
            print(f"{c:<44} IC {v['ic_mean']:+.4f}  t {v['ic_t']:.2f}")
    for c, v in r["premium"].items():
        if c.endswith(f"k{K_PRIMARY}"):
            print(f"premium {c:<36} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] t {v['t']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
