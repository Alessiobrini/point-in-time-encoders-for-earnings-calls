"""Does the ridge penalty grid's ceiling change the forecasts? A check from the code audit of 2026-10-13.

    python -m ecalls.diagnostics.ridge_grid [--targets ret1d o2c sue]

Cross-validation picks the largest penalty of `backtest.ALPHAS` (10^4) in most fits, so the optimum may
lie above the grid. This reruns three specifications per target, the baselines alone, the baselines
with our point-in-time vectors and the baselines with the 2024 vectors, once on the grid of the paper
and once on a grid extended to 10^7, on the same rows, and reports the IC, the increment and the
look-ahead premium under both. Standardized, nearly orthogonal predictors make ridge close to a
rescaling of least squares, which leaves the ranks of the forecasts, and so a rank IC, nearly
unchanged, and this measures how close. Writes `reports/portfolio/ridge_grid_check.json` and nothing
the paper reads.
"""
import argparse
import json

import numpy as np
import polars as pl

from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.portfolio import backtest as BT

EXTENDED = BT.ALPHAS + (1e5, 1e6, 1e7)
FAMILY = "ecalls-ret"


def specs() -> list:
    B = BT.Block("base")
    pit, late = BT.Block("emb", FAMILY), BT.Block("emb", FAMILY, lookahead=True)
    return [BT.Spec("baselines", (B,)), BT.Spec("pit+baselines", (pit, B)),
            BT.Spec("late+baselines", (late, B))]


def mean_ic(preds: pl.DataFrame, col: str) -> tuple[float, np.ndarray]:
    q = BT.quarterly_ic(preds, col)
    return float(q["ic"].mean()), q["ic"].to_numpy()


def one_grid(target, alphas) -> dict:
    saved = BT.ALPHAS
    BT.ALPHAS = alphas
    try:
        panel = BT.load_panel(target)
        frames, chosen = [], {s.name: [] for s in specs()}
        for y in BT.TEST_YEARS:
            df, info = BT.run_year(panel, y, specs(), target.base_cols, target)
            frames.append(df)
            for k, v in info["specs"].items():
                chosen[k].append(v["alpha"])
    finally:
        BT.ALPHAS = saved
    preds = pl.concat(frames, how="diagonal")
    ic = {s.name: mean_ic(preds, s.name) for s in specs()}
    inc = BT.paired_difference(ic["pit+baselines"][1], ic["baselines"][1])
    prem = BT.paired_difference(ic["late+baselines"][1], ic["pit+baselines"][1])
    return {"alphas": list(alphas), "ic": {k: v[0] for k, v in ic.items()},
            "increment": inc, "premium": prem,
            "share_at_ceiling": {k: float(np.mean(np.array(v) == max(alphas))) for k, v in chosen.items()},
            "preds": preds}


def check(target_key: str) -> dict:
    target = BT.TARGETS[target_key]
    a, b = one_grid(target, BT.ALPHAS), one_grid(target, EXTENDED)
    rank = {}
    from scipy.stats import spearmanr
    m = a.pop("preds").join(b.pop("preds").select(["keydevid", *[s.name for s in specs()]]),
                            on="keydevid", suffix="_ext")
    for s in specs():
        rank[s.name] = float(spearmanr(m[s.name], m[f"{s.name}_ext"]).statistic)
    return {"paper_grid": a, "extended_grid": b, "rank_correlation_of_forecasts": rank}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", nargs="+", default=["ret1d", "o2c", "sue"])
    args = ap.parse_args(argv)
    out = {"family": FAMILY, "targets": {}, "git_sha": git_sha(), "ts": utc_now()}
    for t in args.targets:
        out["targets"][t] = r = check(t)
        for g in ("paper_grid", "extended_grid"):
            e = r[g]
            print(f"{t:<6} {g:<14} increment {e['increment']['mean']:+.4f} (t {e['increment']['t']:.2f})  "
                  f"premium {e['premium']['mean']:+.4f} (t {e['premium']['t']:.2f})  "
                  f"at ceiling {e['share_at_ceiling']}", flush=True)
        print(f"{t:<6} rank correlation of forecasts across grids {r['rank_correlation_of_forecasts']}")
    path = BT.OUT_DIR / "ridge_grid_check.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
