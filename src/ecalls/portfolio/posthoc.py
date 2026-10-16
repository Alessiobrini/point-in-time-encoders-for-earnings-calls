"""D27's post hoc check, decided after the first results: does call text add to the current surprise?

The specification in D27 fits text and baselines with one ridge penalty, which cannot shrink a weak
32-column block without shrinking a strong four-column one. The residual fit asks the incremental
question without that compromise: the baseline ridge first, then the text components fitted to its
training residuals, and the two predictions added. Everything else is D27's protocol: the same refits,
split, PCA width, penalty search and winsorizing. Every output is labeled post hoc.

    python -m ecalls.portfolio.posthoc
"""
import json

import numpy as np
import polars as pl

from ecalls.config import paths as P
from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.portfolio import backtest as BT


def residual_year(panel: pl.DataFrame, year: int, text_blocks: list, base_cols: tuple) -> tuple[pl.DataFrame, dict]:
    train, test = BT.split_year(panel, year)
    keys = np.concatenate([train["keydevid"].to_numpy(), test["keydevid"].to_numpy()])
    F = BT.Features(year, keys)
    raws = {b: F.raw(b) for b in text_blocks}
    ok = np.ones(len(keys), dtype=bool)
    for r in raws.values():
        ok &= ~np.isnan(r).any(axis=1)
    ntr = train.height
    tr, te = train.filter(pl.Series(ok[:ntr])), test.filter(pl.Series(ok[ntr:]))
    y = tr["target"].to_numpy()
    lo, hi = np.quantile(y, BT.WINSOR)
    y = np.clip(y, lo, hi)
    base = BT.Block("base")
    Xb_tr, Xb_te = BT.design(base, tr, te, None, None, base_cols)
    pb_te, pb_tr, _ = BT.ridge(Xb_tr, y, Xb_te)
    resid = y - pb_tr
    out = {"keydevid": te["keydevid"].to_list(), "baselines": pb_te}
    one, _, _ = BT.ridge(tr.select("sue_cur").to_numpy(), y, te.select("sue_cur").to_numpy())
    out["sue_cur alone"] = one
    alphas = {}
    for b in text_blocks:
        r = raws[b]
        Xt_tr, Xt_te = BT.design(b, tr, te, r[:ntr][ok[:ntr]], r[ntr:][ok[ntr:]], base_cols)
        pt_te, _, a = BT.ridge(Xt_tr, resid, Xt_te)
        out[f"baselines then {b.name} on residuals"] = pb_te + pt_te
        alphas[b.name] = a
    df = te.select("keydevid", "day0", "target").join(pl.DataFrame(out), on="keydevid")
    return df, {"year": year, "n_train": tr.height, "n_test": te.height, "residual_alpha": alphas}


def main() -> int:
    target = BT.TARGETS["sue"]
    panel = BT.load_panel(target)
    blocks = [BT.Block("emb", f) for f in BT.FAMILIES] + [BT.Block("emb", f, True) for f in BT.FAMILIES] \
        + [BT.Block("tone"), BT.Block("tfidf")]
    frames, infos = [], []
    for y in BT.TEST_YEARS:
        df, info = residual_year(panel, y, blocks, target.base_cols)
        frames.append(df)
        infos.append(info)
    preds = pl.concat(frames)
    cols = [c for c in preds.columns if c not in ("keydevid", "day0", "target")]
    ic = {c: BT.quarterly_ic(preds, c) for c in cols}
    res = {"label": "POST HOC, decided after the first D27 results", "target": "sue", "specs": {},
           "increment_over_baselines": {}, "years": infos}
    for c, q in ic.items():
        res["specs"][c] = {"ic_mean": float(q["ic"].mean()),
                           "ic_t": float(q["ic"].mean() / (q["ic"].std() / np.sqrt(q.height))),
                           "n_quarters": q.height}
        if c.startswith("baselines then"):
            both = q.join(ic["baselines"], on="quarter", suffix="_base")
            res["increment_over_baselines"][c] = BT.paired_difference(both["ic"], both["ic_base"])
    res.update(git_sha=git_sha(), ts=utc_now())
    path = BT.OUT_DIR / "sue_posthoc_residual.json"
    path.write_text(json.dumps(res, indent=2, default=float) + "\n")
    for c, v in res["specs"].items():
        print(f"{c:<52}{v['ic_mean']:>8.4f}{v['ic_t']:>7.2f}")
    for c, v in res["increment_over_baselines"].items():
        print(f"increment {c:<52} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] t {v['t']:.2f}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
