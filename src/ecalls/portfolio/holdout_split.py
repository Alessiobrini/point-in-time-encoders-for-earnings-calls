"""The look-ahead premium in a forecast, split by whether the 2024 vintage trained on the test call.

The forecasting panel uses every call, so most test calls were in the 2024 vintage's training set,
while the retrieval benchmark scores held-out calls only. A premium that comes from having read the
call itself shows on the trained calls and not on the held-out ones. A premium on the held-out calls
comes from what the later vintage knows about later years. Held-out calls are 10% of the panel, so
the held-out premium has wide intervals.

    python -m ecalls.portfolio.holdout_split --target sue
"""
import argparse
import json

import numpy as np
import polars as pl

from ecalls.config import paths as P
from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.portfolio import backtest as BT
from ecalls.train import holdout as HO

FAMILIES = ("ecalls-ret", "ecalls-chain-ret", "chrono-bert-v1", "ecalls-abl")
SPECS = ("", "+baselines")          # the text alone, and text with the baselines
LAST_QUARTER = 20251                # 2025 has no later vintage than its own


def label_holdout(preds: pl.DataFrame) -> pl.DataFrame:
    return preds.with_columns(held_out=pl.Series([HO.is_holdout(int(k)) for k in preds["keydevid"]]))


def premium(df: pl.DataFrame, fam: str, spec: str) -> dict | None:
    """Leaky minus point-in-time quarterly IC on the rows given, paired by quarter."""
    pit, leak = BT.quarterly_ic(df, f"{fam}{spec}"), BT.quarterly_ic(df, f"{fam}-leaky{spec}")
    if pit.height == 0 or leak.height == 0:
        return None
    both = leak.join(pit, on="quarter", suffix="_pit").filter(pl.col("quarter") < LAST_QUARTER)
    if both.height < 4:             # the block bootstrap draws blocks of four quarters
        return None
    return {**BT.paired_difference(both["ic"], both["ic_pit"]), "calls": int(both["n"].sum())}


def split(preds: pl.DataFrame, families=FAMILIES, specs=SPECS) -> dict:
    df = label_holdout(preds)
    out = {}
    for fam in families:
        if f"{fam}-leaky" not in df.columns:
            continue
        for spec in specs:
            out[f"{fam}{spec}"] = {name: premium(df.filter(pl.col("held_out") == flag), fam, spec)
                                   for name, flag in (("held_out", True), ("trained", False))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default="sue", choices=sorted(BT.TARGETS))
    args = ap.parse_args(argv)
    target = BT.TARGETS[args.target]
    preds = pl.read_parquet(P.PROCESSED_DIR / target.predictions_file)
    r = {"target": args.target, "split": split(preds), "git_sha": git_sha(), "ts": utc_now()}
    path = P.REPORTS_DIR / "portfolio" / f"{args.target}_holdout_split.json"
    path.write_text(json.dumps(r, indent=2) + "\n")
    for k, v in r["split"].items():
        for name, e in v.items():
            if e:
                print(f"{k:<28} {name:<9} {e['mean']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] "
                      f"t {e['t']:.2f}  quarters {e['n']}  calls {e['calls']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
