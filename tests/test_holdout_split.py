"""The forecast premium split by whether the later vintage trained on the test call."""
import datetime as dt

import numpy as np
import polars as pl

from ecalls.portfolio import holdout_split as HS
from ecalls.train import holdout as HO


def _panel(n_per_q: int = 400, quarters: int = 8, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    k = 1
    for q in range(quarters):
        day = dt.date(2015 + q // 4, 3 * (q % 4) + 1, 15)
        for _ in range(n_per_q):
            rows.append({"keydevid": k, "day0": day, "target": float(rng.normal())})
            k += 1
    df = pl.DataFrame(rows)
    noise = rng.normal(size=df.height)
    held = np.array([HO.is_holdout(int(x)) for x in df["keydevid"]])
    y = df["target"].to_numpy()
    # The leaky forecast knows the target only on calls it trained on, so its premium is there only.
    return df.with_columns(
        **{"fam": pl.Series(noise),
           "fam-leaky": pl.Series(np.where(held, noise, y + 0.1 * noise))})


def test_a_premium_from_reading_the_call_shows_on_trained_calls_only():
    r = HS.split(_panel(), families=("fam",), specs=("",))["fam"]
    assert r["trained"]["mean"] > 0.5
    assert abs(r["held_out"]["mean"]) < 0.3
    assert r["held_out"]["calls"] + r["trained"]["calls"] == 3200


def test_the_split_follows_the_shared_holdout_contract():
    df = HS.label_holdout(_panel(n_per_q=50, quarters=2))
    assert df["held_out"].to_list() == [HO.is_holdout(int(k)) for k in df["keydevid"]]
    assert 0.03 < df["held_out"].mean() < 0.2


def test_quarters_from_2025_are_left_out():
    p = _panel(quarters=2).with_columns(day0=pl.lit(dt.date(2025, 2, 1)))
    assert HS.premium(p, "fam", "") is None
