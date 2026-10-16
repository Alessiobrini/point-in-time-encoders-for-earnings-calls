"""The portfolio backtest: its statistics, its split, its legs, and how it labels the leak."""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from ecalls.portfolio import backtest as BT


def test_newey_west_coefficients_are_ols_and_lag_zero_is_white():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    y = 0.3 + X @ np.array([1.0, -2.0]) + rng.normal(size=500) * (1 + np.abs(X[:, 0]))
    beta, se = BT.newey_west_ols(y, X, lags=0)
    Xc = np.column_stack([np.ones(500), X])
    assert np.allclose(beta, np.linalg.lstsq(Xc, y, rcond=None)[0])
    u = y - Xc @ beta
    inv = np.linalg.inv(Xc.T @ Xc)
    white = np.sqrt(np.diag(inv @ (Xc * u[:, None]).T @ (Xc * u[:, None]) @ inv))
    assert np.allclose(se, white)


def test_newey_west_widens_the_error_of_a_persistent_series():
    rng = np.random.default_rng(1)
    e = np.convolve(rng.normal(size=2100), np.ones(60) / 60, mode="valid")[:2000]
    X = rng.normal(size=(2000, 1))
    _, se0 = BT.newey_west_ols(e, X, lags=0)
    _, se60 = BT.newey_west_ols(e, X, lags=60)
    assert se60[0] > 3 * se0[0]


def test_paired_difference_of_identical_series_is_zero():
    a = np.linspace(-0.1, 0.1, 40)
    r = BT.paired_difference(a, a)
    assert r["mean"] == 0.0 and r["ci95"] == [0.0, 0.0]


def _panel(n_per_year=300, years=(2011, 2012, 2013), seed=0):
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for y in years:
        for i in range(n_per_year):
            d0 = dt.date(y, 1, 5) + dt.timedelta(days=int(i * 300 / n_per_year))
            rows.append({"keydevid": k, "permno": 1 + k % 30, "day0": d0, "year": y,
                         "sic2": int(10 + k % 7), "target": rng.normal() * 0.1,
                         "ear": rng.normal() * 0.05, "mom": rng.normal(), "rev": rng.normal() * 0.1,
                         "logcap": 16 + rng.random(), "entry_idx": k, "exit_idx": k + 59,
                         "exit_date": d0 + dt.timedelta(days=90)})
            k += 1
    return pl.DataFrame(rows)


def test_training_rows_have_closed_windows_and_both_sides_are_sorted():
    p = _panel().sample(fraction=1.0, shuffle=True, seed=2)
    tr, te = BT.split_year(p, 2013)
    assert tr["exit_date"].max() < dt.date(2013, 1, 1)
    assert te["day0"].dt.year().unique().to_list() == [2013]
    assert tr["day0"].is_sorted() and te["day0"].is_sorted()


def test_the_leaky_specification_reads_the_2024_vintage_and_says_so(tmp_path, monkeypatch):
    """Point in time reads vintage t-1, leaky reads 2024, and the output carries the flag."""
    p = _panel()
    monkeypatch.setattr(BT, "EMB_DIR", tmp_path)
    rng = np.random.default_rng(3)
    keys = p["keydevid"].to_list()
    signal = p["target"].to_numpy()
    for v in (2012, 2024):
        emb = rng.normal(size=(len(keys), 40)).astype(np.float32)
        if v == 2024:
            emb[:, 0] += 20 * signal          # the future-aware encoder knows the outcome
        pl.DataFrame({"keydevid": keys, "embedding": emb.tolist()}).write_parquet(
            tmp_path / f"ecalls-ret-{v}1231.parquet")
    specs = [BT.Spec("baselines", (BT.Block("base"),)),
             BT.Spec("ecalls-ret", (BT.Block("emb", "ecalls-ret"),)),
             BT.Spec("ecalls-ret-leaky", (BT.Block("emb", "ecalls-ret", True),))]
    df, info = BT.run_year(p, 2013, specs)
    assert info["specs"]["ecalls-ret"]["vintages"] == ["ecalls-ret-20121231"]
    assert info["specs"]["ecalls-ret-leaky"]["vintages"] == ["ecalls-ret-20241231"]
    assert info["specs"]["ecalls-ret-leaky"]["lookahead"] is True
    assert info["specs"]["ecalls-ret"]["lookahead"] is False
    ic = {s.name: np.corrcoef(df[s.name], df["target"])[0, 1] for s in specs}
    assert ic["ecalls-ret-leaky"] > 0.5 > abs(ic["ecalls-ret"])
    for s in specs:
        c = info["specs"][s.name]
        assert c["short_cutoff"] < c["long_cutoff"]


def test_a_call_missing_an_embedding_is_dropped_from_every_specification(tmp_path, monkeypatch):
    p = _panel()
    monkeypatch.setattr(BT, "EMB_DIR", tmp_path)
    keys = p["keydevid"].to_list()
    rng = np.random.default_rng(4)
    pl.DataFrame({"keydevid": keys[:-5], "embedding": rng.normal(size=(len(keys) - 5, 40)).astype(np.float32).tolist()}
                 ).write_parquet(tmp_path / "ecalls-ret-20121231.parquet")
    specs = [BT.Spec("baselines", (BT.Block("base"),)),
             BT.Spec("ecalls-ret", (BT.Block("emb", "ecalls-ret"),))]
    df, info = BT.run_year(p, 2013, specs)
    assert info["n_dropped_missing_feature"] == 5
    assert df.height == 300 - 5


def test_daily_legs_go_long_and_short_by_the_cutoffs():
    T = 10
    R = {1: np.full(T, 0.01), 2: np.full(T, -0.02), 3: np.full(T, 0.5)}
    m = np.zeros(T)
    df = pl.DataFrame({"permno": [1, 2, 3], "day0": [dt.date(2013, 1, 2)] * 3,
                       "entry_idx": [2, 2, 2], "exit_idx": [5, 7, 5], "logcap": [0.0, 0.0, 0.0],
                       "p": [1.0, -1.0, 0.0]})
    legs = BT.daily_legs(df, "p", {2013: (0.5, -0.5)}, R, m, T)
    assert np.allclose(legs["l_ew"][2:6], 0.01) and np.isnan(legs["l_ew"][6])
    assert np.allclose(legs["s_ew"][2:8], -0.02)
    assert legs["n_long"][3] == 1 and legs["n_short"][3] == 1   # the middle call trades nothing


def test_a_delisted_stock_earns_the_market_in_its_leg():
    T = 6
    R = {1: np.array([0, 0, 0.01, np.nan, np.nan, 0])}
    m = np.full(T, 0.003)
    df = pl.DataFrame({"permno": [1], "day0": [dt.date(2013, 1, 2)], "entry_idx": [2],
                       "exit_idx": [4], "logcap": [0.0], "p": [1.0]})
    legs = BT.daily_legs(df, "p", {2013: (0.5, -0.5)}, R, m, T)
    assert np.allclose(legs["l_ew"][2:5], [0.01, 0.003, 0.003])


def test_r2_oos_is_one_for_a_perfect_prediction_and_zero_for_the_training_mean():
    df = pl.DataFrame({"target": [1.0, 2.0, 3.0], "train_mean": [1.5] * 3,
                       "good": [1.0, 2.0, 3.0], "mean": [1.5] * 3})
    assert BT.r2_oos(df, "good") == 1.0
    assert BT.r2_oos(df, "mean") == 0.0


def test_a_target_known_earlier_trains_on_more_calls():
    """D27's volatility target is known on day +21, the return on day +61, so a December call can
    train January's refit for the first and not the second."""
    p = _panel().with_columns(known_date=pl.col("day0") + dt.timedelta(days=30))
    tr_known, _ = BT.split_year(p, 2013)
    tr_exit, _ = BT.split_year(p.drop("known_date"), 2013)
    assert tr_known.height > tr_exit.height
    assert tr_known["known_date"].max() < dt.date(2013, 1, 1)


def test_the_increment_pairs_text_with_its_own_controls():
    q = pl.DataFrame({"quarter": [20131, 20132, 20133, 20134], "ic": [0.1, 0.1, 0.1, 0.1]})
    q0 = q.with_columns(ic=pl.lit(0.04))
    ic = {"baselines": q0, "ecalls-ret+baselines": q}
    specs = [BT.Spec("baselines"), BT.Spec("ecalls-ret+baselines")]
    inc = BT.increments(ic, specs)
    assert list(inc) == ["ecalls-ret+baselines minus baselines"]
    assert inc["ecalls-ret+baselines minus baselines"]["mean"] == pytest.approx(0.06)


def test_the_residual_fit_finds_text_that_explains_what_the_baselines_miss(tmp_path, monkeypatch):
    """Post hoc D27 check: a text block carrying the part of the target the baselines miss must add
    to them in the residual fit."""
    from ecalls.portfolio import posthoc as PH
    rng = np.random.default_rng(5)
    p = _panel().with_columns(sue_cur=pl.Series(rng.normal(size=900)))
    hidden = rng.normal(size=900)
    p = p.with_columns(target=pl.col("sue_cur") + pl.Series(hidden) + pl.Series(rng.normal(size=900) * 0.3))
    monkeypatch.setattr(BT, "EMB_DIR", tmp_path)
    emb = rng.normal(size=(900, 40)).astype(np.float32)
    emb[:, 0] += 3 * hidden
    pl.DataFrame({"keydevid": p["keydevid"], "embedding": emb.tolist()}).write_parquet(
        tmp_path / "ecalls-ret-20121231.parquet")
    df, _ = PH.residual_year(p, 2013, [BT.Block("emb", "ecalls-ret")], ("sue_cur",))
    c_base = np.corrcoef(df["baselines"], df["target"])[0, 1]
    c_both = np.corrcoef(df["baselines then ecalls-ret on residuals"], df["target"])[0, 1]
    assert c_both > c_base + 0.2


def test_the_adjacent_block_reads_the_same_year_vintage_and_counts_as_lookahead():
    assert BT.vintage_for("ecalls-ret", 2017, False) == "ecalls-ret-20161231"
    assert BT.vintage_for("ecalls-ret", 2017, True) == "ecalls-ret-20241231"
    assert BT.vintage_for("ecalls-ret", 2017, False, adjacent=True) == "ecalls-ret-20171231"
    b = BT.Block("emb", "ecalls-ret", adjacent=True)
    assert b.name == "ecalls-ret-adjacent"
    assert BT.Spec(b.name, (b,)).lookahead is True


def test_views_name_their_vector_and_the_leak_stays_on_all_chunks():
    names = [s.name for s in BT.specifications(("all", "qa", "pres"))]
    assert "ecalls-ret-qa+identity+baselines" in names and "chrono-bert-v1-pres" in names
    assert "ecalls-ret-leaky" in names and "ecalls-ret-qa-leaky" not in names
    assert BT.Block("emb", "ecalls-ret", view="qa").name == "ecalls-ret-qa"


def test_increments_cover_every_text_specification_against_its_own_controls():
    q = pl.DataFrame({"quarter": [20131, 20132, 20133, 20134], "ic": [0.1] * 4})
    names = ["baselines", "identity+baselines", "ecalls-ret-qa+baselines", "ecalls-ret-qa+identity+baselines"]
    ic = {n: q for n in names}
    inc = BT.increments(ic, [BT.Spec(n) for n in names])
    assert set(inc) == {"ecalls-ret-qa+baselines minus baselines",
                        "ecalls-ret-qa+identity+baselines minus identity+baselines"}


def test_the_daily_long_short_ranks_only_calls_entering_that_close():
    rng = np.random.default_rng(0)
    rows = []
    for d, n in ((dt.date(2014, 1, 6), 30), (dt.date(2014, 1, 7), 10)):
        for i in range(n):
            rows.append({"day0": d, "known_date": d + dt.timedelta(days=1), "p": float(i), "target": float(i) / 100})
    df = pl.DataFrame(rows)
    ls = BT.daily_long_short(df, "p")
    assert ls.height == 1                                   # the 10-call day is not traded
    top, bottom = [i / 100 for i in range(30) if i >= np.quantile(range(30), 0.9)], \
                  [i / 100 for i in range(30) if i <= np.quantile(range(30), 0.1)]
    assert ls["ls"][0] == pytest.approx(np.mean(top) - np.mean(bottom))
    assert ls["earned_on"][0] == dt.date(2014, 1, 7)


def test_lexical_specs_read_no_embedding():
    names = {s.name for s in BT.lexical_specs()}
    assert {"baselines", "identity+baselines", "tone+baselines", "tfidf+baselines"} <= names
    assert all(b.kind != "emb" for s in BT.lexical_specs() for b in s.blocks)


def test_the_quarter_target_reads_the_full_universe_vectors_and_trains_after_day_61():
    t = BT.TARGETS["ret60"]
    assert t.panel == "quarter" and t.known == "exit_date" and t.base_cols == BT.BASE_COLS
    assert t.emb_dir == BT.EMB_DIR_ALL and t.feat_dir == BT.TF.FEAT_DIR_ALL
    assert BT.TARGETS["ret"].emb_dir == BT.EMB_DIR and BT.TARGETS["ret1d"].emb_dir == BT.EMB_DIR_ALL


def test_the_chain_enters_a_target_only_when_its_vectors_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(BT, "EMB_DIR", tmp_path / "sp")
    monkeypatch.setattr(BT, "EMB_DIR_ALL", tmp_path / "all")
    (tmp_path / "sp").mkdir()
    (tmp_path / "all").mkdir()
    assert BT.families_for(BT.TARGETS["sue"]) == BT.SP500_FAMILIES
    assert "ecalls-chain-ret" not in BT.families_for(BT.TARGETS["ret1d"])
    (tmp_path / "all" / "ecalls-chain-ret-20241231.parquet").touch()
    assert "ecalls-chain-ret" in BT.families_for(BT.TARGETS["ret1d"])
    assert "ecalls-chain-ret" not in BT.families_for(BT.TARGETS["sue"])
    (tmp_path / "sp" / "ecalls-chain-ret-20241231.parquet").touch()
    assert "ecalls-chain-ret" in BT.families_for(BT.TARGETS["sue"])
    assert not any("chain" in s.name for s in BT.specifications())
