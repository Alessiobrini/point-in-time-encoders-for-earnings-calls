"""The point-in-time guard on the premise test's features, and the call pooling it reads.

Both fail quietly if wrong: a too-new model leaks the test period into the features and inflates the
very number the test measures, and a pooling bug changes every row by the same rule so nothing looks
off.
"""
import datetime as dt
import importlib.util
import pathlib

import numpy as np
import polars as pl
import pytest

from ecalls.diagnostics import predictive_premise as PP


def frame(rows):
    return pl.DataFrame({"model_id": [r[0] for r in rows],
                         "call_date": [dt.date.fromisoformat(r[1]) for r in rows]})


CUT = dt.date(2022, 2, 10)


def test_per_call_resolution_passes():
    train = frame([("v2016", "2017-04-25"), ("v2020", "2021-07-30")])
    test = frame([("v2021", "2022-05-12")])
    PP.check_point_in_time(train, test, CUT, {"v2016": "2016-12-31", "v2020": "2020-12-31",
                                              "v2021": "2021-12-31"})


def test_one_vintage_older_than_the_cut_passes_on_every_row():
    """D11: at the decision date one model embeds the whole history, including training rows it saw."""
    train = frame([("v2021", "2015-01-20"), ("v2021", "2021-11-02")])
    test = frame([("v2021", "2022-05-12"), ("v2021", "2024-10-30")])
    PP.check_point_in_time(train, test, CUT, {"v2021": "2021-12-31"})


def test_a_vintage_newer_than_the_cut_is_refused_on_training_rows():
    train = frame([("v2022", "2015-01-20")])
    test = frame([("v2021", "2022-05-12")])
    with pytest.raises(ValueError, match="newer than the moment"):
        PP.check_point_in_time(train, test, CUT, {"v2022": "2022-12-31", "v2021": "2021-12-31"})


def test_a_test_row_embedded_by_a_model_newer_than_its_call_is_refused():
    train = frame([("v2021", "2021-11-02")])
    test = frame([("v2023", "2023-05-01")])
    with pytest.raises(ValueError, match="newer than the moment"):
        PP.check_point_in_time(train, test, CUT, {"v2021": "2021-12-31", "v2023": "2023-12-31"})


def test_a_model_with_no_recorded_cutoff_is_refused():
    with pytest.raises(ValueError):
        PP.check_point_in_time(frame([("x", "2015-01-20")]), frame([]).cast(
            {"call_date": pl.Date, "model_id": pl.String}), CUT, {})


def test_the_boundary_is_inclusive_as_in_for_date():
    PP.check_point_in_time(frame([("v", "2015-01-20")]), frame([("v", "2022-05-12")]),
                           dt.date(2021, 12, 31), {"v": "2021-12-31"})


def _call_embeddings():
    p = pathlib.Path(__file__).parent.parent / "scripts" / "call_embeddings.py"
    spec = importlib.util.spec_from_file_location("call_embeddings", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_call_vector_is_the_normalized_mean_of_its_chunks():
    ce = _call_embeddings()
    v = np.array([[3.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    got = np.array(ce.call_vector(v))
    want = np.array([4 / 3, 2 / 3]) / np.linalg.norm([4 / 3, 2 / 3])
    assert np.allclose(got, want, atol=1e-6)
    assert np.isclose(np.linalg.norm(got), 1.0)


def test_a_zero_mean_call_vector_is_returned_rather_than_divided_by_zero():
    ce = _call_embeddings()
    got = ce.call_vector(np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32))
    assert got == [0.0, 0.0]


def _panel(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = [dt.date(2015, 1, 1) + dt.timedelta(days=int(d)) for d in rng.integers(0, 3000, n)]
    return pl.DataFrame({"keydevid": np.arange(n), "call_date": dates,
                         "ticker": rng.choice(["A", "B", "C"], n)})


def test_the_split_returns_rows_in_time_order_whatever_order_it_was_given():
    """`TimeSeriesSplit` reads row order as time order, so an unsorted training set makes the
    penalty search cross-validate on ticker blocks instead of on the past."""
    p = _panel().sort("ticker")
    train, test, cut, gap = PP.time_split(p)
    assert train["call_date"].is_sorted() and test["call_date"].is_sorted()
    assert train["call_date"].max() < cut <= gap <= test["call_date"].min()


def test_the_split_does_not_depend_on_the_input_order():
    p = _panel()
    a = PP.time_split(p)
    b = PP.time_split(p.sample(fraction=1.0, shuffle=True, seed=3))
    assert a[0].equals(b[0]) and a[1].equals(b[1])


def test_pooling_by_call_matches_the_per_call_mean_and_splits_sections():
    ce = _call_embeddings()
    V = np.array([[1, 0], [0, 1], [3, 0], [0, 2], [1, 1]], dtype=np.float32)
    kd = np.array([7, 9, 7, 9, 7])
    sec = np.array(["qa", "presentation", "presentation", "qa", "qa"])
    out = ce.pool_by_call(V, kd, sec)
    n, v_all, v_qa, v_pres = out[7]
    assert n == 3
    assert np.allclose(v_all, ce.call_vector(V[[0, 2, 4]]))
    assert np.allclose(v_qa, ce.call_vector(V[[0, 4]]))
    assert np.allclose(v_pres, ce.call_vector(V[[2]]))
    assert out[9][2] is not None and np.allclose(out[9][3], ce.call_vector(V[[1]]))


def test_a_call_without_a_section_has_no_vector_for_it():
    ce = _call_embeddings()
    out = ce.pool_by_call(np.ones((2, 2), dtype=np.float32), np.array([1, 1]), np.array(["qa", "qa"]))
    assert out[1][3] is None
