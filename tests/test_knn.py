"""Nearest-neighbor forecasting from call vectors (D30)."""
import numpy as np

from ecalls.portfolio import knn as K

D = lambda s: np.datetime64(s, "D")


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _pool():
    vec = _unit([[1, 0], [0.9, 0.1], [0, 1], [0.7, 0.7]])
    known = np.array([D("2015-01-02"), D("2015-01-05"), D("2015-01-02"), D("2015-01-09")])
    y = np.array([1.0, 2.0, -5.0, 10.0])
    firm = np.array([1, 2, 3, 4])
    return vec, known, y, firm


def test_k1_takes_the_most_similar_eligible_call():
    vec, known, y, firm = _pool()
    f = K.knn_forecast(vec, known, y, firm, _unit([[1, 0]]), np.array([D("2015-01-06")]),
                       np.array([9]), ks=(1, 2))
    assert f[1][0] == 1.0 and f[2][0] == 1.5


def test_a_call_known_on_the_event_day_is_not_in_the_pool():
    """The pool is what was known strictly before the event day."""
    vec, known, y, firm = _pool()
    f = K.knn_forecast(vec, known, y, firm, _unit([[0.7, 0.7]]), np.array([D("2015-01-09")]),
                       np.array([9]), ks=(1,))
    assert f[1][0] != 10.0
    g = K.knn_forecast(vec, known, y, firm, _unit([[0.7, 0.7]]), np.array([D("2015-01-10")]),
                       np.array([9]), ks=(1,))
    assert g[1][0] == 10.0


def test_the_own_firm_is_excluded_only_when_asked():
    vec, known, y, firm = _pool()
    q, day = _unit([[1, 0]]), np.array([D("2015-01-06")])
    assert K.knn_forecast(vec, known, y, firm, q, day, np.array([1]), ks=(1,))[1][0] == 2.0
    assert K.knn_forecast(vec, known, y, firm, q, day, np.array([1]), ks=(1,), exclude_same_firm=False)[1][0] == 1.0


def test_a_small_pool_uses_all_of_it_and_an_empty_pool_gives_nan():
    vec, known, y, firm = _pool()
    f = K.knn_forecast(vec, known, y, firm, _unit([[1, 0], [1, 0]]),
                       np.array([D("2015-01-03"), D("2015-01-02")]), np.array([9, 9]), ks=(50,))
    assert f[50][0] == -2.0          # the two calls known on 2015-01-02, mean of 1 and -5
    assert np.isnan(f[50][1])


def test_an_excluded_firm_can_leave_fewer_than_k_neighbors():
    vec, known, y, firm = _pool()
    firm = np.array([7, 7, 7, 4])
    f = K.knn_forecast(vec, known, y, firm, _unit([[1, 0]]), np.array([D("2015-01-06")]),
                       np.array([7]), ks=(3,))
    assert np.isnan(f[3][0])         # the only other firm's call is known after the event day


def test_the_pool_order_does_not_matter():
    rng = np.random.default_rng(0)
    vec = _unit(rng.normal(size=(200, 8)))
    known = D("2015-01-01") + rng.integers(0, 100, 200).astype("timedelta64[D]")
    y, firm = rng.normal(size=200), rng.integers(0, 30, 200)
    q = _unit(rng.normal(size=(20, 8)))
    qd = D("2015-01-01") + rng.integers(20, 120, 20).astype("timedelta64[D]")
    qf = rng.integers(0, 30, 20)
    a = K.knn_forecast(vec, known, y, firm, q, qd, qf, ks=(5, 50))
    p = rng.permutation(200)
    b = K.knn_forecast(vec[p], known[p], y[p], firm[p], q, qd, qf, ks=(5, 50))
    for k in (5, 50):
        assert np.allclose(a[k], b[k], equal_nan=True)


def test_vintages_follow_d11():
    assert K.vintage("ecalls-ret", 2016, "pit") == "ecalls-ret-20151231"
    assert K.vintage("ecalls-ret", 2016, "leaky") == "ecalls-ret-20241231"
    assert K.vintage("ecalls-ret", 2016, "adjacent") == "ecalls-ret-20161231"
