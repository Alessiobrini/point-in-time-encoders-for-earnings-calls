"""The portfolio test's event clock and target. Every failure here leaks the future quietly: a
control that reads a day after +1, an after-close call placed on the call date, or a delisted
stock whose window keeps compounding."""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from ecalls.portfolio import panel as PN


def test_an_after_close_call_is_after_the_close_in_new_york_time():
    d, after = PN.local_event_date(dt.date(2024, 2, 1), dt.time(21, 30))      # 16:30 EST
    assert d == dt.date(2024, 2, 1) and after
    d, after = PN.local_event_date(dt.date(2024, 7, 1), dt.time(19, 0))       # 15:00 EDT
    assert d == dt.date(2024, 7, 1) and not after


def test_a_utc_date_after_midnight_belongs_to_the_previous_new_york_day():
    d, after = PN.local_event_date(dt.date(2024, 2, 2), dt.time(3, 0))        # 22:00 EST on Feb 1
    assert d == dt.date(2024, 2, 1) and after


CAL = np.array(["2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06"], dtype="datetime64[D]")


@pytest.mark.parametrize("date, after, want", [
    (dt.date(2024, 2, 1), False, 0),      # a trading day before the close
    (dt.date(2024, 2, 1), True, 1),       # after the close: next trading day
    (dt.date(2024, 2, 2), True, 2),       # Friday after the close: Monday
    (dt.date(2024, 2, 3), False, 2),      # Saturday: Monday
    (dt.date(2024, 2, 6), True, -1),      # past the end of the calendar
])
def test_day_zero(date, after, want):
    assert PN.day0_index(CAL, date, after) == want


def _series(T=400, r=0.0):
    return np.full(T, r), np.zeros(T), np.full(T, 1e9)


def test_the_target_is_the_window_return_minus_the_market():
    R, m, cap = _series(r=0.01)
    m[:] = 0.002
    ev = PN.event_row(R, m, cap, 300)
    assert ev["target"] == pytest.approx(1.01 ** 60 - 1.002 ** 60)
    assert ev["ear"] == pytest.approx(1.01 ** 3 - 1.002 ** 3)
    assert (ev["entry_idx"], ev["exit_idx"]) == (302, 361)


def test_controls_never_read_day_two_or_later():
    R, m, cap = _series(r=0.001)
    base = PN.event_row(R.copy(), m, cap.copy(), 300)
    R2, cap2 = R.copy(), cap.copy()
    R2[302:] = 0.5
    cap2[299:] = 1e15                       # day -1 onward must not set the cap
    ev = PN.event_row(R2, m, cap2, 300)
    for k in ("ear", "mom", "rev", "logcap"):
        assert ev[k] == pytest.approx(base[k]), k
    assert ev["target"] != pytest.approx(base["target"])


def test_after_a_delisting_the_window_earns_the_market():
    R, m, cap = _series(r=0.0)
    m[:] = 0.001
    R[310] = -0.9                           # the delisting return, on day +10
    R[311:] = np.nan
    ev = PN.event_row(R, m, cap, 300)
    want = 0.1 * (1.001 ** 51) - 1 - (1.001 ** 60 - 1)     # days +2 to +9 earn the stock's own 0
    assert ev["target"] == pytest.approx(want)
    assert ev["n_target_days_filled"] == 51


def test_a_call_with_no_trade_at_the_event_is_dropped():
    R, m, cap = _series()
    R[299:301] = np.nan
    assert PN.event_row(R, m, cap, 300) is None


def test_a_window_past_the_data_or_without_history_is_dropped():
    R, m, cap = _series(T=400)
    assert PN.event_row(R, m, cap, 350) is None      # day +61 beyond the end
    assert PN.event_row(R, m, cap, 100) is None      # no 252 days of history


def test_momentum_needs_enough_history():
    R, m, cap = _series()
    R[300 - 252: 300 - 60] = np.nan
    assert np.isnan(PN.event_row(R, m, cap, 300)["mom"])


# --- D27, abnormal volatility -----------------------------------------------------------------------

def test_abnormal_volatility_is_the_log_ratio_of_post_to_pre_variance():
    R, m, cap = _series()
    R[300 - 21: 300 - 1] = 0.01            # days -21 to -2
    R[302: 322] = 0.02                     # days +2 to +21
    ev = PN.event_row(R, m, cap, 300)
    assert ev["av_target"] == pytest.approx(np.log(4.0))
    assert ev["lrv_month"] == pytest.approx(np.log(1e-4))
    assert ev["av_known_idx"] == 321


def test_volatility_baselines_never_read_day_two_or_later():
    R, m, cap = _series(r=0.001)
    m[:] = 0.002
    base = PN.event_row(R.copy(), m.copy(), cap, 300)
    R2, m2 = R.copy(), m.copy()
    R2[302:] = 0.3
    m2[302:] = 0.3
    ev = PN.event_row(R2, m2, cap, 300)
    for k in ("lrv_week", "lrv_month", "lrv_year", "mkt_lrv_month", "abs_ear"):
        assert ev[k] == pytest.approx(base[k]), k
    assert ev["av_target"] != pytest.approx(base["av_target"])


def test_a_window_with_too_few_returns_has_no_volatility_target():
    R, m, cap = _series(r=0.01)
    R[302: 302 + 8] = np.nan               # 12 of 20 post days left
    assert np.isnan(PN.event_row(R, m, cap, 300)["av_target"])


def test_reproduction_reports_an_exact_match_and_ignores_calls_outside_the_sp500():
    import datetime as dt
    d = dt.date(2015, 3, 2)
    q = pl.DataFrame({"keydevid": [1, 2, 3], "day0": [d, d, d], "target": [0.01, -0.02, 0.5]})
    s = pl.DataFrame({"keydevid": [1, 2], "day0": [d, d], "target": [0.01, -0.02]})
    r = PN.reproduction(q, s)
    assert r == {"n_common": 2, "n_sp500": 2, "max_abs_target_diff": 0.0, "n_day0_differs": 0}


def test_reproduction_reports_a_different_target_and_a_different_day0():
    import datetime as dt
    q = pl.DataFrame({"keydevid": [1, 2], "day0": [dt.date(2015, 3, 2), dt.date(2015, 3, 3)],
                      "target": [0.01, -0.02]})
    s = pl.DataFrame({"keydevid": [1, 2], "day0": [dt.date(2015, 3, 2), dt.date(2015, 3, 2)],
                      "target": [0.01, -0.0201]})
    r = PN.reproduction(q, s)
    assert abs(r["max_abs_target_diff"] - 0.0001) < 1e-12 and r["n_day0_differs"] == 1
