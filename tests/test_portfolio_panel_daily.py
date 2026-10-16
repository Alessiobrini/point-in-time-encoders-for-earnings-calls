"""D28's next-day clock and targets. The entry rule is where a look-ahead would hide: a position
entered at a close the call had not finished by would trade on text nobody had yet heard."""
import datetime as dt

import numpy as np
import pytest

from ecalls.portfolio import panel_daily as PD

# Wed 2024-01-03, Thu 01-04, Fri 01-05, Mon 01-08, Tue 01-09
CAL = np.array(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"], dtype="datetime64[D]")
THU, SAT = dt.date(2024, 1, 4), dt.date(2024, 1, 6)


@pytest.mark.parametrize("day, start, want", [
    (THU, dt.time(7, 0), (1, 0, 1)),     # before the open: enter Thursday's close, session Thursday
    (THU, dt.time(10, 0), (1, 0, -1)),   # during the session: enter Thursday's close
    (THU, dt.time(15, 0), (1, 0, -1)),   # 15:00 still finishes by the close
    (THU, dt.time(15, 30), (2, 0, -1)),  # would not finish by 16:00: enter Friday's close
    (THU, dt.time(17, 0), (2, 1, 2)),    # after the close: reference is Thursday's close
    (SAT, dt.time(10, 0), (3, 2, -1)),   # weekend: enter Monday, reference Friday
])
def test_the_clock(day, start, want):
    assert PD.clock(CAL, day, start) == want


def _series(T=300, r=0.0):
    return (np.full(T, r), np.zeros(T), np.full(T, 1e6), np.full(T, 20.0), np.full(T, 19.0))


def test_the_next_day_target_is_the_day_after_entry_minus_the_market():
    R, m, cap, prc, opn = _series()
    R[261] = 0.05
    m[261] = 0.01
    ev = PD.event(R, m, cap, prc, opn, entry=260, ref=259, session=-1)
    assert ev["target_1d"] == pytest.approx(0.04)


def test_baselines_never_read_after_entry():
    R, m, cap, prc, opn = _series(r=0.001)
    base = PD.event(R.copy(), m, cap.copy(), prc, opn, entry=260, ref=259, session=-1)
    R2, cap2 = R.copy(), cap.copy()
    R2[261:] = 0.4
    cap2[260:] = 1e12                       # the reference close is 259, so 260 on must not count
    ev = PD.event(R2, m, cap2, prc, opn, entry=260, ref=259, session=-1)
    for k in ("reaction", "rev5", "mom", "logcap"):
        assert ev[k] == pytest.approx(base[k]), k
    assert ev["target_1d"] != pytest.approx(base["target_1d"])


def test_the_reaction_runs_from_the_reference_close_to_the_entry_close():
    R, m, cap, prc, opn = _series()
    R[260], R[261] = 0.10, 0.02
    ev = PD.event(R, m, cap, prc, opn, entry=261, ref=259, session=-1)
    assert ev["reaction"] == pytest.approx(1.10 * 1.02 - 1)


def test_open_to_close_uses_the_session_after_the_call():
    R, m, cap, prc, opn = _series()
    ev = PD.event(R, m, cap, prc, opn, entry=260, ref=259, session=260)
    assert ev["target_o2c"] == pytest.approx(20.0 / 19.0 - 1)
    assert np.isnan(PD.event(R, m, cap, prc, opn, entry=260, ref=259, session=-1)["target_o2c"])


def test_a_missing_return_the_day_after_entry_drops_the_call():
    R, m, cap, prc, opn = _series()
    R[261] = np.nan
    assert PD.event(R, m, cap, prc, opn, entry=260, ref=259, session=-1) is None


def test_the_common_stock_filter():
    ok = {"sharetype": "NS", "securitytype": "EQTY", "securitysubtype": "COM", "primaryexch": "Q"}
    assert PD.is_common(ok)
    assert not PD.is_common({**ok, "securitysubtype": "ADR"})
    assert not PD.is_common({**ok, "primaryexch": "X"})


def test_the_overnight_gap_is_known_at_the_open_and_excludes_the_session():
    R, m, cap, prc, opn = _series()
    prc[259], opn[260], prc[260] = 20.0, 22.0, 30.0
    ev = PD.event(R, m, cap, prc, opn, entry=260, ref=259, session=260)
    assert ev["gap"] == pytest.approx(22.0 / 20.0 - 1)
    assert ev["target_o2c"] == pytest.approx(30.0 / 22.0 - 1)
