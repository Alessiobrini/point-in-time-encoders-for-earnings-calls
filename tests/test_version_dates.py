import datetime as dt

import polars as pl
import pytest

from ecalls.paper.version_dates import late_shares, trainable
from ecalls.train.holdout import is_holdout

D = dt.date


def frame(rows):
    return pl.DataFrame(rows, schema={"call_date": pl.Date, "created_date": pl.Date, "trainable": pl.Boolean},
                        orient="row")


def test_counts_only_trainable_calls_on_or_before_the_cutoff():
    calls = frame([
        (D(2015, 3, 1), D(2015, 3, 2), True),      # in the window, created early
        (D(2015, 12, 20), D(2016, 1, 5), True),    # in the window, created after the cutoff: late
        (D(2015, 12, 30), D(2015, 12, 31), True),  # created on the cutoff day itself: not late
        (D(2015, 12, 31), D(2016, 2, 1), True),    # dated on the cutoff, created after: late
        (D(2015, 6, 1), D(2016, 6, 1), False),     # held out: outside every window
        (D(2016, 1, 1), D(2016, 1, 2), True),      # after the cutoff: outside the window
    ])
    e = late_shares(calls, years=[2015])["2015"]
    assert (e["trainable_calls"], e["dated"], e["late"], e["undated"]) == (4, 4, 2, 0)
    assert e["share_late"] == pytest.approx(0.5)


def test_windows_grow_with_the_cutoff():
    calls = frame([
        (D(2015, 12, 20), D(2016, 1, 5), True),    # late for 2015, not for 2016
        (D(2016, 5, 1), D(2017, 1, 1), True),      # only in the 2016 window, late there
        (D(2016, 5, 2), D(2016, 5, 3), True),
    ])
    s = late_shares(calls, years=[2015, 2016])
    assert (s["2015"]["trainable_calls"], s["2015"]["late"]) == (1, 1)
    assert (s["2016"]["trainable_calls"], s["2016"]["late"]) == (3, 1)
    assert s["2016"]["share_late"] == pytest.approx(1 / 3)


def test_undated_version_is_left_out_of_the_share():
    calls = frame([
        (D(2015, 1, 1), None, True),
        (D(2015, 1, 2), D(2016, 1, 1), True),
        (D(2015, 1, 3), D(2015, 1, 3), True),
    ])
    e = late_shares(calls, years=[2015])["2015"]
    assert (e["trainable_calls"], e["dated"], e["undated"], e["late"]) == (3, 2, 1, 1)
    assert e["share_late"] == pytest.approx(0.5)


def test_empty_window_has_no_share():
    calls = frame([(D(2016, 1, 1), D(2016, 1, 2), True)])
    e = late_shares(calls, years=[2015])["2015"]
    assert (e["trainable_calls"], e["late"], e["share_late"]) == (0, 0, None)


def test_trainable_follows_the_holdout_rule():
    ids = list(range(1, 400))
    flags = trainable(pl.Series(ids)).to_list()
    assert flags == [not is_holdout(k) for k in ids]
    assert 0 < flags.count(False) < len(ids)       # the range holds both kinds of call
