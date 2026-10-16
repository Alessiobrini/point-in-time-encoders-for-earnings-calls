"""D27's earnings surprise: the seasonal difference, the split adjustment, the next quarter across a
fiscal year boundary, and the call-to-quarter match."""
import datetime as dt

import polars as pl
import pytest

from ecalls.portfolio import sue as SU


def _fq(rows):
    return pl.DataFrame(rows, schema=SU.FUNDQ_SCHEMA, orient="row")


def test_the_surprise_is_split_adjusted_and_seasonal():
    fq = _fq([
        ("A", dt.date(2019, 12, 31), 2019, 4, dt.date(2020, 1, 30), 2.00, 1.0, 100.0),
        ("A", dt.date(2020, 12, 31), 2020, 4, dt.date(2021, 1, 28), 1.50, 2.0, 60.0),   # 2-for-1 split
    ])
    s = SU.surprises(fq).filter(pl.col("fyearq") == 2020)
    # adjusted EPS 0.75 against 2.00 a year earlier, over adjusted price 30
    assert s["sue"][0] == pytest.approx((0.75 - 2.00) / 30.0)


def test_the_next_quarter_wraps_across_the_fiscal_year():
    fq = _fq([("A", dt.date(2019, 12, 31), 2019, 4, dt.date(2020, 1, 30), 1.0, 1.0, 50.0),
              ("A", dt.date(2020, 3, 31), 2020, 1, dt.date(2020, 4, 30), 2.0, 1.0, 50.0)])
    s = SU.surprises(fq)
    q4 = s.filter(pl.col("fqtr") == 4)
    assert q4["rdq_next"][0] == dt.date(2020, 4, 30)


def test_a_call_matches_the_release_it_accompanies_and_nothing_further():
    sq = pl.DataFrame({"gvkey": ["A", "A"], "datadate": [dt.date(2020, 3, 31), dt.date(2020, 6, 30)],
                       "fyearq": [2020, 2020], "fqtr": [1, 2],
                       "rdq": [dt.date(2020, 4, 30), dt.date(2020, 7, 30)],
                       "sue": [0.1, 0.2], "sue_next": [0.2, None], "rdq_next": [dt.date(2020, 7, 30), None]})
    calls = pl.DataFrame({"keydevid": [1, 2, 3], "gvkey": ["A", "A", "A"],
                          "call_date": [dt.date(2020, 5, 2), dt.date(2020, 5, 10), dt.date(2020, 7, 30)]})
    m = SU.match_calls(calls, sq).sort("keydevid")
    assert m["keydevid"].to_list() == [1, 3]           # 10 days from a release is not that release's call
    assert m.filter(pl.col("keydevid") == 1)["sue_next"][0] == 0.2
    assert m.filter(pl.col("keydevid") == 1)["sue_known_date"][0] == dt.date(2020, 7, 30)
