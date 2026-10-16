"""D28's full-universe link: a permno is used only when the call date is inside both link windows."""
import datetime as dt

import polars as pl

from ecalls.ingest import link_all as LA

D = dt.date


def _calls(rows):
    return pl.DataFrame(rows, schema={"keydevid": pl.Int64, "companyid": pl.Int64, "call_date": pl.Date},
                        orient="row")


def _cid(rows):
    return pl.DataFrame(rows, schema={"companyid": pl.Int64, "gvkey": pl.String, "cid_start": pl.Date,
                                      "cid_end": pl.Date, "primaryflag": pl.Int32}, orient="row")


def _ccm(rows):
    return pl.DataFrame(rows, schema={"gvkey": pl.String, "permno": pl.Int32, "link_start": pl.Date,
                                      "link_end": pl.Date, "linkprim": pl.String, "linktype": pl.String},
                        orient="row")


def test_a_call_outside_the_permno_link_window_is_not_linked():
    """The S&P panel's 17 mislinked permnos were calls dated outside their listed life."""
    calls = _calls([(1, 10, D(2011, 5, 1)), (2, 10, D(2018, 5, 1))])
    cid = _cid([(10, "G", None, None, 1)])
    ccm = _ccm([("G", 555, D(2017, 9, 1), None, "P", "LU")])
    out = LA.resolve(calls, cid, ccm)
    assert out["keydevid"].to_list() == [2]


def test_a_call_outside_the_companyid_window_is_not_linked():
    calls = _calls([(1, 10, D(2015, 5, 1))])
    cid = _cid([(10, "G", D(2016, 1, 1), None, 1)])
    ccm = _ccm([("G", 555, None, None, "P", "LU")])
    assert LA.resolve(calls, cid, ccm).height == 0


def test_the_primary_link_wins_and_the_call_is_flagged():
    calls = _calls([(1, 10, D(2015, 5, 1))])
    cid = _cid([(10, "G", None, None, 1)])
    ccm = _ccm([("G", 777, None, None, "C", "LC"), ("G", 555, None, None, "P", "LU")])
    out = LA.resolve(calls, cid, ccm)
    assert out["permno"].to_list() == [555]
    assert out["link_ambiguous"].to_list() == [True]


def test_open_window_ends_are_unbounded():
    calls = _calls([(1, 10, D(2025, 5, 1))])
    cid = _cid([(10, "G", None, None, 0)])
    ccm = _ccm([("G", 555, D(2000, 1, 1), None, "P", "LU")])
    out = LA.resolve(calls, cid, ccm)
    assert out.height == 1 and out["link_ambiguous"].to_list() == [False]
