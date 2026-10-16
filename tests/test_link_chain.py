"""Edge-case tests for the link chain. Pure, no database."""
import datetime as dt
import sys

import pandas as pd
import pytest

from ecalls.config import constants as C
from ecalls.ingest.link_chain import (
    FAR_FUTURE, FAR_PAST, _as_date, intersect_windows, reconcile_compustat,
)

D = dt.date
CUT = D(2025, 12, 31)
CORPUS_END = D(2026, 12, 31)


def _memb(rows):
    return pd.DataFrame(rows, columns=["permno", "gvkey", "mbr_start", "mbr_end",
                                       "is_open", "crsp_cut_date"])


def test_null_end_is_open_on_the_right():
    assert _as_date(pd.Series([None]), fill=FAR_FUTURE).iloc[0] == FAR_FUTURE


def test_null_start_is_open_on_the_left():
    """Regression. 137,542 of 144,806 rows in ciq_common.wrds_gvkey carry a null startdate.
    Filling only the end silently cut the universe from 985 permnos to 91 while looking fine."""
    assert _as_date(pd.Series([None]), fill=FAR_PAST).iloc[0] == FAR_PAST


def test_undated_link_survives_the_intersection():
    """The same regression at the function level: a link with no dates at all must match."""
    left = pd.DataFrame([{"permno": 1, "mbr_start": D(2010, 1, 1), "mbr_end": D(2015, 1, 1)}])
    right = pd.DataFrame([{"permno": 1, "link_start": FAR_PAST, "link_end": FAR_FUTURE}])
    out = intersect_windows(left, right, "permno")
    assert len(out) == 1
    assert out.iloc[0]["win_start"] == D(2010, 1, 1)
    assert out.iloc[0]["win_end"] == D(2015, 1, 1)


def test_non_overlapping_windows_are_dropped():
    left = pd.DataFrame([{"permno": 1, "mbr_start": D(2010, 1, 1), "mbr_end": D(2012, 1, 1)}])
    right = pd.DataFrame([{"permno": 1, "link_start": D(2015, 1, 1), "link_end": D(2018, 1, 1)}])
    assert intersect_windows(left, right, "permno").empty


def test_intersection_is_the_overlap_not_the_union():
    left = pd.DataFrame([{"permno": 1, "mbr_start": D(2010, 1, 1), "mbr_end": D(2020, 1, 1)}])
    right = pd.DataFrame([{"permno": 1, "link_start": D(2012, 1, 1), "link_end": D(2015, 1, 1)}])
    out = intersect_windows(left, right, "permno").iloc[0]
    assert (out["win_start"], out["win_end"]) == (D(2012, 1, 1), D(2015, 1, 1))


def test_touching_windows_overlap_on_a_single_day():
    left = pd.DataFrame([{"permno": 1, "mbr_start": D(2010, 1, 1), "mbr_end": D(2012, 1, 1)}])
    right = pd.DataFrame([{"permno": 1, "link_start": D(2012, 1, 1), "link_end": D(2018, 1, 1)}])
    out = intersect_windows(left, right, "permno")
    assert len(out) == 1 and out.iloc[0]["win_start"] == out.iloc[0]["win_end"] == D(2012, 1, 1)


def test_one_permno_to_two_companyids_keeps_both():
    """Measured fan-out is real, not an error: 985 permnos resolve to 990 companyids."""
    left = pd.DataFrame([{"permno": 1, "mbr_start": D(2010, 1, 1), "mbr_end": D(2020, 1, 1)}])
    right = pd.DataFrame([
        {"permno": 1, "link_start": FAR_PAST, "link_end": D(2015, 1, 1)},
        {"permno": 1, "link_start": D(2015, 1, 2), "link_end": FAR_FUTURE},
    ])
    assert len(intersect_windows(left, right, "permno")) == 2


def test_compustat_only_addition_becomes_a_spell_with_a_null_permno():
    uni = _memb([(10, "000001", D(2000, 1, 1), CORPUS_END, True, CUT)])
    comp = pd.DataFrame([{"gvkey": "000001", "from_date": D(1999, 1, 1)},
                         {"gvkey": "999999", "from_date": D(2026, 3, 2)}])
    out = reconcile_compustat(uni, comp, CORPUS_END)
    new = out.loc[out["gvkey"] == "999999"].iloc[0]
    assert pd.isna(new["permno"])
    assert new["mbr_start"] == D(2026, 3, 2)
    assert new["mbr_end"] == CORPUS_END
    assert new["source"] == C.TBL_SP500_COMP
    assert bool(new["carryforward"])


def test_a_firm_missing_from_compustat_is_flagged_not_truncated():
    """D3: truncating on a guess would reintroduce survivorship bias, so it keeps its end date."""
    uni = _memb([(10, "000001", D(2000, 1, 1), CORPUS_END, True, CUT)])
    out = reconcile_compustat(uni, pd.DataFrame(columns=["gvkey", "from_date"]), CORPUS_END)
    row = out.iloc[0]
    assert bool(row["possibly_removed"])
    assert row["mbr_end"] == CORPUS_END          # not truncated


def test_a_closed_spell_is_never_flagged_possibly_removed():
    uni = _memb([(10, "000001", D(2000, 1, 1), D(2009, 5, 5), False, CUT)])
    out = reconcile_compustat(uni, pd.DataFrame(columns=["gvkey", "from_date"]), CORPUS_END)
    assert not bool(out.iloc[0]["possibly_removed"])


def test_empty_right_side_yields_empty_not_an_exception():
    left = pd.DataFrame([{"permno": 1, "mbr_start": D(2010, 1, 1), "mbr_end": D(2020, 1, 1)}])
    assert intersect_windows(left, pd.DataFrame(columns=["permno"]), "permno").empty


if __name__ == "__main__":
    print("run with pytest")
    sys.exit(0)


def test_compustat_only_additions_get_a_companyid_so_they_are_not_inert():
    """Regression. The Compustat-only spells were appended after the companyid join, so they carried
    a null companyid. index_label() in consolidate.py matches calls through companyid, so those
    spells labelled zero calls and D3's correction did no work at all. All 15 such gvkeys in the real
    data do have a companyid, holding 724 earnings calls between them."""
    uni = _memb([(10, "000001", D(2000, 1, 1), CORPUS_END, True, CUT)])
    comp = pd.DataFrame([{"gvkey": "999999", "from_date": D(2026, 3, 2)}])
    out = reconcile_compustat(uni, comp, CORPUS_END, gvkey_to_cid={"999999": 4242})
    new = out.loc[out["gvkey"] == "999999"].iloc[0]
    assert new["companyid"] == 4242
    assert pd.isna(new["permno"])


def test_a_compustat_gvkey_with_no_known_companyid_stays_null_rather_than_raising():
    uni = _memb([(10, "000001", D(2000, 1, 1), CORPUS_END, True, CUT)])
    comp = pd.DataFrame([{"gvkey": "999999", "from_date": D(2026, 3, 2)}])
    out = reconcile_compustat(uni, comp, CORPUS_END, gvkey_to_cid={})
    assert pd.isna(out.loc[out["gvkey"] == "999999", "companyid"].iloc[0])


def test_possibly_removed_ignores_a_row_whose_link_window_closed_long_ago():
    """Regression. `is_open` describes the membership spell, but a row is one
    (permno, gvkey, companyid, window) tuple whose CCM link may have ended decades earlier. Flagging
    on the spell alone asserted that Moody's might have been dropped from the index after 2025, on
    two rows whose links ended in 1998 and 2000."""
    uni = pd.DataFrame([
        {"permno": 48506, "gvkey": "004094", "mbr_start": D(1979, 12, 27), "mbr_end": CORPUS_END,
         "is_open": True, "crsp_cut_date": CUT, "win_end": D(1998, 6, 30)},
        {"permno": 48506, "gvkey": "139665", "mbr_start": D(1979, 12, 27), "mbr_end": CORPUS_END,
         "is_open": True, "crsp_cut_date": CUT, "win_end": CORPUS_END},
    ])
    out = reconcile_compustat(uni, pd.DataFrame(columns=["gvkey", "from_date"]), CORPUS_END)
    got = dict(zip(out["gvkey"], out["possibly_removed"]))
    assert got["004094"] is False or got["004094"] == False  # noqa: E712 - link closed in 1998
    assert got["139665"] is True or got["139665"] == True     # noqa: E712 - link still live
