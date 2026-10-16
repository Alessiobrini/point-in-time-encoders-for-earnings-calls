"""Edge-case tests for point-in-time membership.

These are pure, with no database. The live date-range correctness is validated against WRDS by
ecalls.ingest.verify_measurements.
"""
import datetime as dt
import sys

import pandas as pd
import pytest

from ecalls.ingest.universe_sp500 import (
    build_spells, coverage_horizon, ever_members, is_member, member_windows, members_on,
)

D = dt.date
CORPUS_END = D(2026, 12, 31)

# (permno, indno, mbrstartdt, mbrenddt, mbrflg, indfam)
# 10001 held one closed spell. 10002 left and came back, which is the re-addition case.
# 10003 and 10004 are still members, so their ends equal the cut date and both carry forward.
# Two of them, not one: a real coverage horizon is shared by many spells at once (503 in the live
# table), and a lone spell ending last is the final removal instead. See the regression test below.
CUT = D(2025, 12, 31)
ROWS = [
    (10001, 1000500, D(2000, 1, 1), D(2010, 6, 30), "NORM", 1100500),
    (10002, 1000500, D(2001, 1, 1), D(2011, 3, 15), "NORM", 1100500),
    (10002, 1000500, D(2019, 9, 1), D(2022, 4, 30), "NORM", 1100500),
    (10003, 1000500, D(2005, 5, 2), CUT,            "NORM", 1100500),
    (10004, 1000500, D(2012, 8, 8), CUT,            "NORM", 1100500),
]


@pytest.fixture
def memb():
    return build_spells(ROWS, CORPUS_END)


def test_dtypes_survive_assembly(memb):
    assert memb["permno"].dtype == "int32"
    assert all(isinstance(v, dt.date) for v in memb["mbr_start"])


def test_horizon_is_computed_not_hardcoded(memb):
    assert coverage_horizon(memb) == CUT


def test_is_open_set_exactly_on_the_cut(memb):
    assert memb.loc[memb.permno.isin([10003, 10004]), "is_open"].all()
    assert not memb.loc[memb.permno == 10001, "is_open"].any()


def test_carryforward_extends_only_open_spells_and_preserves_the_raw_end(memb):
    open_row = memb.loc[memb.permno == 10003].iloc[0]
    assert open_row["mbr_end"] == CORPUS_END
    assert open_row["mbr_end_raw"] == CUT          # the source value is never overwritten
    assert bool(open_row["carryforward"])
    closed = memb.loc[memb.permno == 10001].iloc[0]
    assert closed["mbr_end"] == closed["mbr_end_raw"] == D(2010, 6, 30)
    assert not bool(closed["carryforward"])


def test_members_on_is_inclusive_at_both_endpoints(memb):
    assert 10001 in members_on(memb, D(2000, 1, 1))     # first day
    assert 10001 in members_on(memb, D(2010, 6, 30))    # last day
    assert 10001 not in members_on(memb, D(1999, 12, 31))
    assert 10001 not in members_on(memb, D(2010, 7, 1))


def test_re_added_firm_is_a_non_member_in_the_gap(memb):
    """The survivorship logic itself. A firm out of the index between two spells must not be a
    member there, which is what a daily panel built by row absence would get wrong."""
    assert is_member(memb, 10002, D(2005, 1, 1))        # first spell
    assert not is_member(memb, 10002, D(2015, 1, 1))    # the gap
    assert is_member(memb, 10002, D(2020, 1, 1))        # second spell
    assert len(member_windows(memb, 10002)) == 2


def test_ever_members_unions_spells_without_duplicates(memb):
    assert ever_members(memb) == {10001, 10002, 10003, 10004}


def test_members_on_raises_past_the_horizon_without_carryforward():
    """A silently empty universe is the failure this guard exists to prevent. The guard now compares
    against the furthest date the frame covers rather than the coverage horizon, because
    carry-forward pushes mbr_end past the horizon and used to silence it."""
    closed_only = build_spells(ROWS[:3], CORPUS_END)    # no spell reaches the cut
    assert not closed_only["carryforward"].any()
    with pytest.raises(ValueError, match="past the furthest date"):
        members_on(closed_only, D(2026, 6, 30))


def test_carried_forward_frame_answers_a_2026_date(memb):
    assert members_on(memb, D(2026, 6, 30)) == {10003, 10004}


def test_coverage_horizon_rejects_a_disagreeing_frame(memb):
    bad = memb.copy()
    bad.loc[bad.index[0], "crsp_cut_date"] = D(2024, 1, 1)
    with pytest.raises(ValueError, match="not unique"):
        coverage_horizon(bad)


def test_empty_source_raises():
    with pytest.raises(ValueError, match="no rows"):
        build_spells([], CORPUS_END)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"run with pytest ({len(tests)} tests; fixtures are required)")
    sys.exit(0)


def test_a_lone_final_removal_is_not_a_coverage_horizon():
    """Regression: build_spells once treated the latest end date as the horizon unconditionally,
    so a single last removal was marked open and carried forward to CORPUS_END, leaving a removed
    firm in the index indefinitely. A real horizon is shared by many spells at once."""
    lone = build_spells(ROWS[:3], CORPUS_END)          # only 10002's 2022 exit is the max
    assert not lone["is_open"].any()
    assert not lone["carryforward"].any()
    assert lone.loc[lone.permno == 10002, "mbr_end"].max() == D(2022, 4, 30)
    assert coverage_horizon(lone) == D(2022, 4, 30)    # still reported, still the last known date


def test_many_spells_sharing_the_max_are_open():
    shared = ROWS[:2] + [
        (20001, 1000500, D(2003, 1, 1), CUT, "NORM", 1100500),
        (20002, 1000500, D(2004, 1, 1), CUT, "NORM", 1100500),
    ]
    memb = build_spells(shared, CORPUS_END)
    assert set(memb.loc[memb.is_open, "permno"]) == {20001, 20002}
    assert members_on(memb, D(2026, 6, 30)) == {20001, 20002}


def test_members_on_refuses_a_date_past_what_the_frame_covers(memb):
    """Regression, and the worst latent bug found in audit. CORPUS_END is baked from the wall-clock
    year at build time, so an artifact built in 2026 carries mbr_end = 2026-12-31. The old guard only
    checked the coverage horizon and was silenced by carry-forward having been applied, so
    members_on(2027-01-01) returned an empty set with no error. Every call in the corpus would have
    been labelled a non-member from 1 January 2027, silently."""
    furthest = max(memb["mbr_end"])
    assert len(members_on(memb, furthest)) > 0
    with pytest.raises(ValueError, match="past the furthest date"):
        members_on(memb, furthest + dt.timedelta(days=1))
