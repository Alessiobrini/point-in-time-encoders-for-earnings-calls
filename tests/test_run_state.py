"""Edge cases for the incremental refresh logic (D9). Pure, no database."""
import datetime as dt
import sys

import pandas as pd
import pytest

from ecalls.config import constants as C
from ecalls.ingest.run_state import watermark_cutoff, years_needing_reindex
from ecalls.ingest.transcript_index import merge_index


def test_no_prior_run_means_pull_everything():
    assert watermark_cutoff({}) is None
    assert watermark_cutoff({"max_creation_date": None}) is None


def test_cutoff_subtracts_the_margin_not_the_raw_watermark():
    """An exact cutoff would permanently skip rows WRDS loads after the run that set it, because
    transcriptcreationdate_utc is Capital IQ's clock and the snapshot trails it."""
    out = watermark_cutoff({"max_creation_date": "2026-09-13"}, margin_days=30)
    assert out == dt.date(2026, 8, 14)


def test_cutoff_uses_the_configured_default_margin():
    out = watermark_cutoff({"max_creation_date": "2026-09-13"})
    assert out == dt.date(2026, 9, 13) - dt.timedelta(days=C.WATERMARK_MARGIN_DAYS)


def test_a_zero_margin_is_honored_when_asked_for():
    assert watermark_cutoff({"max_creation_date": "2026-09-13"}, margin_days=0) == dt.date(2026, 9, 13)


def test_first_run_reindexes_nothing():
    assert years_needing_reindex({}, {2024: 100}) == []


def test_a_year_whose_count_moved_is_reindexed():
    state = {"counts_by_year": {"2023": 50, "2024": 100}}
    assert years_needing_reindex(state, {2023: 50, 2024: 101}) == [2024]


def test_a_year_that_lost_rows_is_also_reindexed():
    """A count can fall if the vendor withdraws a transcript. That is still drift worth catching."""
    state = {"counts_by_year": {"2024": 100}}
    assert years_needing_reindex(state, {2024: 99}) == [2024]


def test_a_brand_new_year_is_reindexed():
    state = {"counts_by_year": {"2024": 100}}
    assert years_needing_reindex(state, {2024: 100, 2025: 7}) == [2025]


def test_unchanged_counts_reindex_nothing():
    state = {"counts_by_year": {"2023": 50, "2024": 100}}
    assert years_needing_reindex(state, {2023: 50, 2024: 100}) == []


def _idx(rows):
    """Columns follow the real grain: (transcriptid, companyid), not transcriptid alone."""
    return pd.DataFrame(rows, columns=["transcriptid", "companyid", "keydevid", "collection_type"])


def test_merge_upserts_rather_than_appends():
    """A revision can restate an existing transcript row, so the fetched copy must win and the
    row must not be duplicated."""
    old = _idx([(1, 7, 100, "Edited Copy"), (2, 7, 100, "Proofed Copy")])
    new = _idx([(2, 7, 100, "Audited Copy"), (3, 7, 101, "Edited Copy")])
    out = merge_index(old, new)
    assert len(out) == 3
    assert not out.duplicated(["transcriptid", "companyid"]).any()
    assert out.loc[out.transcriptid == 2, "collection_type"].iloc[0] == "Audited Copy"
    assert out.loc[out.transcriptid == 1, "collection_type"].iloc[0] == "Edited Copy"


def test_merge_with_no_prior_index_returns_the_new_rows():
    new = _idx([(2, 7, 100, "x"), (1, 7, 100, "y")])
    out = merge_index(None, new)
    assert list(out["transcriptid"]) == [1, 2]      # sorted, deterministic


def test_merge_output_is_order_independent():
    old = _idx([(1, 7, 100, "a"), (2, 7, 100, "b")])
    new = _idx([(3, 7, 101, "c")])
    a = merge_index(old, new)
    b = merge_index(old.iloc[::-1].reset_index(drop=True), new)
    pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    print("run with pytest")
    sys.exit(0)


def test_a_transcript_shared_by_parent_and_subsidiary_keeps_both_rows():
    """A call covering a parent and its subsidiary carries one transcriptid under two companyids
    (Brasil Telecom / Oi, CVB Financial / Citizens Business Bank). Collapsing on transcriptid alone
    would drop one company's route to the call."""
    old = _idx([(30636, 402308, 62089210, "Edited Copy"),
                (30636, 876962, 62089210, "Edited Copy")])
    new = _idx([(30636, 402308, 62089210, "Audited Copy")])
    out = merge_index(old, new)
    assert len(out) == 2
    assert set(out["companyid"]) == {402308, 876962}
    assert out.loc[out.companyid == 402308, "collection_type"].iloc[0] == "Audited Copy"
    assert out.loc[out.companyid == 876962, "collection_type"].iloc[0] == "Edited Copy"
