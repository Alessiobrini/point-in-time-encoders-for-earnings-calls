"""Edge cases for the embedding registry. Pure, no models and no embeddings needed."""
import datetime as dt
import json
import sys

import pytest

from ecalls.chunk.registry import ATEMPORAL, VINTAGE, Entry, Registry, load

D = dt.date


def v(mid, cutoff, dim=768):
    return Entry(mid, VINTAGE, f"huggingface:test/{mid}", dim, training_cutoff=cutoff)


def a(mid, dim=4096):
    return Entry(mid, ATEMPORAL, f"huggingface:test/{mid}", dim, notes="no calendar range")


@pytest.fixture
def reg():
    return Registry([v("m2015", "2015-12-31"), v("m2020", "2020-12-31"),
                     v("m2024", "2024-12-31"), a("fin-e5")])


# --- resolving by date -------------------------------------------------------------------------

def test_for_date_returns_the_largest_cutoff_at_or_before_the_date(reg):
    assert reg.for_date("2018-06-30").model_id == "m2015"
    assert reg.for_date("2021-01-01").model_id == "m2020"
    assert reg.for_date("2026-09-26").model_id == "m2024"


def test_a_cutoff_exactly_equal_to_the_date_is_eligible(reg):
    """A model that saw text through 2020-12-31 may embed 2020-12-31 to predict what follows."""
    assert reg.for_date("2020-12-31").model_id == "m2020"


def test_the_day_before_a_cutoff_falls_to_the_previous_vintage(reg):
    assert reg.for_date("2020-12-30").model_id == "m2015"


def test_for_date_raises_rather_than_falling_back_to_the_earliest(reg):
    """Falling back would embed an early date with a model trained on later text, which is the
    leakage this resolver exists to prevent."""
    with pytest.raises(LookupError, match="no registered vintage has a cutoff at or before"):
        reg.for_date("2010-01-01")


def test_for_date_never_returns_an_atemporal_model():
    """A model whose cutoff is unknown cannot support a predictive claim. Several strong finance
    embedding models are in that position because they train on synthetic data."""
    only_atemporal = Registry([a("fin-e5"), a("bam")])
    with pytest.raises(LookupError, match="no vintage models are registered"):
        only_atemporal.for_date("2020-01-01")


def test_for_date_accepts_a_date_object_as_well_as_a_string(reg):
    assert reg.for_date(D(2018, 6, 30)).model_id == reg.for_date("2018-06-30").model_id


def test_vintages_are_ordered_by_cutoff_and_exclude_atemporal(reg):
    assert [e.model_id for e in reg.vintages()] == ["m2015", "m2020", "m2024"]


# --- by name -----------------------------------------------------------------------------------

def test_by_name_reaches_atemporal_models_for_retrieval(reg):
    assert reg.by_name("fin-e5").kind == ATEMPORAL


def test_by_name_on_an_unknown_id_lists_what_is_known(reg):
    with pytest.raises(KeyError, match="m2015"):
        reg.by_name("nope")


# --- validation --------------------------------------------------------------------------------

def test_a_vintage_without_a_cutoff_is_refused_at_construction():
    with pytest.raises(ValueError, match="needs a training_cutoff"):
        Entry("bad", VINTAGE, "src", 768)


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="kind must be"):
        Entry("bad", "whatever", "src", 768)


def test_a_malformed_cutoff_is_refused_at_construction_not_at_query_time():
    with pytest.raises(ValueError):
        Entry("bad", VINTAGE, "src", 768, training_cutoff="not-a-date")


def test_duplicate_model_ids_are_refused():
    with pytest.raises(ValueError, match="duplicate model_id"):
        Registry([v("m", "2020-01-01"), v("m", "2021-01-01")])


def test_register_refuses_to_overwrite_unless_asked(reg):
    with pytest.raises(ValueError, match="already registered"):
        reg.register(v("m2015", "2016-12-31"))
    reg.register(v("m2015", "2016-12-31"), replace=True)
    assert reg.by_name("m2015").training_cutoff == "2016-12-31"


# --- on disk -----------------------------------------------------------------------------------

def test_resolving_a_model_whose_embeddings_are_absent_names_the_path(reg):
    with pytest.raises(FileNotFoundError, match="not on this machine"):
        reg.path_by_name("m2015")


def test_round_trip_through_json_preserves_every_entry(reg, tmp_path):
    p = tmp_path / "embeddings.json"
    reg.save(p)
    back = load(p)
    assert len(back) == len(reg)
    assert [e.model_id for e in back.vintages()] == [e.model_id for e in reg.vintages()]
    assert back.by_name("fin-e5").kind == ATEMPORAL
    assert json.loads(p.read_text())[0]["model_id"] == "fin-e5"   # sorted on write


def test_loading_a_missing_registry_gives_an_empty_one(tmp_path):
    assert len(load(tmp_path / "nope.json")) == 0


if __name__ == "__main__":
    print("run with pytest")
    sys.exit(0)


# --- two families now share every cutoff --------------------------------------------------------

def _two_families():
    return Registry([
        v("chrono-bert-v1-20101231", "2010-12-31"),
        Entry("ecalls-ret-20101231", VINTAGE, "local:ecalls-ret-20101231", 384,
              training_cutoff="2010-12-31"),
        v("chrono-bert-v1-20151231", "2015-12-31"),
        Entry("ecalls-ret-20151231", VINTAGE, "local:ecalls-ret-20151231", 384,
              training_cutoff="2015-12-31"),
    ])


def test_a_tied_cutoff_is_refused_rather_than_resolved_by_sort_order():
    """Phase 5 compares our vintage against ChronoBERT's checkpoint of the same year, so both have
    the same cutoff. Returning the last eligible entry would let the registry pick the winner."""
    reg = _two_families()
    with pytest.raises(LookupError, match="share the cutoff"):
        reg.for_date("2016-03-15")


def test_the_refusal_names_the_candidates_and_how_to_choose():
    reg = _two_families()
    try:
        reg.for_date("2016-03-15")
    except LookupError as e:
        msg = str(e)
    assert "chrono-bert-v1-20151231" in msg and "ecalls-ret-20151231" in msg
    assert "family=" in msg


def test_a_family_resolves_the_tie_and_still_respects_the_cutoff():
    reg = _two_families()
    assert reg.for_date("2016-03-15", family="ecalls-ret").model_id == "ecalls-ret-20151231"
    assert reg.for_date("2016-03-15", family="chrono-bert-v1").model_id == "chrono-bert-v1-20151231"
    # the family must not let a too-new model through
    assert reg.for_date("2012-06-01", family="ecalls-ret").model_id == "ecalls-ret-20101231"


def test_a_family_with_no_vintage_before_the_date_still_raises_about_leakage():
    reg = _two_families()
    with pytest.raises(LookupError, match="cutoff at or before"):
        reg.for_date("2009-01-01", family="ecalls-ret")


def test_an_unknown_family_names_the_ones_that_exist():
    reg = _two_families()
    with pytest.raises(LookupError, match="Registered families"):
        reg.for_date("2016-03-15", family="not-a-family")


def test_one_family_per_cutoff_needs_no_family_argument():
    """The pre-2026-10-01 case must keep working, or every existing caller breaks."""
    reg = Registry([v("chrono-bert-v1-20101231", "2010-12-31"),
                    v("chrono-bert-v1-20151231", "2015-12-31")])
    assert reg.for_date("2016-03-15").model_id == "chrono-bert-v1-20151231"


def test_families_lists_the_prefixes():
    assert _two_families().families() == ["chrono-bert-v1", "ecalls-ret"]


# --- a date can arrive as several types ----------------------------------------------------------

def test_for_date_accepts_a_datetime_not_only_a_date():
    """Parquet stores a call date as a datetime, and comparing one to a date raises TypeError
    instead of coercing, so every caller reading a date out of an artifact hit this."""
    reg = Registry([v("chrono-bert-v1-20101231", "2010-12-31"),
                    v("chrono-bert-v1-20151231", "2015-12-31")])
    assert reg.for_date(dt.datetime(2016, 3, 15, 13, 45)).model_id == "chrono-bert-v1-20151231"
    assert reg.for_date(dt.date(2016, 3, 15)).model_id == "chrono-bert-v1-20151231"
    assert reg.for_date("2016-03-15").model_id == "chrono-bert-v1-20151231"
    assert reg.for_date("2016-03-15T00:00:00").model_id == "chrono-bert-v1-20151231"


def test_as_date_refuses_something_that_is_not_a_date():
    from ecalls.chunk.registry import as_date
    with pytest.raises(TypeError, match="cannot read a calendar date"):
        as_date(20160315)


def test_a_datetime_late_in_the_day_does_not_resolve_to_a_later_vintage():
    """Truncating to the date must not round up, or a call on the cutoff day itself would resolve to
    the next vintage and embed with a model that saw it."""
    from ecalls.chunk.registry import as_date
    assert as_date(dt.datetime(2015, 12, 31, 23, 59, 59)) == dt.date(2015, 12, 31)
    reg = Registry([v("chrono-bert-v1-20151231", "2015-12-31")])
    assert reg.for_date(dt.datetime(2015, 12, 31, 23, 59)).model_id == "chrono-bert-v1-20151231"


def test_require_point_in_time_accepts_a_cutoff_before_first_use_and_refuses_otherwise():
    reg = Registry([v("v-2010", "2010-12-31"), a("atemp")])
    assert reg.require_point_in_time("v-2010", "2011-01-01").model_id == "v-2010"
    assert reg.require_point_in_time("v-2010", dt.date(2011, 4, 1)).model_id == "v-2010"
    with pytest.raises(LookupError, match="not before"):
        reg.require_point_in_time("v-2010", "2010-12-31")     # the cutoff day itself is too early
    with pytest.raises(LookupError, match="not before"):
        reg.require_point_in_time("v-2010", "2010-06-30")
    with pytest.raises(LookupError, match="no cutoff"):
        reg.require_point_in_time("atemp", "2030-01-01")
    with pytest.raises(KeyError):
        reg.require_point_in_time("unregistered", "2030-01-01")


def test_every_point_in_time_rule_in_the_paper_passes_the_registry_check():
    """The shipped pairings: vintage Y for quarter (Y+1)Q2 in retrieval and RAG, the chain's 2010
    member, and vintage t-1 for forecast year t, for every family the paper reports."""
    from ecalls.diagnostics import rag_lookahead as RG
    from ecalls.diagnostics import seed_summary as S
    from ecalls.portfolio import backtest as BT
    if not load().vintages():
        pytest.skip("no registry file on this machine")
    for y in S.LOOKAHEAD_YEARS:
        for fam in S.LOOKAHEAD_FAMILIES:
            S.pit_vintage(fam, y)
        for fam in RG.FAMILIES:
            RG.vintage(fam, y)
    for t in BT.TEST_YEARS:
        for fam in ("ecalls-ret", "ecalls-abl", "ecalls-chain-ret", "chrono-bert-v1"):
            BT.vintage_for(fam, t, lookahead=False)
    # and the same check refuses the same-year vintage, which the adjacent arm uses on purpose
    from ecalls.chunk import registry as R
    with pytest.raises(LookupError):
        R.require_point_in_time("ecalls-ret-20131231", "2013-01-01")
