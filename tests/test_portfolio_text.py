"""The lexical baselines: the point-in-time dictionary and the tone arithmetic."""
import polars as pl
import pytest

from ecalls.portfolio import text_features as TF


@pytest.mark.parametrize("flag, year, want", [
    (2009, 2008, False), (2009, 2009, True), (2009, 2020, True),
    (-2020, 2019, True), (-2020, 2020, False), (0, 2020, False),
])
def test_dictionary_membership_follows_the_year_codes(flag, year, want):
    assert TF.in_category(flag, year) is want


def _lm():
    return pl.DataFrame({"Word": ["LOSS", "GAIN", "REFINANCE", "BEST"],
                         "Negative": ["2009", "0", "-2020", "0"],
                         "Positive": ["0", "2009", "0", "2020"]})


def test_a_word_added_later_does_not_score_earlier_text():
    lists = {2019: TF.word_lists(2019, _lm()), 2021: TF.word_lists(2021, _lm())}
    t = TF.tone_by_year("best results despite a loss and a refinance", lists)
    # 2019: negative LOSS and REFINANCE, no positive. 2021: REFINANCE dropped, BEST added.
    assert t[2019] == pytest.approx(2 / 8)
    assert t[2021] == pytest.approx((1 - 1) / 8)


def test_tone_is_case_insensitive_and_counts_all_words_in_the_denominator():
    lists = {2020: TF.word_lists(2020, _lm())}
    assert TF.tone_by_year("Gain gain LOSS ok", lists)[2020] == pytest.approx((1 - 2) / 4)


def test_empty_text_has_no_tone():
    import math
    assert math.isnan(TF.tone_by_year("123 456", {2020: TF.word_lists(2020, _lm())})[2020])
