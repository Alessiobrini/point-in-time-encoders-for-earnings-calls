"""The restatement split of the look-ahead premium (D34)."""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from ecalls.diagnostics import restatement as RS

D = dt.date


def frames():
    test = pl.DataFrame({"keydevid": [1], "companyid": [10], "call_date": [D(2016, 5, 1)]})
    calls = pl.DataFrame({
        "keydevid": [2, 3, 4, 5, 6, 7, 8, 9],
        "companyid": [10, 10, 10, 10, 10, 10, 10, 99],
        "call_date": [D(2016, 2, 1), D(2016, 5, 1), D(2016, 8, 1), D(2016, 11, 1),
                      D(2017, 2, 1), D(2017, 5, 1), D(2017, 8, 1), D(2016, 8, 1)],
        "holdout": [False, False, False, True, False, False, False, False]})
    return test, calls


def test_later_calls_are_the_same_firms_trainable_calls_strictly_after_the_test_call_inside_the_window():
    test, calls = frames()
    got = RS.later_calls(test, calls, D(2016, 12, 31))["later"].to_list()
    # 2 is earlier, 3 is the same day, 5 is held out, 6 to 8 are after the window, 9 is another firm
    assert got == [4]


def test_later_calls_keep_only_the_next_k_in_date_order():
    test, calls = frames()
    got = RS.later_calls(test, calls, D(2024, 12, 31), k=2)["later"].to_list()
    assert got == [4, 6]


def test_terciles_have_equal_size_and_follow_the_score():
    s = np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2])
    t = RS.terciles(s)
    assert sorted(np.bincount(t).tolist()) == [2, 2, 2]
    assert t[np.argmax(s)] == 2 and t[np.argmin(s)] == 0


def test_a_score_compares_an_answer_only_with_its_own_firms_later_text():
    answers = ["margins expanded on pricing", "margins expanded on pricing"]
    later = ["margins expanded on pricing again", "weather hurt shipments"]
    sc = RS.max_cosine(answers, np.array([1, 2]), later, np.array([1, 2]))
    assert sc[0] > 0.5 and sc[1] < sc[0]


def test_an_answer_whose_firm_has_no_later_text_scores_zero():
    sc = RS.max_cosine(["margins expanded"], np.array([1]), ["margins expanded"], np.array([2]))
    assert sc[0] == 0.0


@pytest.mark.parametrize("n", [3, 7, 3000])
def test_terciles_cover_every_question_once(n):
    t = RS.terciles(np.random.default_rng(0).random(n))
    assert t.min() == 0 and t.max() == 2 and len(t) == n
