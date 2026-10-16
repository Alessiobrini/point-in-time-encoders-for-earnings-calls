import polars as pl
import pytest

from ecalls.paper.chance_facts import chance


def test_chance_is_mean_inverse_pool_over_questions():
    answers = pl.DataFrame({"keydevid": [1, 1, 1, 1, 2, 2]})
    queries = pl.DataFrame({"keydevid": [1, 2, 2]})
    # pools of 4 and 2: (1/4 + 1/2 + 1/2) / 3, which weights calls by their questions
    assert chance(queries, answers) == pytest.approx((0.25 + 0.5 + 0.5) / 3)


def test_question_without_pool_raises():
    with pytest.raises(ValueError):
        chance(pl.DataFrame({"keydevid": [3]}), pl.DataFrame({"keydevid": [1]}))
