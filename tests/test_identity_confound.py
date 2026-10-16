"""Edge cases for the identity diagnostic's pure parts. No database, no corpus."""
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ecalls.diagnostics.identity_confound import mask_identity, similarity_gap


def test_the_ticker_pass_does_not_eat_the_name_placeholder():
    """Regression. An uppercase " NAME " placeholder written before the ticker pass was itself
    matched by \\b[A-Z]{2,5}\\b and rewritten as TICKER, collapsing names and tickers into one token."""
    out = mask_identity("Travelers grew.", "Travelers Inc", [])
    assert "__name__" in out
    assert "TICKER" not in out.replace("__ticker__", "")


def test_company_and_speaker_names_are_masked():
    out = mask_identity("Travelers reported growth. Alan Schnitzer said so.",
                        "The Travelers Companies, Inc.", ["Alan Schnitzer"])
    assert "Travelers" not in out and "Schnitzer" not in out and "Alan" not in out
    assert "__name__" in out


def test_short_and_generic_name_tokens_are_not_masked():
    """Masking 'Inc' or 'Group' would gut every call's text rather than its identity."""
    out = mask_identity("The group grew and the company reported.", "Acme Group Inc", [])
    assert "group" in out.lower() and "company" in out.lower()


def test_ticker_shaped_tokens_are_masked():
    out = mask_identity("Our EPS beat and AAPL moved.", "Nothing", [])
    assert "AAPL" not in out and "__ticker__" in out


def test_ordinary_lowercase_words_survive_masking():
    text = "revenue increased across the segment on strong demand"
    assert mask_identity(text, "Nothing", []) == text


def test_non_string_input_returns_empty():
    assert mask_identity(None, "X", []) == ""
    assert mask_identity(float("nan"), "X", []) == ""


def test_similarity_gap_is_positive_when_same_company_calls_are_identical():
    """Two companies, two identical calls each, orthogonal across companies. The gap must be 1."""
    X = csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))
    r = similarity_gap(X, np.array([7, 7, 9, 9]))
    assert r["within_mean"] == pytest.approx(1.0)
    assert r["cross_mean"] == pytest.approx(0.0)
    assert r["gap"] == pytest.approx(1.0)
    assert r["within_n"] == 2          # one pair per company


def test_similarity_gap_is_zero_when_identity_carries_no_information():
    """All calls identical, so within and cross similarity must coincide and the gap vanish."""
    X = csr_matrix(np.ones((6, 3)) / np.sqrt(3))
    r = similarity_gap(X, np.array([1, 1, 2, 2, 3, 3]))
    assert r["gap"] == pytest.approx(0.0, abs=1e-9)


def test_within_pair_count_is_the_combinatorial_count():
    X = csr_matrix(np.eye(6))
    r = similarity_gap(X, np.array([1, 1, 1, 2, 2, 2]))
    assert r["within_n"] == 6          # 2 * C(3,2)


if __name__ == "__main__":
    print("run with pytest")
    sys.exit(0)
