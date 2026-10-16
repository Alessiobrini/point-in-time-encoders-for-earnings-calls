"""Edge cases for the completeness verdicts. Pure, no database and no artifacts."""
import datetime as dt
import sys

import pandas as pd
import pytest

from ecalls.audit.completeness import classify

D = dt.date


def _expected(pairs):
    return pd.DataFrame(pairs, columns=["keydevid", "chosen_transcriptid"])


def test_a_present_call_is_kept():
    out = classify(_expected([(1, 100)]), present={1}, failed_tx=set())
    assert out.verdict.tolist() == ["KEPT"]
    assert out.reason.tolist() == [""]


def test_an_absent_call_whose_batch_failed_is_dropped_with_a_reason():
    out = classify(_expected([(1, 100)]), present=set(), failed_tx={100})
    assert out.verdict.tolist() == ["DROPPED"]
    assert out.reason.iloc[0] == "pull batch failed after retries"


def test_an_absent_call_with_no_logged_failure_is_unexplained():
    """The verdict that must never occur. It means a call vanished with no record of why."""
    out = classify(_expected([(1, 100)]), present=set(), failed_tx=set())
    assert out.verdict.tolist() == ["UNEXPLAINED"]


def test_present_wins_over_a_logged_failure():
    """A batch can fail, get retried by a later run, and land. Present is ground truth."""
    out = classify(_expected([(1, 100)]), present={1}, failed_tx={100})
    assert out.verdict.tolist() == ["KEPT"]


def test_the_three_verdicts_partition_the_expected_set():
    exp = _expected([(1, 100), (2, 200), (3, 300), (4, 400)])
    out = classify(exp, present={1, 2}, failed_tx={300})
    assert len(out) == 4
    assert dict(zip(out.keydevid, out.verdict)) == {
        1: "KEPT", 2: "KEPT", 3: "DROPPED", 4: "UNEXPLAINED"}
    assert set(out.verdict) <= {"KEPT", "DROPPED", "UNEXPLAINED"}


def test_a_failure_on_a_different_transcript_does_not_excuse_this_call():
    """The failure log is matched on transcriptid, so an unrelated failure cannot launder a gap."""
    out = classify(_expected([(1, 100)]), present=set(), failed_tx={999})
    assert out.verdict.tolist() == ["UNEXPLAINED"]


def test_empty_expected_set_yields_empty_audit():
    out = classify(_expected([]), present=set(), failed_tx=set())
    assert out.empty


def test_nothing_is_dropped_from_the_expected_set():
    exp = _expected([(i, i * 100) for i in range(1, 51)])
    out = classify(exp, present=set(range(1, 26)), failed_tx=set())
    assert len(out) == 50
    assert set(out.keydevid) == set(exp.keydevid)


if __name__ == "__main__":
    print("run with pytest")
    sys.exit(0)
