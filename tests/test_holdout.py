"""Edge cases for the held-out split.

Its whole value is stability. A split that moves between processes or between vintages would let a call
be trained on in one run and evaluated in the next, and the contamination would look like success on the
one number Phase 4 steers by.
"""
import subprocess
import sys

import pytest

from ecalls.train import holdout as H


def test_the_split_is_stable_across_processes():
    """Python's `hash` is randomized per process for strings, so a split built on it would reassign
    calls between runs. This is the test that would catch a switch to `hash`."""
    code = ("from ecalls.train import holdout as H;"
            "print([int(H.is_holdout(k)) for k in range(200)])")
    a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}).stdout
    b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONHASHSEED": "1", "PATH": "/usr/bin:/bin"}).stdout
    assert a and a == b
    assert a.strip() == str([int(H.is_holdout(k)) for k in range(200)])


def test_the_fraction_is_approximately_honored():
    n = 50_000
    held = sum(H.is_holdout(k) for k in range(n))
    assert abs(held / n - H.DEFAULT_FRACTION) < 0.005


def test_a_larger_fraction_is_a_superset_of_a_smaller_one():
    """Nesting means the evaluation set can be widened without retraining, since no call moves from
    held out to trained on."""
    small = {k for k in range(5_000) if H.is_holdout(k, 0.05)}
    big = {k for k in range(5_000) if H.is_holdout(k, 0.20)}
    assert small and small < big


def test_zero_holds_nothing_and_the_fraction_is_bounded():
    assert not any(H.is_holdout(k, 0.0) for k in range(1_000))
    with pytest.raises(ValueError, match="fraction must be in"):
        H.is_holdout(1, 1.0)
    with pytest.raises(ValueError, match="fraction must be in"):
        H.is_holdout(1, -0.1)


def test_changing_the_salt_reassigns_calls():
    a = {k for k in range(5_000) if H.is_holdout(k, salt="one")}
    b = {k for k in range(5_000) if H.is_holdout(k, salt="two")}
    assert a != b
    assert len(a & b) < len(a)          # they overlap only by chance, not by construction


def test_split_ids_partitions_without_loss_or_duplication():
    ids = list(range(1, 3_000))
    train, held = H.split_ids(ids)
    assert len(train) + len(held) == len(ids)
    assert not set(train) & set(held)
    assert sorted(train + held) == ids


def test_split_ids_preserves_input_order():
    ids = [99, 7, 4242, 13, 8]
    train, held = H.split_ids(ids)
    assert train == [i for i in ids if not H.is_holdout(i)]
    assert held == [i for i in ids if H.is_holdout(i)]


def test_the_rule_is_recorded_for_a_later_reader():
    d = H.describe()
    assert d["holdout_salt"] == H.SALT
    assert d["holdout_fraction"] == H.DEFAULT_FRACTION
    assert "sha1" in d["holdout_rule"]


def test_a_string_id_and_an_int_id_agree():
    """Call ids arrive as numpy ints from parquet and as Python ints from tests."""
    import numpy as np
    assert H.is_holdout(np.int64(12345)) == H.is_holdout(12345)
