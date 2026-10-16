"""Splitting the D28 embedding tasks into shard parts and joining them back."""
import importlib.util
from pathlib import Path

import polars as pl
import pytest

_p = Path(__file__).resolve().parents[1] / "scripts" / "pf_embeddings.py"
_spec = importlib.util.spec_from_file_location("pf_embeddings", _p)
PE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PE)

THROUGH = "2013-12-31"


def _calls():
    shards = [f"{y}Q{q}" for y in (2011, 2012) for q in (1, 2, 3, 4)]
    return pl.DataFrame({"keydevid": list(range(16)), "shard": [s for s in shards for _ in (0, 1)]})


def _part(keys):
    return pl.DataFrame({"keydevid": keys, "embedding": [[0.0, 1.0]] * len(keys)})


def _meta(n, missing=0, through=THROUGH):
    return {"through": through, "n_calls_requested": n, "n_calls_embedded": n - missing, "n_missing": missing}


def test_parts_cover_every_call_once():
    calls = _calls()
    parts = [PE.select_part(calls, k, 3) for k in range(3)]
    keys = [k for p in parts for k in p["keydevid"].to_list()]
    assert sorted(keys) == list(range(16))
    # a call never splits from its shard, so pooling per part equals pooling in one pass
    for p in parts:
        assert set(p["shard"]) & set().union(*(set(q["shard"]) for q in parts if q is not p)) == set()


def test_parts_are_dealt_round_robin():
    p0 = PE.select_part(_calls(), 0, 3)
    assert sorted(set(p0["shard"])) == ["2011Q1", "2011Q4", "2012Q3"]


@pytest.mark.parametrize("part,parts", [(3, 3), (-1, 3)])
def test_part_outside_range_raises(part, parts):
    with pytest.raises(ValueError):
        PE.select_part(_calls(), part, parts)


def test_more_parts_than_shards_raises():
    with pytest.raises(ValueError):
        PE.select_part(_calls(), 0, 9)


def test_merge_joins_complete_parts():
    df = PE.merge_parts({1, 2, 3}, [_part([3, 1]), _part([2])], [_meta(2), _meta(1)], THROUGH)
    assert df["keydevid"].to_list() == [1, 2, 3]


def test_merge_rejects_a_missing_call():
    with pytest.raises(ValueError, match="miss 1"):
        PE.merge_parts({1, 2, 3}, [_part([1]), _part([2])], [_meta(1), _meta(1)], THROUGH)


def test_merge_rejects_a_call_outside_the_universe():
    with pytest.raises(ValueError, match="outside"):
        PE.merge_parts({1, 2}, [_part([1, 9]), _part([2])], [_meta(2), _meta(1)], THROUGH)


def test_merge_rejects_a_duplicate_call():
    with pytest.raises(ValueError, match="more than one part"):
        PE.merge_parts({1, 2}, [_part([1, 2]), _part([2])], [_meta(2), _meta(1)], THROUGH)


def test_merge_rejects_a_short_part_even_if_the_union_matches():
    with pytest.raises(ValueError, match="1 missing"):
        PE.merge_parts({1, 2}, [_part([1]), _part([2])], [_meta(2, missing=1), _meta(1)], THROUGH)


def test_merge_rejects_a_part_from_another_horizon():
    with pytest.raises(ValueError, match="through"):
        PE.merge_parts({1, 2}, [_part([1]), _part([2])], [_meta(1, through="2012-12-31"), _meta(1)], THROUGH)


def test_covers_reads_coverage_not_existence(tmp_path):
    m = tmp_path / "x.json"
    assert not PE._covers(m, THROUGH, 5)
    m.write_text('{"through": "2013-12-31", "n_calls_embedded": 4, "n_missing": 0}')
    assert not PE._covers(m, THROUGH, 5)
    m.write_text('{"through": "2013-12-31", "n_calls_embedded": 5, "n_missing": 0}')
    assert PE._covers(m, THROUGH, 5)
