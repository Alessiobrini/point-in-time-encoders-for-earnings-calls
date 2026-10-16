"""Edge cases for the embedding audit. Every failure mode is constructed, not hoped for.

The audit's job is to fail. A test suite that only builds a healthy embedding set and checks the audit
says OK would pass against an audit that always says OK, which is the shape of the `integrity_check`
defect found in phase 1: every check derived from the same glob, so a deleted shard was undetectable.
So each test here breaks one thing and asserts the audit names that one thing.
"""
import json

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from ecalls.audit import embeddings as A
from ecalls.chunk.registry import ATEMPORAL, VINTAGE, Entry, Registry
from ecalls.embed import encode as E


DIM = 4


@pytest.fixture
def entry():
    return Entry("stub-v1", VINTAGE, "huggingface:test/stub-v1", DIM,
                 training_cutoff="2024-12-31", normalized=True)


@pytest.fixture(autouse=True)
def stub_registry(monkeypatch, entry):
    """The audit resolves the model itself, so the real 26-vintage registry has to be replaced."""
    monkeypatch.setattr(A.R, "load", lambda *a, **k: Registry([entry]))


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A two-shard chunk layer with the paths redirected into tmp_path."""
    monkeypatch.setattr(A.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(A.P, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(A.P, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(A.P, "chunks_shard", lambda s: A.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    monkeypatch.setattr(E.P, "CHUNKS_DIR", A.P.CHUNKS_DIR)
    monkeypatch.setattr(E.P, "EMBEDDINGS_DIR", A.P.EMBEDDINGS_DIR)
    A.P.CHUNKS_DIR.mkdir(parents=True)
    ids = {"2015Q1": ["a", "b", "c"], "2015Q2": ["d", "e"]}
    for shard, cid in ids.items():
        pl.DataFrame({"chunk_id": cid, "text": ["x"] * len(cid)}).write_parquet(
            A.P.chunks_shard(shard))
    return ids


def write_part(shard, ids, vectors=None, model_id="stub-v1", dim=DIM):
    """A part file with whatever content the test needs to break."""
    if vectors is None:
        v = np.zeros((len(ids), dim), dtype=np.float32)
        v[:, 0] = 1.0                                   # unit norm
    else:
        v = np.asarray(vectors, dtype=np.float32)
    dst = E.part_path("stub-v1", shard)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(E.to_table(list(ids), v, meta={"model_id": model_id}), dst)
    return dst


# --- the healthy case, so the failure cases mean something ---------------------------------------

def test_a_complete_healthy_set_reports_ok(layout, entry):
    for shard, ids in layout.items():
        write_part(shard, ids)
    df = A.run("stub-v1")
    assert set(df["status"]) == {"OK"}
    assert int(df["n_missing"].sum()) == 0
    assert int(df["n_embedded"].sum()) == 5
    assert A.report(df, entry) == 0


# --- each failure mode, one at a time ------------------------------------------------------------

def test_a_shard_with_no_part_file_is_reported_not_skipped(layout, entry):
    """The failure that matters most: a shard nobody noticed was never written. Sixteen concurrent
    tasks on a preemptible partition is exactly how one goes missing."""
    write_part("2015Q1", layout["2015Q1"])
    df = A.run("stub-v1")
    row = df[df["shard"] == "2015Q2"].iloc[0]
    assert row["status"] == "NO_PART_FILE"
    assert row["n_missing"] == 2
    assert A.report(df, entry) == 1


def test_a_short_part_file_is_incomplete(layout, entry):
    write_part("2015Q1", ["a", "b"])                    # c is absent
    write_part("2015Q2", layout["2015Q2"])
    df = A.run("stub-v1")
    row = df[df["shard"] == "2015Q1"].iloc[0]
    assert row["status"] == "INCOMPLETE" and row["n_missing"] == 1
    assert A.report(df, entry) == 1


def test_a_superset_part_file_passes_but_is_flagged(layout, entry):
    """Extra rows still cover every expected chunk, so this is not a failure. It is worth surfacing
    because it means the chunk layer changed under an embedding set."""
    write_part("2015Q1", ["a", "b", "c", "zzz"])
    write_part("2015Q2", layout["2015Q2"])
    df = A.run("stub-v1")
    row = df[df["shard"] == "2015Q1"].iloc[0]
    assert row["status"] == "OK_WITH_EXTRA" and row["n_extra"] == 1 and row["n_missing"] == 0
    assert A.report(df, entry) == 0


def test_a_wrong_dimension_is_caught(layout, entry):
    write_part("2015Q1", layout["2015Q1"],
               vectors=np.tile([1.0, 0, 0, 0, 0, 0], (3, 1)), dim=6)
    write_part("2015Q2", layout["2015Q2"])
    df = A.run("stub-v1")
    assert df[df["shard"] == "2015Q1"].iloc[0]["status"] == "WRONG_DIM"
    assert A.report(df, entry) == 1


def test_a_wrong_model_id_in_the_file_is_caught(layout, entry):
    """Two embedding sets differing only by model is the one mix-up no consumer could detect."""
    write_part("2015Q1", layout["2015Q1"], model_id="some-other-model")
    write_part("2015Q2", layout["2015Q2"])
    df = A.run("stub-v1")
    assert df[df["shard"] == "2015Q1"].iloc[0]["status"] == "WRONG_MODEL"
    assert A.report(df, entry) == 1


def test_all_zero_rows_are_caught_although_the_counts_are_right(layout, entry):
    """A zero row passes every count-based check. Mean pooling over an all-padding mask makes one."""
    write_part("2015Q1", layout["2015Q1"], vectors=np.zeros((3, DIM)))
    write_part("2015Q2", layout["2015Q2"])
    df = A.run("stub-v1")
    row = df[df["shard"] == "2015Q1"].iloc[0]
    assert row["status"] == "ZERO_ROWS" and row["n_embedded"] == 3
    assert A.report(df, entry) == 1


def test_non_finite_vectors_are_caught(layout, entry):
    v = np.zeros((3, DIM), dtype=np.float32)
    v[:, 0] = 1.0
    v[1, 2] = np.nan
    write_part("2015Q1", layout["2015Q1"], vectors=v)
    write_part("2015Q2", layout["2015Q2"])
    df = A.run("stub-v1")
    assert df[df["shard"] == "2015Q1"].iloc[0]["status"] == "NON_FINITE"
    assert A.report(df, entry) == 1


def test_unnormalized_vectors_are_caught_for_a_normalized_entry(layout, entry):
    write_part("2015Q1", layout["2015Q1"], vectors=np.full((3, DIM), 3.0))
    write_part("2015Q2", layout["2015Q2"])
    df = A.run("stub-v1")
    assert df[df["shard"] == "2015Q1"].iloc[0]["status"] == "BAD_NORM"
    assert A.report(df, entry) == 1


def test_float16_rounding_alone_does_not_trip_the_norm_check(layout, entry):
    """The tolerance has to clear the storage format. Measured drift is about 5e-4."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(3, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    write_part("2015Q1", layout["2015Q1"], vectors=v)
    write_part("2015Q2", layout["2015Q2"])
    df = A.run("stub-v1")
    assert df[df["shard"] == "2015Q1"].iloc[0]["status"] == "OK"


def test_an_empty_embedding_set_reports_every_chunk_missing(layout, entry):
    df = A.run("stub-v1")
    assert int(df["n_missing"].sum()) == 5
    assert set(df["status"]) == {"NO_PART_FILE"}
    assert A.report(df, entry) == 1


def test_the_csv_is_written_where_the_next_stage_looks_for_it(layout, entry):
    for shard, ids in layout.items():
        write_part(shard, ids)
    A.run("stub-v1")
    out = A.P.PROCESSED_DIR / "embedding_audit_stub-v1.csv"
    assert out.exists()
    assert len(pl.read_csv(out)) == 2


def test_no_chunk_layer_raises_rather_than_reporting_a_clean_audit(tmp_path, monkeypatch):
    """Zero shards would otherwise sum to zero missing and look like a pass."""
    monkeypatch.setattr(A.P, "CHUNKS_DIR", tmp_path / "empty")
    monkeypatch.setattr(E.P, "CHUNKS_DIR", A.P.CHUNKS_DIR)
    A.P.CHUNKS_DIR.mkdir()
    with pytest.raises(SystemExit, match="no chunk shards"):
        A.run("stub-v1")


# --- an unreadable file must be a reported row, not an exception ---------------------------------

def test_a_corrupt_part_file_is_reported_rather_than_crashing_the_audit(layout, entry):
    """The audit loops over 88 shards. A raise partway through says nothing about the other 87, which
    is the opposite of what an audit is for. Three concurrent writers on one shard could have left a
    parquet whose footer does not parse, which is the case this covers."""
    write_part("2015Q2", layout["2015Q2"])
    bad = E.part_path("stub-v1", "2015Q1")
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"PAR1" + b"\x00" * 200)            # plausible magic, unparseable footer

    df = A.run("stub-v1")                                # must not raise
    row = df[df["shard"] == "2015Q1"].iloc[0]
    assert row["status"] == "CORRUPT"
    assert row["error"]
    assert row["bytes"] > 0
    assert df[df["shard"] == "2015Q2"].iloc[0]["status"] == "OK"   # the other shard still assessed
    assert A.report(df, entry) == 1


def test_a_truncated_part_file_is_corrupt_not_incomplete(layout, entry):
    """Truncation is not the same finding as a short-but-valid file, and conflating them would send
    someone looking for missing chunk ids in a file that cannot be opened."""
    write_part("2015Q1", layout["2015Q1"])
    write_part("2015Q2", layout["2015Q2"])
    p = E.part_path("stub-v1", "2015Q1")
    p.write_bytes(p.read_bytes()[: len(p.read_bytes()) // 2])
    df = A.run("stub-v1")
    assert df[df["shard"] == "2015Q1"].iloc[0]["status"] == "CORRUPT"


def test_a_stray_temp_file_is_noted(layout, entry, capsys):
    for shard, ids in layout.items():
        write_part(shard, ids)
    (E.part_path("stub-v1", "2015Q1").parent / "part_2015Q1.parquet.node.123.abcd.tmp").write_bytes(b"x")
    A.run("stub-v1")
    assert "stray temp file" in capsys.readouterr().out


# --- the expanding window: a vintage's later shards are absent on purpose ------------------------
#
# The fixture holds two shards, 2015Q1 and 2015Q2, so a cutoff of 2015-03-31 splits them.

def _vintage(cutoff):
    return Entry("stub-v1", VINTAGE, "huggingface:test/stub-v1", DIM, training_cutoff=cutoff)


def test_a_vintage_is_audited_against_its_window_not_every_shard(layout, monkeypatch):
    """Expecting every shard is wrong for 14 of 15 vintages and would report intentionally absent
    quarters as missing. That is worse than no gate, because a real absence would hide inside a list of
    expected ones."""
    v = _vintage("2015-03-31")
    monkeypatch.setattr(A.R, "load", lambda *a, **k: Registry([v]))
    write_part("2015Q1", layout["2015Q1"])          # 2015Q2 is outside the window, deliberately absent
    df = A.run("stub-v1", extra_years=0)
    assert list(df["shard"]) == ["2015Q1"]
    assert int(df["n_missing"].sum()) == 0
    assert A.report(df, v) == 0


def test_a_gap_inside_the_window_is_still_caught(layout, monkeypatch):
    """The narrower expectation must not become a way to pass while short."""
    v = _vintage("2015-12-31")
    monkeypatch.setattr(A.R, "load", lambda *a, **k: Registry([v]))
    write_part("2015Q1", layout["2015Q1"])          # 2015Q2 is inside the window and missing
    df = A.run("stub-v1", extra_years=0)
    assert sorted(df["shard"]) == ["2015Q1", "2015Q2"]
    assert int(df["n_missing"].sum()) == 2
    assert A.report(df, v) == 1


def test_the_extra_year_widens_what_the_audit_expects(layout, monkeypatch):
    v = _vintage("2014-12-31")
    monkeypatch.setattr(A.R, "load", lambda *a, **k: Registry([v]))
    write_part("2015Q1", layout["2015Q1"])
    assert A.run("stub-v1", extra_years=0).empty                  # nothing in 2014 exists in the fixture
    df = A.run("stub-v1", extra_years=1)                          # reaches the end of 2015
    assert sorted(df["shard"]) == ["2015Q1", "2015Q2"]


def test_an_atemporal_set_is_still_audited_against_every_shard(layout, monkeypatch):
    """A model with no cutoff has no window, so anything short of the whole corpus is a real gap."""
    a = Entry("stub-v1", ATEMPORAL, "huggingface:test/stub-v1", DIM)
    monkeypatch.setattr(A.R, "load", lambda *a_, **k: Registry([a]))
    df = A.run("stub-v1")
    assert sorted(df["shard"]) == ["2015Q1", "2015Q2"]
    assert int(df["n_missing"].sum()) == 5


def test_the_window_is_read_from_the_sets_own_sidecar_when_it_records_one(layout, monkeypatch):
    """Self-describing beats inferred: the set records what it was asked to cover, so the audit checks
    the intent rather than re-deriving it from a flag someone has to remember."""
    v = _vintage("2015-12-31")
    monkeypatch.setattr(A.R, "load", lambda *a, **k: Registry([v]))
    write_part("2015Q1", layout["2015Q1"])
    meta = A.E.meta_path("stub-v1")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"expected_shards": ["2015Q1"], "extra_years": 0}))
    df = A.run("stub-v1")
    assert list(df["shard"]) == ["2015Q1"]
    assert A.report(df, v) == 0


def test_a_damaged_sidecar_falls_back_to_the_wider_window_rather_than_failing(layout, monkeypatch):
    """Unreadable metadata must not quietly narrow what counts as complete, so the fallback is the
    builder's own default window and a gap inside it still shows."""
    v = _vintage("2015-03-31")
    monkeypatch.setattr(A.R, "load", lambda *a, **k: Registry([v]))
    write_part("2015Q1", layout["2015Q1"])
    meta = A.E.meta_path("stub-v1")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text("{not json at all")
    df = A.run("stub-v1")                                  # cutoff + 1 year reaches the end of 2016
    assert sorted(df["shard"]) == ["2015Q1", "2015Q2"]
    assert int(df["n_missing"].sum()) == 2                 # and the absent shard is reported


def test_an_unrecorded_window_defaults_to_the_wider_one_so_a_gap_is_reported(layout, monkeypatch):
    """The bug this guards. With no `extra_years` in the sidecar the fallback decides what complete
    means, and defaulting to 0 made the audit agree with whatever was on disk: it passed the 2022 and
    2023 vintages with MISSING = 0 while both were four shards short. Unsure must mean expect more and
    report, never expect less and pass."""
    v = _vintage("2014-12-31")
    monkeypatch.setattr(A.R, "load", lambda *a, **k: Registry([v]))
    write_part("2015Q1", layout["2015Q1"])          # 2015Q2 is inside cutoff+1 and absent
    assert A.E.meta_path("stub-v1").exists() is False       # nothing recorded the window
    df = A.run("stub-v1")
    assert sorted(df["shard"]) == ["2015Q1", "2015Q2"]      # the wider window, so the gap is visible
    assert int(df["n_missing"].sum()) == 2
    assert A.report(df, v) == 1


def test_the_audit_default_matches_the_builder_default():
    """If these drift, the audit silently checks a different window than the one being produced."""
    import inspect
    from ecalls.embed import encode as Enc
    src = inspect.getsource(Enc.main)
    assert '"--extra-years", type=int, default=1' in src
    assert A.DEFAULT_EXTRA_YEARS == 1
