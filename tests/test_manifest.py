"""Edge cases for the shared shard-key and batching helpers."""
import datetime as dt
import os
import sys

import pytest

from ecalls.ingest.manifest import batched, shard_key


def test_shard_key_on_quarter_boundaries():
    assert shard_key(dt.date(2014, 1, 1)) == "2014Q1"
    assert shard_key(dt.date(2014, 3, 31)) == "2014Q1"
    assert shard_key(dt.date(2014, 4, 1)) == "2014Q2"
    assert shard_key(dt.date(2014, 9, 30)) == "2014Q3"
    assert shard_key(dt.date(2014, 10, 1)) == "2014Q4"
    assert shard_key(dt.date(2014, 12, 31)) == "2014Q4"


def test_batches_partition_the_input_exactly():
    src = list(range(1007))
    out = list(batched(src, 250))
    assert [x for b in out for x in b] == src     # no drops, no duplicates, order preserved
    assert len(out) == 5
    assert len(out[-1]) == 1007 - 4 * 250         # last batch may be short


def test_batching_empty_and_exact_multiple():
    assert list(batched([], 10)) == []
    assert list(batched(range(20), 10)) == [[*range(10)], [*range(10, 20)]]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


# --- provenance where there is no git checkout ---------------------------------------------------

def test_git_sha_falls_back_to_the_synced_file_when_git_is_unavailable(tmp_path, monkeypatch):
    """The cluster copy is an rsync of `src/`, not a clone, so `git rev-parse` fails there. Without
    this fallback every embedding written on a compute node stamped `unknown`, which defeats the field
    on exactly the artifacts whose provenance matters most."""
    from ecalls.ingest import manifest as M

    monkeypatch.setattr(M.P, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(M.P, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(M.subprocess, "check_output",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no git here")))
    assert M.git_sha() == "unknown"

    (tmp_path / M.SHA_FILE).write_text("abc1234\n")
    assert M.git_sha() == "abc1234"


def test_git_sha_ignores_an_empty_sha_file(tmp_path, monkeypatch):
    from ecalls.ingest import manifest as M

    monkeypatch.setattr(M.P, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(M.P, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(M.subprocess, "check_output",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no git here")))
    (tmp_path / M.SHA_FILE).write_text("   \n")
    assert M.git_sha() == "unknown"


def test_git_sha_prefers_git_over_the_file(tmp_path, monkeypatch):
    from ecalls.ingest import manifest as M

    monkeypatch.setattr(M.P, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(M.subprocess, "check_output", lambda *a, **k: "real567\n")
    (tmp_path / M.SHA_FILE).write_text("stale00\n")
    assert M.git_sha() == "real567"


# --- concurrent writers must not corrupt each other ---------------------------------------------

def test_temp_sibling_is_unique_per_process_and_beside_the_target(tmp_path):
    """A fixed `<name>.tmp` is safe for one writer and unsafe for two. Three SLURM tasks landed on the
    same shard, interleaved their writes into one temp file, and the first rename could have moved a
    corrupt file into place while the others failed on a temp file that had been renamed away."""
    from ecalls.ingest import manifest as M

    target = tmp_path / "part_2010Q1.parquet"
    t = M.temp_sibling(target)
    assert t.parent == target.parent                  # same directory, so the rename stays atomic
    assert t.name.startswith("part_2010Q1.parquet.")
    assert t.name.endswith(".tmp")
    assert str(os.getpid()) in t.name


def test_two_writers_pick_different_temp_names(tmp_path, monkeypatch):
    from ecalls.ingest import manifest as M

    target = tmp_path / "x.parquet"
    a = M.temp_sibling(target)
    monkeypatch.setattr(M.os, "getpid", lambda: 999999)
    b = M.temp_sibling(target)
    assert a != b


def test_concurrent_writes_to_one_path_both_produce_a_readable_file(tmp_path):
    """The property that matters: with per-writer temp names, two writers racing on the same
    destination leave a valid file rather than a spliced one. The winner is whichever renames last."""
    import threading

    import pyarrow as pa
    import pyarrow.parquet as pq

    from ecalls.ingest import manifest as M

    target = tmp_path / "race.parquet"
    errors = []

    def write(tag, n):
        try:
            for _ in range(8):
                M.atomic_write_table(
                    pa.table({"who": pa.array([tag] * n), "i": pa.array(list(range(n)))}), target)
        except Exception as exc:      # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=write, args=(t, n)) for t, n in (("a", 500), ("b", 500))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert errors == [], errors
    back = pq.read_table(target)                      # readable, not spliced
    who = set(back["who"].to_pylist())
    assert len(who) == 1                              # one writer's rows, never a mix
    assert back.num_rows == 500
    assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"


def test_a_failed_write_leaves_no_temp_file_behind(tmp_path):
    """A stray temp file would be picked up by the next glob over the directory."""
    from ecalls.ingest import manifest as M

    target = tmp_path / "bad.parquet"
    with pytest.raises(Exception):
        M.atomic_write_table("not a table at all", target)
    assert not list(tmp_path.glob("*.tmp"))
    assert not target.exists()
