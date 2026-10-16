"""Edge cases for the hybrid index, centred on the point-in-time refusals.

Those refusals are the reason this project exists, so they get constructed failures rather than a happy
path. The distinction the tests pin down is the one that is easy to get wrong: filtering rows by
`call_date` and checking the *model's* cutoff are different guarantees, and doing only the first looks
correct while leaving every vector contaminated.
"""
import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from ecalls.chunk import index as X
from ecalls.chunk.registry import ATEMPORAL, VINTAGE, Entry, Registry
from ecalls.embed import encode as E

D = dt.date
DIM = 4


def vintage(cutoff="2015-12-31", mid="v2015"):
    return Entry(mid, VINTAGE, f"huggingface:test/{mid}", DIM, training_cutoff=cutoff)


def atemporal(mid="bge-ish"):
    return Entry(mid, ATEMPORAL, f"huggingface:test/{mid}", DIM, pooling="cls")


# --- the model-cutoff refusal, which row filtering cannot substitute for ------------------------

def test_a_model_trained_after_the_decision_date_is_refused():
    """A 2024-cutoff encoder embedding a 2015 call has already read 2016 to 2024 language, so the
    representation carries the future even when the document does not."""
    with pytest.raises(ValueError, match="after as_of"):
        X.check_point_in_time(vintage("2024-12-31"), D(2015, 6, 30))


def test_a_model_trained_before_the_decision_date_is_allowed():
    X.check_point_in_time(vintage("2015-12-31"), D(2016, 1, 1))
    X.check_point_in_time(vintage("2015-12-31"), D(2015, 12, 31))   # the boundary is inclusive


def test_an_atemporal_model_is_refused_for_every_as_of_date():
    """Every strong retriever is in this position, since none report a cutoff. Refusing is correct
    rather than inconvenient: searchable and teachable, barred from a backtest (D16)."""
    a = atemporal()
    for date in (D(1999, 1, 1), D(2015, 6, 30), D(2030, 1, 1)):
        with pytest.raises(ValueError, match="cannot support an as_of query"):
            X.check_point_in_time(a, date)


def test_the_refusal_message_points_at_the_resolver_rather_than_just_complaining():
    """Someone hitting this needs to know the fix is a different model, not a different filter."""
    with pytest.raises(ValueError, match="for_date"):
        X.check_point_in_time(vintage("2024-12-31"), D(2015, 1, 1))
    with pytest.raises(ValueError, match="for_date"):
        X.check_point_in_time(atemporal(), D(2015, 1, 1))


def test_a_vintage_entry_without_a_cutoff_cannot_be_constructed():
    """The refusal above has a companion guarantee upstream: the registry will not let a vintage exist
    without a cutoff, so `check_point_in_time` cannot be handed one."""
    with pytest.raises(ValueError, match="needs a training_cutoff"):
        Entry("bad", VINTAGE, "huggingface:test/bad", DIM)


# --- building ------------------------------------------------------------------------------------

@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Two chunk shards and matching embeddings, with every path inside tmp_path."""
    monkeypatch.setattr(X.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(X.P, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(X.P, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(X.P, "chunks_shard", lambda s: X.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    monkeypatch.setattr(E.P, "CHUNKS_DIR", X.P.CHUNKS_DIR)
    monkeypatch.setattr(E.P, "EMBEDDINGS_DIR", X.P.EMBEDDINGS_DIR)
    X.P.CHUNKS_DIR.mkdir(parents=True)

    spec = {"2015Q1": (["a", "b"], D(2015, 2, 10)), "2016Q1": (["c"], D(2016, 2, 10))}
    # All comfortably over MIN_INDEX_CHARS, because these stand in for real chunks and the index drops
    # anything under 50 characters as a pleasantry.
    texts = {"a": "gross margin expanded on freight leverage and better mix across the portfolio",
             "b": "free cash flow guidance is unchanged for the full year despite the working capital build",
             "c": "we took a restructuring charge in the quarter related to the European footprint review"}
    for shard, (ids, date) in spec.items():
        pl.DataFrame({
            "chunk_id": ids, "text": [texts[i] for i in ids],
            "keydevid": [1] * len(ids), "companyid": [7] * len(ids),
            "permno": [10001] * len(ids), "call_date": [date] * len(ids),
            "in_index_at_call": [True] * len(ids), "section": ["qa"] * len(ids),
            "speaker_type": ["Executives"] * len(ids), "speaker_name": ["CFO"] * len(ids),
            "chunker_version": ["qa-pair-v2"] * len(ids),
        }).write_parquet(X.P.chunks_shard(shard))

        rng = np.random.default_rng(abs(hash(shard)) % 1000)
        v = rng.normal(size=(len(ids), DIM)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        dst = E.part_path("v2015", shard)
        dst.parent.mkdir(parents=True, exist_ok=True)
        import pyarrow.parquet as pq
        pq.write_table(E.to_table(ids, v, meta={"model_id": "v2015"}), dst)
    return spec


@pytest.fixture(autouse=True)
def stub_registry(monkeypatch):
    monkeypatch.setattr(X.R, "load", lambda *a, **k: Registry([vintage(), atemporal()]))


def test_build_indexes_every_row_and_records_which_shards_are_in(layout):
    res = X.build("v2015", fts=False, quiet=True)
    assert res["n_rows"] == 3
    meta = X.read_meta("v2015")
    assert meta["shards"] == ["2015Q1", "2016Q1"]
    assert meta["training_cutoff"] == "2015-12-31"
    assert meta["kind"] == VINTAGE


def test_rebuilding_does_not_duplicate_rows(layout):
    """Appending a shard twice duplicates rows, and no count check on a 12 M row table would notice,
    which is why the resume marker names the shards rather than counting them."""
    X.build("v2015", fts=False, quiet=True)
    again = X.build("v2015", fts=False, quiet=True)
    assert again["n_rows"] == 3
    assert again["shards"] == 0            # nothing re-added


def test_building_one_shard_then_the_other_gives_the_same_table(layout):
    X.build("v2015", shards=["2015Q1"], fts=False, quiet=True)
    res = X.build("v2015", shards=["2016Q1"], fts=False, quiet=True)
    assert res["n_rows"] == 3
    assert X.read_meta("v2015")["shards"] == ["2015Q1", "2016Q1"]


def test_a_missing_embedding_part_file_raises_with_the_path_it_wanted(layout):
    E.part_path("v2015", "2016Q1").unlink()
    with pytest.raises(FileNotFoundError, match="no embeddings for 2016Q1"):
        X.build("v2015", fts=False, quiet=True)


def test_a_partial_embedding_part_file_raises_rather_than_silently_shortening_the_index(layout):
    """An inner join would quietly drop the unmatched chunks and every recall number computed from the
    index would then be wrong with nothing reporting a gap."""
    import pyarrow.parquet as pq
    v = np.zeros((1, DIM), dtype=np.float32)
    v[0, 0] = 1.0
    pq.write_table(E.to_table(["a"], v), E.part_path("v2015", "2015Q1"))   # b is gone
    with pytest.raises(ValueError, match="silently short"):
        X.build("v2015", fts=False, quiet=True)


def test_overwrite_rebuilds_from_scratch(layout):
    X.build("v2015", shards=["2015Q1"], fts=False, quiet=True)
    res = X.build("v2015", fts=False, overwrite=True, quiet=True)
    assert res["n_rows"] == 3 and res["shards"] == 2


# --- searching -----------------------------------------------------------------------------------

def test_searching_a_missing_index_says_how_to_build_it(layout):
    with pytest.raises(FileNotFoundError, match="build"):
        X.open_table("v2015")


def test_as_of_filters_out_later_calls(layout):
    X.build("v2015", fts=True, quiet=True)
    got = X.search("margin", "v2015", k=10, as_of="2015-12-31", mode="fts")
    assert set(got["chunk_id"].to_list()) <= {"a", "b"}          # the 2016 call is excluded
    assert "c" not in got["chunk_id"].to_list()


def test_as_of_after_everything_returns_all_of_it(layout):
    X.build("v2015", fts=True, quiet=True)
    got = X.search("charge", "v2015", k=10, as_of="2016-12-31", mode="fts")
    assert "c" in got["chunk_id"].to_list()


def test_search_without_as_of_does_not_filter(layout):
    X.build("v2015", fts=True, quiet=True)
    got = X.search("charge", "v2015", k=10, mode="fts")
    assert "c" in got["chunk_id"].to_list()


def test_an_as_of_before_the_models_cutoff_refuses_even_though_rows_could_be_filtered(layout):
    """The point of the whole design. Rows on or before 2014 exist and could be returned, and the query
    still has to fail, because the vectors were written by a model that read 2015."""
    X.build("v2015", fts=True, quiet=True)
    with pytest.raises(ValueError, match="after as_of"):
        X.search("margin", "v2015", as_of="2014-06-30", mode="fts")


def test_a_where_clause_combines_with_the_as_of_filter(layout):
    X.build("v2015", fts=True, quiet=True)
    got = X.search("margin", "v2015", k=10, as_of="2016-12-31", mode="fts",
                   where="section = 'qa'")
    assert got.height >= 1
    none = X.search("margin", "v2015", k=10, as_of="2016-12-31", mode="fts",
                    where="section = 'presentation'")
    assert none.height == 0


def test_an_unknown_mode_raises_before_any_model_is_loaded(layout):
    """Validated up front, so a typo costs nothing. It used to load the encoder first and fail trying
    to download a model that does not exist, which buries the real mistake."""
    X.build("v2015", fts=False, quiet=True)
    with pytest.raises(ValueError, match="mode must be"):
        X.search("margin", "v2015", mode="magic")


def test_asking_for_fts_on_a_dense_only_index_says_so(layout):
    X.build("v2015", fts=False, quiet=True)
    with pytest.raises(ValueError, match="no lexical index"):
        X.search("margin", "v2015", mode="fts")


def test_table_list_returns_plain_names_across_lancedb_versions():
    """`list_tables()` is the non-deprecated call and it returns a ListTablesResponse, not a list, so
    `name in db.list_tables()` is always False. Swapping it in to silence the warning made every build
    recreate the table with mode='overwrite' and discard the shards already indexed."""
    from types import SimpleNamespace

    modern = SimpleNamespace(list_tables=lambda: SimpleNamespace(tables=["chunks"]))
    assert X.table_list(modern) == ["chunks"]

    plain = SimpleNamespace(list_tables=lambda: ["chunks"])
    assert X.table_list(plain) == ["chunks"]

    class Old:
        def table_names(self):
            return ["chunks"]

        def __getattr__(self, name):
            raise AttributeError(name)

    assert X.table_list(Old()) == ["chunks"]


# --- the approximate vector index ----------------------------------------------------------------

def test_a_small_table_skips_the_ann_index_rather_than_failing(layout, capsys):
    """PQ training needs at least 256 rows and errors below that. A build over one quarter, for a test
    or a spot check, has to just work, and an exact scan at that size is already fast."""
    res = X.build("v2015", fts=False, ann=True, quiet=False)
    assert res["n_rows"] == 3
    assert "skipping the ANN index" in capsys.readouterr().out
    assert X.read_meta("v2015")["ann"] is False


def test_the_meta_records_whether_the_ann_index_was_actually_built(layout):
    """Recording the request rather than the outcome would claim an index that is not there, and a
    latency measurement would then be blamed on the wrong thing."""
    X.build("v2015", fts=False, ann=True, quiet=True)
    assert X.read_meta("v2015")["ann"] is False          # asked for, correctly skipped, recorded false


def test_the_encoder_is_cached_across_searches(layout, monkeypatch):
    """Measured at 2.74s to load against 0.006s to embed a query, so reloading per call is the whole
    interactive latency."""
    X.build("v2015", fts=False, ann=False, quiet=True)
    X._ENCODER_CACHE.clear()
    calls = []

    def fake_load(entry, device=None, *a, **k):
        calls.append(entry.model_id)
        return ("tok", "model", "cpu")

    monkeypatch.setattr(X.E, "load_encoder", fake_load)
    e = vintage()
    X.cached_encoder(e)
    X.cached_encoder(e)
    X.cached_encoder(e)
    assert calls == ["v2015"]                            # loaded once, not three times
    X._ENCODER_CACHE.clear()


def test_a_different_device_reloads_rather_than_reusing_the_cached_encoder(layout, monkeypatch):
    X._ENCODER_CACHE.clear()
    calls = []
    monkeypatch.setattr(X.E, "load_encoder",
                        lambda entry, device=None, *a, **k: (calls.append(device), ("t", "m", "x"))[1])
    e = vintage()
    X.cached_encoder(e, "cpu")
    X.cached_encoder(e, "mps")
    X.cached_encoder(e, "cpu")
    assert calls == ["cpu", "mps"]
    X._ENCODER_CACHE.clear()


# --- trivial chunks are kept in the corpus and left out of the index -----------------------------

def test_short_chunks_are_excluded_from_the_index(tmp_path, monkeypatch):
    """"Thank you." appears 12,404 times in the corpus and is never a useful retrieval result. It stays
    in the corpus, because ingest is maximal and filtering is downstream, and the index is downstream."""
    monkeypatch.setattr(X.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(X.P, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(X.P, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(X.P, "chunks_shard", lambda s: X.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    monkeypatch.setattr(E.P, "CHUNKS_DIR", X.P.CHUNKS_DIR)
    monkeypatch.setattr(E.P, "EMBEDDINGS_DIR", X.P.EMBEDDINGS_DIR)
    X.P.CHUNKS_DIR.mkdir(parents=True)

    ids = ["keep", "drop", "edge"]
    texts = {"keep": "x" * 300, "drop": "Thank you.", "edge": "y" * 50}
    pl.DataFrame({
        "chunk_id": ids, "text": [texts[i] for i in ids],
        "keydevid": [1] * 3, "companyid": [7] * 3, "permno": [1] * 3,
        "call_date": [D(2015, 2, 1)] * 3, "in_index_at_call": [True] * 3,
        "section": ["qa"] * 3, "speaker_type": ["Executives"] * 3, "speaker_name": ["CFO"] * 3,
        "chunker_version": ["qa-pair-v2"] * 3,
    }).write_parquet(X.P.chunks_shard("2015Q1"))
    import pyarrow.parquet as pq
    v = np.eye(3, DIM, dtype=np.float32)
    dst = E.part_path("v2015", "2015Q1")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(E.to_table(ids, v, meta={"model_id": "v2015"}), dst)

    res = X.build("v2015", fts=False, ann=False, quiet=True, min_chars=50)
    assert res["n_rows"] == 2                              # "keep" and the 50-char edge case
    assert X.read_meta("v2015")["min_chars"] == 50

    got = set(X.open_table("v2015").search().limit(10).to_polars()["chunk_id"].to_list())
    assert got == {"keep", "edge"}                         # the boundary is inclusive


def test_min_chars_zero_indexes_everything(tmp_path, monkeypatch):
    test_short_chunks_are_excluded_from_the_index.__wrapped__ if False else None
    monkeypatch.setattr(X.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(X.P, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(X.P, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(X.P, "chunks_shard", lambda s: X.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    monkeypatch.setattr(E.P, "CHUNKS_DIR", X.P.CHUNKS_DIR)
    monkeypatch.setattr(E.P, "EMBEDDINGS_DIR", X.P.EMBEDDINGS_DIR)
    X.P.CHUNKS_DIR.mkdir(parents=True)
    pl.DataFrame({
        "chunk_id": ["a", "b"], "text": ["Yes.", "z" * 300],
        "keydevid": [1, 1], "companyid": [7, 7], "permno": [1, 1],
        "call_date": [D(2015, 2, 1)] * 2, "in_index_at_call": [True] * 2,
        "section": ["qa"] * 2, "speaker_type": ["Executives"] * 2, "speaker_name": ["CFO"] * 2,
        "chunker_version": ["qa-pair-v2"] * 2,
    }).write_parquet(X.P.chunks_shard("2015Q1"))
    import pyarrow.parquet as pq
    dst = E.part_path("v2015", "2015Q1")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(E.to_table(["a", "b"], np.eye(2, DIM, dtype=np.float32)), dst)
    assert X.build("v2015", fts=False, ann=False, quiet=True, min_chars=0)["n_rows"] == 2


def test_the_embedding_coverage_check_runs_before_the_length_filter(tmp_path, monkeypatch):
    """Otherwise an incomplete embedding set would be hidden behind an intended exclusion, and the index
    would be short for a reason nobody chose."""
    monkeypatch.setattr(X.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(X.P, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(X.P, "chunks_shard", lambda s: X.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    monkeypatch.setattr(E.P, "CHUNKS_DIR", X.P.CHUNKS_DIR)
    monkeypatch.setattr(E.P, "EMBEDDINGS_DIR", X.P.EMBEDDINGS_DIR)
    X.P.CHUNKS_DIR.mkdir(parents=True)
    pl.DataFrame({
        "chunk_id": ["a", "b"], "text": ["Yes.", "z" * 300],   # "a" would be filtered anyway
        "keydevid": [1, 1], "companyid": [7, 7], "permno": [1, 1],
        "call_date": [D(2015, 2, 1)] * 2, "in_index_at_call": [True] * 2,
        "section": ["qa"] * 2, "speaker_type": ["Executives"] * 2, "speaker_name": ["CFO"] * 2,
        "chunker_version": ["qa-pair-v2"] * 2,
    }).write_parquet(X.P.chunks_shard("2015Q1"))
    import pyarrow.parquet as pq
    dst = E.part_path("v2015", "2015Q1")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(E.to_table(["b"], np.eye(1, DIM, dtype=np.float32)), dst)   # "a" has no embedding
    with pytest.raises(ValueError, match="silently short"):
        X.shard_frame("2015Q1", "v2015")
