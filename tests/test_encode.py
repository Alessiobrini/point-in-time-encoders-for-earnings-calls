"""Edge cases for the embedding pass. No model download and no GPU: a stub encoder stands in.

The two invariants worth real tests are the ones a plausible-looking implementation gets wrong
silently. Length-sorted batching reorders the inputs, so the vectors must be put back; and padding a
short chunk beside a long one must not change the short chunk's vector.
"""
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from ecalls.chunk.registry import ATEMPORAL, VINTAGE, Entry
from ecalls.embed import encode as E


# --- a stub encoder whose output depends on the tokens, so a mix-up is visible -------------------

class StubTok:
    """Characters to token ids, padded to the batch. Padding is real, so the mask has to be used."""

    def __call__(self, batch, padding=True, truncation=True, max_length=None, return_tensors="pt"):
        ids = [[ord(c) % 97 + 1 for c in t][:max_length] for t in batch]
        width = max(len(x) for x in ids)
        return {
            "input_ids": torch.tensor([x + [0] * (width - len(x)) for x in ids]),
            "attention_mask": torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids]),
        }


class StubModel:
    config = SimpleNamespace(hidden_size=4)

    def __call__(self, input_ids=None, attention_mask=None):
        f = input_ids.float()
        h = torch.stack([f, f * 0.01, torch.sqrt(f), torch.ones_like(f)], dim=-1)
        return SimpleNamespace(last_hidden_state=h)

    def eval(self):
        return self


def entry(dim=4, normalized=True):
    return Entry("stub-v1", VINTAGE, "huggingface:test/stub-v1", dim,
                 training_cutoff="2024-12-31", normalized=normalized)


TEXTS = ["short", "a much longer piece of text than the others here", "mid length text",
         "x", "another moderately sized chunk of words", "zz"]


# --- order and padding --------------------------------------------------------------------------

def test_length_sorted_batching_restores_the_input_order():
    """Batches are formed over length-sorted texts. If the output is not scattered back, every
    vector lands on the wrong chunk_id, which no downstream check would notice."""
    tok, model = StubTok(), StubModel()
    batched = E.encode_texts(TEXTS, tok, model, "cpu", batch_size=4, max_length=64)
    one_at_a_time = np.vstack([
        E.encode_texts([t], tok, model, "cpu", batch_size=1, max_length=64) for t in TEXTS])
    assert np.allclose(batched, one_at_a_time, atol=1e-6)


def test_padding_does_not_change_a_short_chunks_vector():
    """The single-text encoding has no padding at all, so agreeing with it is the padding test."""
    tok, model = StubTok(), StubModel()
    alone = E.encode_texts(["x"], tok, model, "cpu", batch_size=1, max_length=64)[0]
    crowded = E.encode_texts(["x", "a" * 400], tok, model, "cpu", batch_size=2, max_length=64)[0]
    assert np.allclose(alone, crowded, atol=1e-6)


def test_mean_pool_ignores_padded_positions():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(E.mean_pool(hidden, mask), torch.tensor([[2.0, 2.0]]))


def test_normalization_is_applied_or_not_as_the_entry_says():
    tok, model = StubTok(), StubModel()
    v = E.encode_texts(TEXTS, tok, model, "cpu", batch_size=3, normalize=True)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    raw = E.encode_texts(TEXTS, tok, model, "cpu", batch_size=3, normalize=False)
    assert not np.allclose(np.linalg.norm(raw, axis=1), 1.0, atol=1e-2)


def test_max_length_truncates():
    tok, model = StubTok(), StubModel()
    long = "abcdefghij" * 40
    short = E.encode_texts([long], tok, model, "cpu", max_length=8)[0]
    full = E.encode_texts([long], tok, model, "cpu", max_length=400)[0]
    assert not np.allclose(short, full, atol=1e-4)
    same = E.encode_texts([long[:8]], tok, model, "cpu", max_length=400)[0]
    assert np.allclose(short, same, atol=1e-6)


def test_empty_input_returns_an_empty_matrix_of_the_right_width():
    out = E.encode_texts([], StubTok(), StubModel(), "cpu")
    assert out.shape == (0, 4)


# --- the stored table ---------------------------------------------------------------------------

def test_table_round_trips_ids_and_vectors(tmp_path):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(7, 4)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ids = [f"c{i}" for i in range(7)]
    t = E.to_table(ids, v, meta={"model_id": "stub-v1"})
    p = tmp_path / "part_2015Q1.parquet"
    import pyarrow.parquet as pq
    pq.write_table(t, p, compression="zstd")
    back_ids, back_v = E.read_vectors(p)
    assert back_ids == ids
    assert back_v.shape == (7, 4)
    assert np.allclose(back_v, v, atol=1e-3)          # float16 storage, so 1e-3 not 1e-7
    assert pq.read_schema(p).metadata[b"model_id"] == b"stub-v1"


def test_table_refuses_a_length_mismatch():
    with pytest.raises(ValueError, match="against"):
        E.to_table(["a", "b"], np.zeros((3, 4), dtype=np.float32))


def test_float32_storage_is_exact_when_asked_for(tmp_path):
    v = np.array([[0.1234567, 0.2, 0.3, 0.4]], dtype=np.float32)
    import pyarrow.parquet as pq
    p = tmp_path / "f32.parquet"
    pq.write_table(E.to_table(["a"], v, dtype="float32"), p)
    _, back = E.read_vectors(p)
    assert np.array_equal(back, v)


# --- resume, which must compare coverage rather than existence ----------------------------------

def _part(path, ids, dim=4):
    import pyarrow.parquet as pq
    v = np.zeros((len(ids), dim), dtype=np.float32)
    pq.write_table(E.to_table(list(ids), v), path)


def test_a_missing_part_file_is_not_complete(tmp_path):
    assert E.shard_is_complete(tmp_path / "nope.parquet", ["a"]) is False


def test_a_short_part_file_is_not_complete(tmp_path):
    """The failure this whole function exists for: the file is there and it is missing chunks."""
    p = tmp_path / "part_2015Q1.parquet"
    _part(p, ["a", "b"])
    assert E.shard_is_complete(p, ["a", "b", "c"]) is False


def test_a_covering_part_file_is_complete(tmp_path):
    p = tmp_path / "part_2015Q1.parquet"
    _part(p, ["a", "b", "c"])
    assert E.shard_is_complete(p, ["a", "b", "c"]) is True
    assert E.shard_is_complete(p, ["a", "b"]) is True          # a superset still covers


def test_a_corrupt_part_file_is_not_complete(tmp_path):
    p = tmp_path / "part_2015Q1.parquet"
    p.write_bytes(b"not parquet at all")
    assert E.shard_is_complete(p, ["a"]) is False


def test_an_empty_shard_is_complete_once_its_empty_part_exists(tmp_path):
    p = tmp_path / "part_2004Q3.parquet"
    _part(p, [])
    assert E.shard_is_complete(p, []) is True


# --- the registry guard rails -------------------------------------------------------------------

def test_load_encoder_refuses_an_unknown_source_scheme():
    """Two schemes load: a huggingface repo and a `local:` model this project trained. Anything else,
    a bare path most of all, is refused, because a model resolved by path can be the wrong vintage."""
    e = Entry("local-thing", ATEMPORAL, "file:///models/local-thing", 768)
    with pytest.raises(ValueError, match="huggingface:|local:"):
        E.load_encoder(e)


def test_a_local_source_with_no_exported_model_raises_with_the_path(monkeypatch, tmp_path):
    """The contrastive stage writes the export. Before it runs, the registry entry points at nothing,
    and saying which directory was missing is the difference between a one-line fix and a hunt."""
    monkeypatch.setattr(E.P, "PROCESSED_DIR", tmp_path)
    e = Entry("ecalls-ret-20101231", "vintage", "local:ecalls-ret-20101231", 384,
              training_cutoff="2010-12-31")
    with pytest.raises(FileNotFoundError, match="ecalls-ret-20101231"):
        E.load_encoder(e)


def test_embed_shard_skips_a_covered_shard_without_touching_the_model(tmp_path, monkeypatch):
    """Resume must not need an encoder. On the cluster the skip path runs before any GPU is claimed,
    and a model load inside it would make a no-op run cost a GPU-minute per shard."""
    monkeypatch.setattr(E.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(E.P, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(E.P, "chunks_shard", lambda s: E.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    E.P.CHUNKS_DIR.mkdir(parents=True)
    pl.DataFrame({"chunk_id": ["a", "b"], "text": ["x", "y"]}).write_parquet(
        E.P.chunks_shard("2015Q1"))
    dst = E.part_path("stub-v1", "2015Q1")
    dst.parent.mkdir(parents=True)
    _part(dst, ["a", "b"])

    boom = lambda *a, **k: pytest.fail("the skip path loaded the model")   # noqa: E731
    res = E.embed_shard("2015Q1", entry(), boom, boom, "cpu")
    assert res["status"] == "skipped" and res["n_chunks"] == 2


# --- token-budget batching ----------------------------------------------------------------------

def test_budget_batches_never_exceed_the_budget():
    lengths = [5, 500, 12, 300, 7, 800, 200, 3, 60, 450]
    for b in (100, 600, 1000, 4000):
        for batch in E.plan_batches(lengths, char_budget=b):
            padded = len(batch) * max(lengths[i] for i in batch)
            assert padded <= b or len(batch) == 1, (b, batch, padded)


def test_budget_batches_partition_the_inputs_exactly():
    """Every index once and only once. A batcher that drops or duplicates an input is the bug that
    would silently shorten an embedding set, and the coverage check downstream would then rerun the
    shard forever."""
    lengths = [7, 3, 900, 44, 44, 1, 612, 88, 88, 88, 2000]
    for b in (50, 500, 5000):
        got = sorted(i for batch in E.plan_batches(lengths, char_budget=b) for i in batch)
        assert got == list(range(len(lengths))), b


def test_a_text_longer_than_the_whole_budget_still_gets_its_own_batch():
    """Otherwise the loop either drops it or never terminates."""
    batches = E.plan_batches([10, 5000, 10], char_budget=100)
    assert sorted(i for b in batches for i in b) == [0, 1, 2]
    assert [1] in batches


def test_budget_batches_are_shortest_first():
    lengths = [500, 5, 200, 1]
    flat = [i for b in E.plan_batches(lengths, char_budget=1000) for i in b]
    assert [lengths[i] for i in flat] == sorted(lengths)


def test_budget_makes_bigger_batches_for_short_text_than_a_fixed_count_would():
    """The whole point. 400 short chunks under a fixed count of 64 is 7 batches; under a budget they
    pack into far fewer, which is where the idle GPU on the short half of the corpus goes."""
    lengths = [20] * 400
    fixed = E.plan_batches(lengths, batch_size=64)
    budget = E.plan_batches(lengths, char_budget=8000)
    assert len(budget) < len(fixed)
    assert max(len(b) for b in budget) > 64


def test_budget_and_fixed_batching_give_the_same_vectors():
    """Batching is a throughput device and must not touch the numbers."""
    tok, model = StubTok(), StubModel()
    texts = TEXTS * 4
    fixed = E.encode_texts(texts, tok, model, "cpu", batch_size=4, max_length=64)
    budget = E.encode_texts(texts, tok, model, "cpu", char_budget=120, max_length=64)
    assert np.allclose(fixed, budget, atol=1e-6)


def test_an_empty_input_plans_no_batches():
    assert E.plan_batches([], char_budget=100) == []
    assert E.plan_batches([], batch_size=8) == []


# --- the benchmark sampler ----------------------------------------------------------------------

def test_stratified_sample_spans_the_length_range_where_a_head_slice_would_not():
    """A head slice of a shard is not a sample: chunk length tracks the section and the era, so
    `texts[:n]` measures whichever quarter's short prepared remarks sort first. This is the same
    mistake as reading a length-sorted run's first cumulative rate as its throughput."""
    from ecalls.embed import bench

    texts = [f"{'x' * n}" for n in range(10, 3010, 10)]        # 300 texts, 10 to 3000 chars
    s = bench.stratified_sample(texts, 30, seed=0)
    assert len(s) == 30
    lens = sorted(len(t) for t in s)
    assert lens[0] <= 50 and lens[-1] >= 2950
    head = [len(t) for t in texts[:30]]
    assert max(head) < 500                                     # what the naive slice would have given


def test_stratified_sample_returns_everything_when_asked_for_more_than_exists():
    from ecalls.embed import bench
    texts = ["a", "bb", "ccc"]
    assert sorted(bench.stratified_sample(texts, 99)) == ["a", "bb", "ccc"]


def test_stratified_sample_does_not_duplicate():
    from ecalls.embed import bench
    texts = [f"{'x' * n}" for n in range(1, 201)]
    s = bench.stratified_sample(texts, 50, seed=3)
    assert len(set(s)) == 50


# --- pooling, which is a property of the model and not of the caller ----------------------------

def test_cls_pooling_takes_the_first_token_and_ignores_the_rest():
    hidden = torch.tensor([[[1.0, 2.0], [50.0, 60.0], [70.0, 80.0]]])
    mask = torch.tensor([[1, 1, 1]])
    assert torch.allclose(E.cls_pool(hidden, mask), torch.tensor([[1.0, 2.0]]))


def test_cls_and_mean_pooling_give_different_vectors():
    """If they agreed, the registry's `pooling` field would be decorative and a BGE-family retriever
    would be silently mean-pooled, which degrades its retrieval."""
    tok, model = StubTok(), StubModel()
    m = E.encode_texts(TEXTS, tok, model, "cpu", batch_size=3, pooling="mean")
    c = E.encode_texts(TEXTS, tok, model, "cpu", batch_size=3, pooling="cls")
    assert not np.allclose(m, c, atol=1e-3)


def test_an_unknown_pooling_raises_rather_than_falling_back_to_mean():
    with pytest.raises(ValueError, match="pooling must be one of"):
        E.encode_texts(["x"], StubTok(), StubModel(), "cpu", pooling="max")


def test_cls_pooling_also_restores_the_input_order():
    """The scatter is shared, but a pooling change is exactly when someone rewrites that loop."""
    tok, model = StubTok(), StubModel()
    batched = E.encode_texts(TEXTS, tok, model, "cpu", batch_size=4, pooling="cls")
    alone = np.vstack([E.encode_texts([t], tok, model, "cpu", batch_size=1, pooling="cls")
                       for t in TEXTS])
    assert np.allclose(batched, alone, atol=1e-6)


# --- OOM resilience, which is why three of sixteen array tasks died -----------------------------

class OOMOnceModel(StubModel):
    """Raises OutOfMemoryError for any batch above `limit`, then succeeds. Stands in for an 11 GB card."""

    def __init__(self, limit):
        self.limit = limit
        self.attempts = []      # every call, including the ones that raise
        self.succeeded = []     # only the ones that returned

    def __call__(self, input_ids=None, attention_mask=None):
        size = int(input_ids.shape[0])
        self.attempts.append(size)
        if size > self.limit:
            raise torch.OutOfMemoryError("stub OOM")
        self.succeeded.append(size)
        return super().__call__(input_ids=input_ids, attention_mask=attention_mask)


def test_an_oom_splits_the_batch_and_still_embeds_everything():
    """The real failure: batch 256 fit a 24 GB A5000 and OOMed a 10.58 GB 2080, killing the task in
    `apply_rotary_pos_emb`. Splitting must recover rather than lose the shard."""
    tok = StubTok()
    model = OOMOnceModel(limit=2)
    texts = TEXTS * 2                                    # 12 texts
    got = E.encode_texts(texts, tok, model, "cpu", batch_size=8, quiet_oom=True)
    ref = E.encode_texts(texts, tok, StubModel(), "cpu", batch_size=1)
    assert np.allclose(got, ref, atol=1e-6)              # same vectors, in the right rows
    assert max(model.succeeded) <= 2                      # every batch that ran was within the limit
    assert max(model.attempts) == 8                       # and it really did hit the limit first
    assert sum(model.succeeded) == len(texts)             # each text embedded exactly once


def test_a_single_sequence_that_oomes_raises_rather_than_looping_forever():
    """There is nothing left to split, so the honest outcome is a failure the log can show."""
    model = OOMOnceModel(limit=0)
    with pytest.raises(torch.OutOfMemoryError):
        E.encode_texts(["one text"], StubTok(), model, "cpu", batch_size=1, quiet_oom=True)


def test_progress_counts_each_text_once_across_an_oom_split():
    """A split must not double-count, or the reported rate and the eta go wrong mid-shard."""
    seen = []
    E.encode_texts(TEXTS * 3, StubTok(), OOMOnceModel(limit=3), "cpu", batch_size=9,
                   quiet_oom=True, progress=lambda d, n: seen.append(d))
    assert seen[-1] == len(TEXTS) * 3
    assert seen == sorted(seen)


# --- the memory-scaled budget -------------------------------------------------------------------

def test_auto_char_budget_is_none_off_cuda_so_the_fixed_batch_is_used():
    assert E.auto_char_budget("cpu") is None
    assert E.auto_char_budget("mps") is None


def test_auto_char_budget_scales_with_device_memory(monkeypatch):
    monkeypatch.setattr(E, "gpu_memory_gb", lambda d: 10.58)
    small = E.auto_char_budget("cuda")
    monkeypatch.setattr(E, "gpu_memory_gb", lambda d: 24.0)
    big = E.auto_char_budget("cuda")
    assert small < big
    # the 11 GB card must land below the ~770k padded chars that actually OOMed it
    assert small < 770_000


def test_auto_char_budget_is_capped_so_a_huge_card_does_not_make_absurd_batches(monkeypatch):
    monkeypatch.setattr(E, "gpu_memory_gb", lambda d: 640.0)
    assert E.auto_char_budget("cuda") == E.MAX_CHAR_BUDGET


def test_gpu_memory_gb_returns_none_rather_than_raising_off_cuda():
    assert E.gpu_memory_gb("cpu") is None
    assert E.gpu_memory_gb("mps") is None


# --- a broken GPU must fail fast, not run on CPU -------------------------------------------------

def test_require_gpu_raises_instead_of_falling_back_to_cpu(monkeypatch):
    """The measured failure: a node reported "Failed to get device handle for GPU 0", CUDA went
    invisible, and this silently returned "cpu". The task then ran at 4 chunk/s against 265, an eta of
    64 hours, and burned 1h44 of an 8 hour allocation before anyone looked. Nothing failed and nothing
    finished, which is the worst available outcome."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert E.resolve_device() == "cpu"
    with pytest.raises(RuntimeError, match="no GPU is visible"):
        E.resolve_device(require_gpu=True)


def test_require_gpu_is_satisfied_by_cuda_or_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert E.resolve_device(require_gpu=True) == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert E.resolve_device(require_gpu=True) == "mps"


def test_an_explicit_device_still_wins_over_require_gpu(monkeypatch):
    """`--device cpu --require-gpu` is contradictory, and an explicit choice is the caller being
    deliberate, so it is honored rather than second-guessed."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert E.resolve_device("cpu", require_gpu=True) == "cpu"


# --- a short-context model must not be asked for more positions than it has ----------------------

def _entry_with(limit, dim=4):
    from ecalls.chunk.registry import ATEMPORAL
    extra = {"max_seq_len": limit} if limit else {}
    return Entry("stub-ctx", ATEMPORAL, "huggingface:test/stub-ctx", dim, extra=extra)


def test_max_length_is_capped_by_the_models_own_context():
    """bge-small accepts 512 and MiniLM 256, while the chunk-layer default is 768. A BERT-family model
    asked for more positions than it has raises on its position embeddings."""
    assert E.effective_max_length(_entry_with(512)) == 512
    assert E.effective_max_length(_entry_with(256)) == 256
    assert E.effective_max_length(_entry_with(8192)) == E.MAX_LENGTH      # no cap needed


def test_an_entry_with_no_declared_context_keeps_the_requested_length():
    assert E.effective_max_length(_entry_with(None), 768) == 768


def test_the_cap_never_raises_the_requested_length():
    """A 8192-context model does not mean 'send 8192'; the chunk layer's own ceiling still governs."""
    assert E.effective_max_length(_entry_with(8192), 512) == 512


def test_embed_shard_applies_the_cap_rather_than_trusting_the_caller(tmp_path, monkeypatch):
    """The cap has to live in the pipeline, not only in the diagnostics, or the two disagree."""
    monkeypatch.setattr(E.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(E.P, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(E.P, "chunks_shard", lambda s: E.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    E.P.CHUNKS_DIR.mkdir(parents=True)
    pl.DataFrame({"chunk_id": ["a"], "text": ["q" * 5000],
                  "chunker_version": ["qa-pair-v2"]}).write_parquet(E.P.chunks_shard("2015Q1"))

    seen = {}
    tok = StubTok()

    class RecordingTok:
        def __call__(self, batch, **kw):
            seen["max_length"] = kw.get("max_length")
            return tok(batch, **kw)

    ent = _entry_with(64)
    E.embed_shard("2015Q1", ent, RecordingTok(), StubModel(), "cpu", max_length=768)
    assert seen["max_length"] == 64


# --- the expanding window: a vintage embeds only text up to its own cutoff -----------------------

def test_shard_end_is_the_last_day_of_the_quarter():
    import datetime as dt
    assert E.shard_end("2015Q1") == dt.date(2015, 3, 31)
    assert E.shard_end("2015Q2") == dt.date(2015, 6, 30)
    assert E.shard_end("2015Q3") == dt.date(2015, 9, 30)
    assert E.shard_end("2015Q4") == dt.date(2015, 12, 31)


def test_shards_up_to_a_cutoff_excludes_later_quarters():
    shards = ["2014Q3", "2014Q4", "2015Q1", "2015Q2", "2015Q3", "2015Q4", "2016Q1"]
    assert E.shards_up_to("2015-12-31", shards) == ["2014Q3", "2014Q4", "2015Q1", "2015Q2",
                                                    "2015Q3", "2015Q4"]


def test_a_shard_straddling_the_cutoff_is_excluded_rather_than_partly_included():
    """A quarter ending after the cutoff holds calls on both sides of it. Taking the whole shard would
    embed post-cutoff text; taking part of it would leave the shard's coverage check unsatisfiable
    forever, so the resume marker would re-run it on every pass."""
    shards = ["2015Q2", "2015Q3", "2015Q4"]
    assert E.shards_up_to("2015-08-15", shards) == ["2015Q2"]


def test_a_cutoff_before_everything_yields_nothing():
    assert E.shards_up_to("1990-01-01", ["2015Q1", "2015Q2"]) == []


def test_a_cutoff_after_everything_yields_all_of_it():
    shards = ["2015Q1", "2026Q3"]
    assert E.shards_up_to("2030-01-01", shards) == shards


def test_shards_up_to_accepts_a_date_object_as_well_as_a_string():
    import datetime as dt
    shards = ["2015Q1", "2015Q2", "2016Q1"]
    assert E.shards_up_to(dt.date(2015, 12, 31), shards) == ["2015Q1", "2015Q2"]


# --- a vintage serves decisions until the next vintage exists ------------------------------------

def test_extra_years_extends_the_window_past_the_cutoff():
    """A vintage is the model you hold until the next one arrives, so it must cover the calls that
    arrive in between. At extra_years=0 the 2015 vintage cannot see the first half of 2016, which makes
    a mid-2016 decision blind to 5.7% of chunks and specifically the most recent six months."""
    shards = ["2015Q3", "2015Q4", "2016Q1", "2016Q2", "2016Q4", "2017Q1"]
    assert E.shards_up_to("2015-12-31", shards, extra_years=0) == ["2015Q3", "2015Q4"]
    assert E.shards_up_to("2015-12-31", shards, extra_years=1) == [
        "2015Q3", "2015Q4", "2016Q1", "2016Q2", "2016Q4"]


def test_the_extended_horizon_is_a_year_end_not_a_rolling_twelve_months():
    """From a 2016 cutoff with one extra year the horizon is 2017-12-31, so every 2017 quarter is in and
    2018Q1 is out. A rolling twelve months would have cut inside 2017 and left a vintage covering part of
    a year, which is harder to reason about and does not line up with the next vintage's arrival."""
    shards = ["2016Q4", "2017Q1", "2017Q4", "2018Q1"]
    assert E.shards_up_to("2016-12-31", shards, extra_years=1) == ["2016Q4", "2017Q1", "2017Q4"]


def test_two_extra_years_reaches_further():
    shards = ["2015Q4", "2016Q4", "2017Q4", "2018Q1"]
    assert E.shards_up_to("2015-12-31", shards, extra_years=2) == ["2015Q4", "2016Q4", "2017Q4"]


def test_extra_years_beyond_the_corpus_is_harmless():
    shards = ["2015Q4", "2016Q4"]
    assert E.shards_up_to("2015-12-31", shards, extra_years=99) == shards
