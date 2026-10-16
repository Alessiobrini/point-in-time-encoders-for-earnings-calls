"""Edge cases for the held-out masked language modeling loss.

The whole point of this module is that a difference of a few thousandths between two vintages is
meaningful, so the things that could manufacture a difference that size are what the tests cover: the
mask pattern varying by model, batch-weighted averaging, and scoring on training text.
"""
import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from ecalls.train import evaluate as EV
from ecalls.train import pretrain as PT
from ecalls.train import pretrain_data as PD
from ecalls.train import tokenizer as T


@pytest.fixture
def held(tmp_path, monkeypatch):
    """A trained model and a held-out pack, both tiny."""
    for mod in (PD, T):
        monkeypatch.setattr(mod.P, "CHUNKS_DIR", tmp_path / "chunks")
        monkeypatch.setattr(mod.P, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(mod.P, "chunks_shard", lambda s: tmp_path / "chunks" / f"chunks_{s}.parquet")
    monkeypatch.setattr(PD, "PRETRAIN_DIR", tmp_path / "processed" / "pretrain")
    monkeypatch.setattr(T, "TOKENIZER_DIR", tmp_path / "processed" / "tokenizers")
    monkeypatch.setattr(PT, "MODELS_DIR", tmp_path / "processed" / "models")
    monkeypatch.setattr(EV, "MODELS_DIR", tmp_path / "processed" / "models", raising=False)
    from ecalls.embed import encode as E
    monkeypatch.setattr(E.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    PD.P.CHUNKS_DIR.mkdir(parents=True)

    texts = [f"margin freight inventory guidance item {i} " * 6 for i in range(600)]
    pl.DataFrame({"keydevid": [i // 20 for i in range(600)],
                  "chunk_id": [f"c{i:04d}" for i in range(600)], "text": texts,
                  "n_chars": [len(t) for t in texts]}).write_parquet(PD.P.chunks_shard("2010Q1"))
    T.fit("2010-12-31", vocab_size=512, name="ev-tok")
    train = PD.pack_window("2010-12-31", "ev-tok", context=64, quiet=True)
    ho = PD.pack_window("2010-12-31", "ev-tok", context=64, quiet=True, holdout_only=True)
    r = PT.train(train["name"], "ev-tok", max_steps=2, batch_sequences=2, accum=1, device="cpu",
                 checkpoint_every=1, log_every=100)
    return ho["name"], train["name"], r["name"]


def test_scoring_a_training_pack_is_refused(held):
    """The gate. A loss measured on training text is a memorization score wearing the same units."""
    ho, train, _ = held
    with pytest.raises(ValueError, match="training pack"):
        EV.evaluate(train, device="cpu", n_sequences=8, batch=4)


def test_the_held_out_pack_is_accepted_and_reports_the_vintage_cutoff(held):
    ho, _, name = held
    r = EV.evaluate(ho, models=[name], device="cpu", n_sequences=8, batch=4)
    assert len(r["models"]) == 1
    row = r["models"][0]
    assert row["model"] == name
    assert row["cutoff"] == "2010-12-31"
    assert row["loss"] > 0
    assert row["n_masked_positions"] > 0


def test_the_same_model_scored_twice_gives_the_identical_loss(held):
    """The mask is seeded from the batch index alone. Seeding it from anything that varies per model
    would put noise the size of the measured effect between two identical models."""
    ho, _, name = held
    a = EV.evaluate(ho, models=[name], device="cpu", n_sequences=16, batch=4)["models"][0]["loss"]
    b = EV.evaluate(ho, models=[name], device="cpu", n_sequences=16, batch=4)["models"][0]["loss"]
    assert a == b


def test_two_models_are_asked_to_predict_the_same_tokens(held):
    """Recorded directly off the labels rather than inferred from the losses agreeing."""
    ho, _, name = held
    arr, meta = PD.load_window(ho)
    tok = T.load("ev-tok")
    specials = [tok.token_to_id(s) for s in T.SPECIALS]
    seen = []

    class Recorder:
        def __call__(self, input_ids=None, labels=None):
            seen.append(labels.clone())
            return SimpleNamespace(loss=torch.tensor(1.0))

    for _ in range(2):
        EV.mlm_loss(Recorder(), arr, tok.token_to_id("[MASK]"), meta["vocab_size"], specials,
                    "cpu", n_sequences=16, batch=4)
    half = len(seen) // 2
    assert half, "nothing was scored"
    for i in range(half):
        assert torch.equal(seen[i], seen[half + i]), f"batch {i} was masked differently"


def test_the_loss_is_token_weighted_not_batch_weighted(held):
    """A batch with few masked positions must not count as much as a full one. Masked counts differ
    between batches because special tokens are never chosen."""
    ho, _, _ = held
    arr, meta = PD.load_window(ho)
    tok = T.load("ev-tok")
    specials = [tok.token_to_id(s) for s in T.SPECIALS]
    losses, counts = [], []

    class Varying:
        """Returns a different loss per batch, so the two weightings cannot coincide."""
        def __call__(self, input_ids=None, labels=None):
            k = int((labels != -100).sum())
            v = 1.0 + 2.0 * len(losses)          # 1.0, 3.0, 5.0, ...
            losses.append(v)
            counts.append(k)
            return SimpleNamespace(loss=torch.tensor(v))

    r = EV.mlm_loss(Varying(), arr, tok.token_to_id("[MASK]"), meta["vocab_size"], specials,
                    "cpu", n_sequences=24, batch=4)
    token_weighted = sum(v * k for v, k in zip(losses, counts)) / sum(counts)
    batch_weighted = sum(losses) / len(losses)
    assert r["n_masked_positions"] == sum(counts)
    assert r["loss"] == pytest.approx(token_weighted)
    # The fixture must actually distinguish the two, or this test passes on a coincidence.
    assert token_weighted != pytest.approx(batch_weighted), "batches had equal masked counts"


def test_a_model_without_a_checkpoint_raises_rather_than_scoring_random_weights(held, tmp_path):
    """An untrained model has a loss near log(vocab), which looks like a result rather than an error."""
    ho, _, _ = held
    (PT.MODELS_DIR / "ecalls-mlm-19991231").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        EV.evaluate(ho, models=["ecalls-mlm-19991231"], device="cpu", n_sequences=8, batch=4)


# --- the sample must span the set, not its head -------------------------------------------------

def test_the_sample_spans_the_whole_set_rather_than_its_prefix():
    """A packed corpus is in calendar order, so a prefix is the oldest text rather than a sample.

    This is the defect that made a 2010-window held-out set and a full-span one give byte-identical
    losses across all fifteen vintages: the 2010 set is the leading slice of the full-span one, so a
    prefix of each is the same text. Nothing failed; fifteen numbers simply matched to four decimals.
    """
    idx = EV.eval_indices(100_000, 1_000)
    assert len(idx) == 1_000
    assert idx[0] == 0
    assert idx[-1] == 99_999
    # and the sample is not concentrated anywhere: each tenth of the array gets about a tenth
    hist = np.histogram(idx, bins=10, range=(0, 100_000))[0]
    assert hist.min() >= 90, hist


def test_asking_for_everything_returns_everything_in_order():
    idx = EV.eval_indices(500, 10_000)
    assert len(idx) == 500
    assert list(idx) == list(range(500))


def test_two_sets_where_one_is_a_prefix_of_the_other_give_different_samples():
    """The direct statement of the bug: with a prefix rule these two would be the same indices."""
    small = EV.eval_indices(44_100, 20_000)
    large = EV.eval_indices(375_818, 20_000)
    assert small[-1] == 44_099
    assert large[-1] == 375_817
    assert not np.array_equal(small, large)


def test_an_empty_set_raises_rather_than_reporting_a_loss_of_zero():
    with pytest.raises(ValueError, match="nothing to evaluate"):
        EV.eval_indices(0, 100)


def test_the_result_records_the_span_it_actually_covered(held):
    """So a later reader can tell a full-set evaluation from a sampled one without rerunning it."""
    ho, _, name = held
    r = EV.evaluate(ho, models=[name], device="cpu", n_sequences=8, batch=4)
    row = r["models"][0]
    assert row["n_sequences"] == 8
    assert row["n_available"] >= row["n_sequences"]
    assert row["last_index"] >= row["n_sequences"] - 1
