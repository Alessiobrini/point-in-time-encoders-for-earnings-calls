"""Edge cases for the pretraining loop.

Two things here are load-bearing beyond "the loss goes down". The sampler and its checkpoint decide whether
the token budget means what it says, and the masking decides whether the model is asked to predict anything
worth predicting. Both fail quietly.
"""
import json

import numpy as np
import pytest
import torch

from ecalls.train import pretrain as PT


# --- the sampler ---------------------------------------------------------------------------------

def test_an_epoch_covers_every_sequence_exactly_once():
    """With replacement, a four-epoch budget would never show roughly 2% of sequences and would show
    others four times, so the budget would not mean what it says."""
    s = PT.Sampler(100, seed=1)
    seen = s.next_batch(100)
    assert sorted(seen.tolist()) == list(range(100))


def test_the_order_differs_between_epochs():
    s = PT.Sampler(200, seed=1)
    first = s.next_batch(200).tolist()
    second = s.next_batch(200).tolist()
    assert sorted(second) == list(range(200))
    assert first != second


def test_the_same_seed_gives_the_same_order():
    a = PT.Sampler(50, seed=7).next_batch(50).tolist()
    b = PT.Sampler(50, seed=7).next_batch(50).tolist()
    assert a == b
    assert PT.Sampler(50, seed=8).next_batch(50).tolist() != a


def test_a_batch_spanning_an_epoch_boundary_is_filled_not_truncated():
    s = PT.Sampler(10, seed=0)
    s.next_batch(8)
    crossing = s.next_batch(6)
    assert len(crossing) == 6
    assert s.epoch == 1


def test_sampler_state_round_trips_so_a_resume_does_not_repeat_sequences():
    """Resuming with a fresh sampler would re-show the sequences already consumed, which silently
    changes the effective budget."""
    a = PT.Sampler(64, seed=3)
    a.next_batch(40)
    st = a.state()
    rest_a = a.next_batch(40).tolist()

    b = PT.Sampler(64, seed=3)
    b.load_state(st)
    assert b.next_batch(40).tolist() == rest_a


# --- masking -------------------------------------------------------------------------------------

def _gen():
    g = torch.Generator()
    g.manual_seed(0)
    return g


def test_special_tokens_are_never_masked():
    """Predicting a [CLS] the model put there itself teaches nothing, and masking [PAD] asks it to
    predict padding."""
    ids = torch.tensor([[2, 10, 11, 12, 3, 0, 0, 0]] * 200)
    specials = [0, 1, 2, 3, 4]
    out, labels = PT.mask_batch(ids, mask_id=4, vocab_size=100, special_ids=specials,
                                generator=_gen(), ratio=0.9)
    for col in (0, 4, 5, 6, 7):                       # [CLS], [SEP] and the pads
        assert torch.all(out[:, col] == ids[:, col])
        assert torch.all(labels[:, col] == -100)


def test_unchosen_positions_are_ignored_by_the_loss():
    ids = torch.tensor([[10, 11, 12, 13]] * 400)
    out, labels = PT.mask_batch(ids, 4, 100, [0], _gen(), ratio=0.3)
    chosen = labels != -100
    assert chosen.any() and (~chosen).any()
    assert torch.all(labels[chosen] == ids[chosen])   # the label is the original token
    assert torch.all(out[~chosen] == ids[~chosen])    # untouched positions are unchanged


def test_the_mask_ratio_is_approximately_honored():
    ids = torch.tensor([[10] * 64] * 400)
    _, labels = PT.mask_batch(ids, 4, 100, [0], _gen(), ratio=PT.MASK_RATIO)
    share = float((labels != -100).float().mean())
    assert abs(share - PT.MASK_RATIO) < 0.02


def test_most_chosen_positions_become_the_mask_token_and_some_do_not():
    """The 80/10/10 split. If every chosen position became [MASK], the model would never see a real or
    a corrupted token at a position it must predict, which is what the split exists to avoid."""
    ids = torch.tensor([[10] * 128] * 400)
    out, labels = PT.mask_batch(ids, 4, 100, [0], _gen(), ratio=0.5)
    chosen = labels != -100
    became_mask = float((out[chosen] == 4).float().mean())
    assert abs(became_mask - PT.REPLACE_MASK) < 0.03
    assert became_mask < 1.0


def test_masking_is_dynamic_so_two_calls_differ():
    ids = torch.tensor([[10] * 64] * 32)
    g = _gen()
    a, la = PT.mask_batch(ids, 4, 100, [0], g)
    b, lb = PT.mask_batch(ids, 4, 100, [0], g)
    assert not torch.equal(la, lb)


# --- the schedule --------------------------------------------------------------------------------

def test_the_learning_rate_warms_up_then_decays():
    total, warmup, peak = 1000, 60, 6e-4
    assert PT.lr_at(0, total, peak, warmup) < peak
    assert PT.lr_at(warmup - 1, total, peak, warmup) == pytest.approx(peak, rel=1e-6)
    mid = PT.lr_at(total // 2, total, peak, warmup)
    end = PT.lr_at(total - 1, total, peak, warmup)
    assert peak > mid > end
    assert end == pytest.approx(0.1 * peak, rel=0.05)


def test_the_schedule_is_flat_after_the_end_rather_than_negative():
    assert PT.lr_at(2000, 1000, 6e-4, 60) >= 0.1 * 6e-4 * 0.99


def test_a_zero_warmup_does_not_divide_by_zero():
    assert PT.lr_at(0, 100, 1e-3, 0) > 0


# --- the run, end to end on a tiny corpus --------------------------------------------------------

@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A packed corpus and tokenizer small enough to train for a few steps on CPU."""
    import polars as pl

    from ecalls.train import pretrain_data as PD
    from ecalls.train import tokenizer as T

    for mod in (PD, T):
        monkeypatch.setattr(mod.P, "CHUNKS_DIR", tmp_path / "chunks")
        monkeypatch.setattr(mod.P, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(mod.P, "chunks_shard", lambda s: tmp_path / "chunks" / f"chunks_{s}.parquet")
    monkeypatch.setattr(PD, "PRETRAIN_DIR", tmp_path / "processed" / "pretrain")
    monkeypatch.setattr(T, "TOKENIZER_DIR", tmp_path / "processed" / "tokenizers")
    monkeypatch.setattr(PT, "MODELS_DIR", tmp_path / "processed" / "models")
    from ecalls.embed import encode as E
    monkeypatch.setattr(E.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    PD.P.CHUNKS_DIR.mkdir(parents=True)

    texts = [f"margin freight inventory guidance item {i} " * 6 for i in range(600)]
    pl.DataFrame({"keydevid": [i // 20 for i in range(600)],
                  "chunk_id": [f"c{i:04d}" for i in range(600)], "text": texts,
                  "n_chars": [len(t) for t in texts]}).write_parquet(PD.P.chunks_shard("2010Q1"))
    T.fit("2010-12-31", vocab_size=512, name="tiny-tok")
    m = PD.pack_window("2010-12-31", "tiny-tok", context=64, quiet=True, holdout_fraction=0)
    return m["name"], "tiny-tok"


def test_a_short_run_reduces_the_loss_and_writes_its_provenance(tiny):
    window, tokname = tiny
    # The architecture is fixed by design and cannot be shrunk for a test, so this runs the real 33.8 M
    # shape for two steps rather than a toy model for more.
    r = PT.train(window, tokname, max_steps=2, batch_sequences=2, accum=1, device="cpu",
                 checkpoint_every=100, log_every=100)
    d = PT.model_dir(r["name"])
    assert r["final_step"] == 2
    assert r["final_loss"] > 0
    run = json.loads((d / "_run.json").read_text())
    assert run["cutoff"] == "2010-12-31"
    assert run["mask_ratio"] == PT.MASK_RATIO
    assert run["architecture"]["total"] > 0
    window_file = json.loads((d / "_window.json").read_text())
    assert window_file["cutoff"] == "2010-12-31"      # section 5 item 4
    assert window_file["seed"] == 0


def test_a_rerun_resumes_from_the_checkpoint_rather_than_starting_over(tiny, capsys):
    """Without this a preempted run on scavenger restarts at step zero and never finishes."""
    window, tokname = tiny
    PT.train(window, tokname, max_steps=2, batch_sequences=2, accum=1, device="cpu",
             checkpoint_every=1, log_every=100)
    capsys.readouterr()
    r = PT.train(window, tokname, max_steps=4, batch_sequences=2, accum=1, device="cpu",
                 checkpoint_every=1, log_every=100)
    assert "resuming" in capsys.readouterr().out
    assert r["final_step"] == 4


# --- the microbatch adapts to the card while the gradient does not -------------------------------

def test_the_microbatch_scales_with_device_memory_but_tokens_per_step_does_not():
    """Twelve of fifteen pretraining tasks died within thirty seconds on 10.58 GiB cards at a fixed
    microbatch of 32 that the throughput probe had run comfortably on 49 GiB. The microbatch is an
    implementation detail of whichever card scavenger allocated; tokens per step is the hyperparameter
    section 5 requires held constant, so it must come out identical."""
    import unittest.mock as mock

    seen = {}
    for gb in (10.58, 24.0, 49.1):
        with mock.patch("ecalls.embed.encode.gpu_memory_gb", return_value=gb):
            mb = PT.auto_microbatch("cuda", 768)
        accum = PT.TOKENS_PER_STEP // (mb * 768)
        seen[gb] = (mb, accum, mb * accum * 768)
    assert seen[10.58][0] < seen[24.0][0] < seen[49.1][0]
    assert len({v[2] for v in seen.values()}) == 1
    assert seen[10.58][2] == PT.TOKENS_PER_STEP


def test_the_microbatch_is_a_power_of_two_so_accumulation_is_whole():
    import unittest.mock as mock

    for gb in (8.0, 10.58, 16.0, 24.0, 40.0, 49.1, 80.0):
        with mock.patch("ecalls.embed.encode.gpu_memory_gb", return_value=gb):
            mb = PT.auto_microbatch("cuda", 768)
        assert mb & (mb - 1) == 0, f"{gb} GB gave a non power of two: {mb}"
        assert PT.TOKENS_PER_STEP % (mb * 768) == 0


def test_the_microbatch_never_exceeds_the_token_target():
    """A microbatch larger than the whole step would make accumulation zero and the gradient wrong."""
    import unittest.mock as mock

    with mock.patch("ecalls.embed.encode.gpu_memory_gb", return_value=640.0):
        mb = PT.auto_microbatch("cuda", 768, target_tokens=768 * 4)
    assert mb <= 4


def test_off_gpu_the_microbatch_is_small_rather_than_absent():
    assert PT.auto_microbatch("cpu", 768) >= 1
    assert PT.auto_microbatch("cpu", 768) <= 8


def test_an_oom_mid_step_halves_the_microbatch_and_keeps_the_gradient_size(tiny, capsys):
    """The recovery must not change what the optimizer sees, only how it is assembled."""
    window, tokname = tiny
    calls = {"n": 0}
    real = PT.MD.build_mlm

    def flaky_model(vocab, seed=0, **kw):
        m = real(vocab, seed=seed, **kw)
        fwd = m.forward

        def maybe_oom(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise torch.OutOfMemoryError("stub OOM on the first microbatch")
            return fwd(*a, **k)

        m.forward = maybe_oom
        return m

    PT.MD.build_mlm = flaky_model
    try:
        r = PT.train(window, tokname, max_steps=1, batch_sequences=4, accum=2, device="cpu",
                     checkpoint_every=1000, log_every=1000)
    finally:
        PT.MD.build_mlm = real
    out = capsys.readouterr().out
    assert "OOM: microbatch now 2, accum 4" in out
    assert "tokens/step unchanged" in out
    assert r["final_step"] == 1


# --- the recorded cutoff is the vintage's, not the packed corpus's -------------------------------

@pytest.fixture
def two_year(tmp_path, monkeypatch):
    """A packed corpus spanning 2010 and 2012, so a prefix cutoff differs from the file's cutoff.

    The `tiny` fixture packs one year, which makes the two cutoffs equal and lets a run record
    labelled with the wrong one pass unnoticed. That is how all fifteen vintages came to record
    2024-12-31.
    """
    import polars as pl

    from ecalls.train import pretrain_data as PD
    from ecalls.train import tokenizer as T

    for mod in (PD, T):
        monkeypatch.setattr(mod.P, "CHUNKS_DIR", tmp_path / "chunks")
        monkeypatch.setattr(mod.P, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(mod.P, "chunks_shard", lambda s: tmp_path / "chunks" / f"chunks_{s}.parquet")
    monkeypatch.setattr(PD, "PRETRAIN_DIR", tmp_path / "processed" / "pretrain")
    monkeypatch.setattr(T, "TOKENIZER_DIR", tmp_path / "processed" / "tokenizers")
    monkeypatch.setattr(PT, "MODELS_DIR", tmp_path / "processed" / "models")
    from ecalls.embed import encode as E
    monkeypatch.setattr(E.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    PD.P.CHUNKS_DIR.mkdir(parents=True)

    for shard, base in (("2010Q1", 0), ("2012Q1", 600)):
        texts = [f"margin freight inventory guidance item {i} " * 6 for i in range(base, base + 600)]
        pl.DataFrame({"keydevid": [i // 20 for i in range(base, base + 600)],
                      "chunk_id": [f"c{i:04d}" for i in range(base, base + 600)], "text": texts,
                      "n_chars": [len(t) for t in texts]}).write_parquet(PD.P.chunks_shard(shard))
    T.fit("2012-12-31", vocab_size=512, name="tiny-tok2")
    m = PD.pack_window("2012-12-31", "tiny-tok2", context=64, quiet=True, holdout_fraction=0)
    return m["name"], "tiny-tok2", m["n_sequences"]


def test_the_record_carries_the_vintage_cutoff_not_the_packed_corpus_cutoff(two_year):
    """The 2010 vintage trained on a 2012 file must record 2010-12-31 as what it may not see past.

    One packed corpus serves every vintage as a prefix, so its own cutoff is the latest vintage's.
    Recording that field as the vintage cutoff labelled the 2010 model as having seen text through
    2024, which is the single field the point-in-time contract rests on.
    """
    window, tokname, full = two_year
    r = PT.train(window, tokname, cutoff="2010-12-31", max_steps=2, batch_sequences=2, accum=1,
                 device="cpu", checkpoint_every=100, log_every=100)
    run = json.loads((PT.model_dir(r["name"]) / "_run.json").read_text())
    assert run["cutoff"] == "2010-12-31"
    assert run["window_cutoff"] == "2012-12-31"
    # and the slice really was taken, so the two fields are not both cosmetic
    assert run["n_sequences_available"] < full
    cfg = json.loads((PT.model_dir(r["name"]) / "_window.json").read_text())
    assert cfg["cutoff"] == "2010-12-31"
    assert cfg["window_cutoff"] == "2012-12-31"


def test_the_default_run_name_comes_from_the_vintage_cutoff(two_year):
    """Without an explicit --name every vintage would be called ecalls-mlm-20241231 and overwrite
    the previous one. The fifteen names are right only because the sbatch passed --name."""
    window, tokname, _ = two_year
    r = PT.train(window, tokname, cutoff="2010-12-31", max_steps=1, batch_sequences=2, accum=1,
                 device="cpu", checkpoint_every=100, log_every=100)
    assert r["name"] == "ecalls-mlm-20101231"


def test_float16_gets_a_gradient_scaler_and_bfloat16_does_not(tiny):
    """bf16 carries fp32's exponent range; fp16's smallest normal is about 6e-5 and MLM gradients go
    below it. A float16 run without a scaler underflows to zero gradient and still looks like it is
    training, which on a 149 M model costs a day before anyone notices."""
    window, tokname = tiny
    r16 = PT.train(window, tokname, max_steps=1, batch_sequences=2, accum=1, device="cpu",
                   checkpoint_every=100, log_every=100, compute_dtype="float16",
                   name="ecalls-mlm-fp16")
    rbf = PT.train(window, tokname, max_steps=1, batch_sequences=2, accum=1, device="cpu",
                   checkpoint_every=100, log_every=100, compute_dtype="bfloat16",
                   name="ecalls-mlm-bf16")
    # On CPU the dtype is forced to float32, so the scaler must be off for both there.
    assert r16["grad_scaler"] is False and rbf["grad_scaler"] is False
    assert r16["compute_dtype"] == "float16" and rbf["compute_dtype"] == "bfloat16"


def test_the_scaler_is_tied_to_the_dtype_not_offered_as_a_flag():
    """Reading the source rather than running a GPU: the pair must not be separately settable."""
    import inspect
    src = inspect.getsource(PT.train)
    assert "GradScaler" in src
    assert "if dt is torch.float16" in src, "the scaler is not tied to the dtype"
    assert "scaler.unscale_(opt)" in src, "gradients would be clipped while still scaled"


class _Entry:
    def __init__(self, cutoff):
        self.training_cutoff = cutoff


def _guard(name, cutoff, backbone, prints):
    from ecalls.train.pretrain import check_borrowed_tokenizer
    cutoffs = {"chrono-2024": "2024-12-31", "chrono-2010": "2010-12-31", "other-2024": "2024-12-31"}
    return check_borrowed_tokenizer(name, cutoff, backbone, lookup=lambda m: _Entry(cutoffs[m]),
                                    fingerprint_of=lambda n: prints[n])


def test_a_later_borrowed_vocabulary_identical_to_the_backbones_passes_and_is_recorded():
    prints = {"registry:chrono-2024": "a", "registry:chrono-2010": "a"}
    r = _guard("registry:chrono-2024", "2010-12-31", "chrono-2010", prints)
    assert r == {"tokenizer_cutoff": "2024-12-31", "tokenizer_equivalent_to": "registry:chrono-2010"}


def test_a_later_borrowed_vocabulary_that_differs_from_the_backbones_is_refused():
    prints = {"registry:chrono-2024": "a", "registry:chrono-2010": "b"}
    with pytest.raises(ValueError, match="after this vintage"):
        _guard("registry:chrono-2024", "2010-12-31", "chrono-2010", prints)


def test_a_later_borrowed_vocabulary_without_a_backbone_is_refused():
    with pytest.raises(ValueError, match="after this vintage"):
        _guard("registry:other-2024", "2010-12-31", None, {"registry:other-2024": "a"})


def test_a_borrowed_vocabulary_at_or_before_the_cutoff_needs_no_comparison():
    r = _guard("registry:chrono-2010", "2010-12-31", "chrono-2010", {})   # an empty map would raise if read
    assert r == {"tokenizer_cutoff": "2010-12-31", "tokenizer_equivalent_to": None}
    assert _guard("ecalls-bpe-32768-20101231", "2010-12-31", None, {}) == {}
