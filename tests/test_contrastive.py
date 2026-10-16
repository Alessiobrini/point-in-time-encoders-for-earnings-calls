"""Edge cases for the contrastive stage.

The failure modes worth testing are the ones that produce a plausible number rather than an error: a
model trained on the calls the benchmark scores, a batch whose negatives are all easy, a duplicate
answer counted as a negative, and an exported model that is not the model that was trained.
"""
import json
import pathlib

import numpy as np
import polars as pl
import pytest
import torch

from ecalls.train import contrastive as CT
from ecalls.train import pretrain as PT
from ecalls.train import tokenizer as T


# --- the sampler ---------------------------------------------------------------------------------

def test_an_epoch_covers_every_call_exactly_once():
    s = CT.CallSampler(10, per_batch=5, seed=0)
    seen = list(s.next()) + list(s.next())
    assert sorted(seen) == list(range(10))
    assert s.epoch == 0


def test_a_batch_spanning_an_epoch_boundary_is_filled_from_the_next_epoch():
    """A short batch would carry fewer negatives than every other, which changes the loss scale."""
    s = CT.CallSampler(7, per_batch=5, seed=0)
    s.next()
    b = s.next()
    assert len(b) == 5
    assert s.epoch == 1


def test_sampler_state_round_trips():
    a = CT.CallSampler(20, per_batch=4, seed=3)
    a.next(); a.next()
    b = CT.CallSampler(20, per_batch=4, seed=3)
    b.load_state(a.state())
    assert list(a.next()) == list(b.next())


def test_resuming_against_a_different_call_count_is_refused():
    """Otherwise a resume samples a different set than the run it claims to continue."""
    a = CT.CallSampler(20, per_batch=4, seed=0)
    b = CT.CallSampler(21, per_batch=4, seed=0)
    with pytest.raises(ValueError, match="checkpoint"):
        b.load_state(a.state())


# --- batching: the hard negatives are the point --------------------------------------------------

def _frame(n_calls=6, per_call=5):
    rows = [{"keydevid": c, "query_id": f"q{c}-{i}", "answer_id": f"a{c}-{i}",
             "text": f"question {i} for call {c}", "answer_text": f"answer {i} for call {c}",
             "shard": "2010Q1"}
            for c in range(n_calls) for i in range(per_call)]
    return pl.DataFrame(rows)


def test_a_batch_holds_several_pairs_from_each_chosen_call():
    """Without this the in-call negatives are absent and every negative is separable on vocabulary."""
    df = _frame()
    groups, calls = CT.group_pairs(df)
    rng = np.random.default_rng(0)
    rows = CT.draw_batch(groups, np.array([0, 2, 4]), pairs_per_call=3, rng=rng)
    assert len(rows) == 9
    got = df["keydevid"].to_numpy()[rows]
    counts = {int(c): int((got == c).sum()) for c in set(got.tolist())}
    assert counts == {0: 3, 2: 3, 4: 3}


def test_a_call_with_fewer_pairs_than_asked_for_contributes_all_of_them():
    """Short calls are kept rather than dropped, which would bias training toward long Q&A sessions."""
    df = pl.concat([_frame(n_calls=2, per_call=5),
                    pl.DataFrame([{"keydevid": 99, "query_id": "q99", "answer_id": "a99",
                                   "text": "lone question", "answer_text": "lone answer",
                                   "shard": "2010Q1"}])])
    groups, calls = CT.group_pairs(df)
    i = int(np.where(calls == 99)[0][0])
    rows = CT.draw_batch(groups, np.array([i]), pairs_per_call=4, rng=np.random.default_rng(0))
    assert len(rows) == 1


def test_grouping_preserves_every_pair():
    df = _frame(n_calls=7, per_call=3)
    groups, calls = CT.group_pairs(df)
    assert sorted(np.concatenate(groups).tolist()) == list(range(df.height))
    assert len(calls) == 7


# --- the loss ------------------------------------------------------------------------------------

def _unit(v):
    return torch.nn.functional.normalize(torch.tensor(v, dtype=torch.float32), p=2, dim=1)


def test_the_loss_falls_when_the_gold_pairs_align():
    q = _unit([[1.0, 0.0], [0.0, 1.0]])
    aligned = _unit([[1.0, 0.0], [0.0, 1.0]])
    swapped = _unit([[0.0, 1.0], [1.0, 0.0]])
    ids = np.array(["a", "b"])
    assert float(CT.info_nce(q, aligned, ids)) < float(CT.info_nce(q, swapped, ids))


def test_a_repeated_answer_is_not_counted_as_a_negative():
    """Two questions in a call can share a gold answer. Counting it as a negative for the other
    question trains the model to separate two texts that belong together."""
    q = _unit([[1.0, 0.0], [1.0, 0.0]])
    a = _unit([[1.0, 0.0], [1.0, 0.0]])
    dup = np.array(["same", "same"])
    distinct = np.array(["one", "two"])
    # With the duplicate masked, each anchor faces no competing candidate and the loss is zero.
    assert float(CT.info_nce(q, a, dup)) == pytest.approx(0.0, abs=1e-6)
    # Unmasked, two identical candidates split the probability and the loss is log 2.
    assert float(CT.info_nce(q, a, distinct)) == pytest.approx(np.log(2), abs=1e-4)


def test_the_diagonal_survives_the_duplicate_mask():
    """Masking the diagonal too would leave the gold pair out of the numerator and the loss infinite."""
    q = _unit([[1.0, 0.0], [0.0, 1.0]])
    a = _unit([[1.0, 0.0], [0.0, 1.0]])
    loss = float(CT.info_nce(q, a, np.array(["same", "same"])))
    assert np.isfinite(loss)


def test_a_lower_temperature_sharpens_the_loss():
    """The temperature has to reach the similarities, or it is a recorded hyperparameter doing nothing."""
    q = _unit([[1.0, 0.1], [0.1, 1.0]])
    a = _unit([[1.0, 0.3], [0.3, 1.0]])
    ids = np.array(["a", "b"])
    assert float(CT.info_nce(q, a, ids, 0.01)) < float(CT.info_nce(q, a, ids, 1.0))


# --- the real thing ------------------------------------------------------------------------------

@pytest.fixture
def stage(tmp_path, monkeypatch):
    """A pretrained backbone, a training pair set and a held-out pair set, all tiny."""
    from ecalls.train import pairs as PR
    from ecalls.train import pretrain_data as PD

    for mod in (PD, T, PR):
        monkeypatch.setattr(mod.P, "CHUNKS_DIR", tmp_path / "chunks")
        monkeypatch.setattr(mod.P, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(mod.P, "chunks_shard", lambda s: tmp_path / "chunks" / f"chunks_{s}.parquet")
    monkeypatch.setattr(PD, "PRETRAIN_DIR", tmp_path / "processed" / "pretrain")
    monkeypatch.setattr(PR, "PAIRS_DIR", tmp_path / "processed" / "pairs")
    monkeypatch.setattr(T, "TOKENIZER_DIR", tmp_path / "processed" / "tokenizers")
    monkeypatch.setattr(PT, "MODELS_DIR", tmp_path / "processed" / "models")
    monkeypatch.setattr(CT, "MODELS_DIR", tmp_path / "processed" / "models")
    from ecalls.embed import encode as E
    monkeypatch.setattr(E.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    PD.P.CHUNKS_DIR.mkdir(parents=True)

    texts = [f"margin freight inventory guidance item {i} " * 6 for i in range(400)]
    pl.DataFrame({"keydevid": [i // 20 for i in range(400)],
                  "chunk_id": [f"c{i:04d}" for i in range(400)], "text": texts,
                  "n_chars": [len(t) for t in texts]}).write_parquet(PD.P.chunks_shard("2010Q1"))
    T.fit("2010-12-31", vocab_size=512, name="ct-tok")
    w = PD.pack_window("2010-12-31", "ct-tok", context=64, quiet=True)
    mlm = PT.train(w["name"], "ct-tok", max_steps=1, batch_sequences=2, accum=1, device="cpu",
                   checkpoint_every=1, log_every=100)

    # Pair sets written directly: the real builder reads the components layer, which is not what
    # these tests are about.
    for nm, calls, ho in (("qa-pairs-20101231", range(0, 8), False),
                          ("qa-pairs-20101231-holdout", range(8, 11), True)):
        d = PR.pairs_dir(nm)
        d.mkdir(parents=True, exist_ok=True)
        rows = [{"keydevid": c, "query_id": f"q{c}-{i}", "answer_id": f"a{c}-{i}",
                 "text": f"what drove the margin in segment {i} of call {c}",
                 "answer_text": f"pricing and freight moved segment {i} in call {c}",
                 "shard": "2010Q1"}
                for c in calls for i in range(6)]
        pl.DataFrame(rows).write_parquet(d / "pairs.parquet")
        (d / "_meta.json").write_text(json.dumps(
            {"name": nm, "cutoff": "2010-12-31", "holdout_only": ho, "n_pairs": len(rows),
             "holdout_fraction": 0.1, "holdout_salt": "ecalls-holdout-v1"}) + "\n")
    return mlm["name"], "qa-pairs-20101231", "qa-pairs-20101231-holdout"


def test_training_on_the_held_out_pair_set_is_refused(stage):
    """The gate. It would train on the calls the benchmark scores, making 0.5378 unbeatable and
    meaningless at the same time."""
    mlm, train_pairs, ho_pairs = stage
    with pytest.raises(ValueError, match="held-out pair set"):
        CT.train(mlm, ho_pairs, cutoff="2010-12-31", device="cpu", total_steps=1)


def test_a_training_set_passed_as_the_progress_check_is_refused(stage):
    mlm, train_pairs, ho_pairs = stage
    with pytest.raises(ValueError, match="not a held-out one"):
        CT.train(mlm, train_pairs, cutoff="2010-12-31", holdout_pairs=train_pairs,
                 device="cpu", total_steps=1)


def test_a_backbone_from_a_different_vintage_is_refused(stage):
    """A retriever inherits its backbone's cutoff, so training the 2015 vintage from the 2010
    backbone would record a cutoff the weights do not have."""
    mlm, train_pairs, _ = stage
    with pytest.raises(ValueError, match="pretrained to"):
        CT.train(mlm, train_pairs, cutoff="2015-12-31", device="cpu", total_steps=1)


def test_a_short_run_trains_writes_provenance_and_exports_a_loadable_model(stage):
    mlm, train_pairs, ho_pairs = stage
    r = CT.train(mlm, train_pairs, cutoff="2010-12-31", holdout_pairs=ho_pairs, device="cpu",
                 total_steps=2, calls_per_batch=3, pairs_per_call=2, microbatch=4, max_length_query=32, max_length_answer=48,
                 compute_dtype="float32", checkpoint_every=1, log_every=100, eval_every=0)
    d = CT.model_dir(r["name"])
    assert r["name"] == "ecalls-ret-20101231"
    assert r["final_step"] == 2
    assert r["cutoff"] == "2010-12-31"
    assert r["backbone"] == mlm
    assert r["final_holdout_recall_at_1"] is not None
    run = json.loads((d / "_run.json").read_text())
    assert run["temperature"] == CT.TEMPERATURE
    assert run["pooling"] == "mean"
    assert (d / "hf" / "config.json").exists()


def test_the_exported_model_gives_the_same_vectors_as_the_trained_one(stage):
    """The export is what the registry, the embedding pass, the benchmark and the index all read, so
    an export that silently differs would make every downstream number belong to another model."""
    from transformers import AutoModel, AutoTokenizer

    mlm, train_pairs, ho_pairs = stage
    r = CT.train(mlm, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=1,
                 calls_per_batch=2, pairs_per_call=2, microbatch=4, max_length_query=32, max_length_answer=48, compute_dtype="float32",
                 checkpoint_every=1, log_every=100, eval_every=0)
    d = CT.model_dir(r["name"])

    trained, _ = CT.load_backbone(mlm, r["vocab_size"], "cpu")
    ck = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
    trained.load_state_dict(ck["model"])
    trained.eval()
    tok_ours = CT.hf_tokenizer(r["tokenizer"], 32)

    reloaded = AutoModel.from_pretrained(d / "hf").eval()
    tok_hf = AutoTokenizer.from_pretrained(d / "hf")

    texts = ["what drove the margin", "pricing and freight moved the segment"]
    with torch.no_grad():
        a = CT.embed(trained, tok_ours, texts, "cpu", 32, None)
        b = CT.embed(reloaded, tok_hf, texts, "cpu", 32, None)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_a_resume_continues_rather_than_restarting(stage, capsys):
    mlm, train_pairs, _ = stage
    CT.train(mlm, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=1,
             calls_per_batch=2, pairs_per_call=2, microbatch=4, max_length_query=32, max_length_answer=48, compute_dtype="float32",
             checkpoint_every=1, log_every=100, eval_every=0)
    capsys.readouterr()
    r = CT.train(mlm, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=3,
                 calls_per_batch=2, pairs_per_call=2, microbatch=4, max_length_query=32, max_length_answer=48, compute_dtype="float32",
                 checkpoint_every=1, log_every=100, eval_every=0)
    assert "resuming" in capsys.readouterr().out
    assert r["final_step"] == 3


def test_the_two_sides_are_capped_separately(stage):
    """Measured over 40,000 pairs: questions average 83 tokens and answers 187, so one cap of 256
    truncates 0.88% of questions and 22.46% of answers. A symmetric cap would also give our model a
    shorter answer budget than bge-small's 512, which is the score it has to beat."""
    mlm, train_pairs, _ = stage
    r = CT.train(mlm, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=1,
                 calls_per_batch=2, pairs_per_call=2, microbatch=4, max_length_query=32, max_length_answer=48,
                 compute_dtype="float32", checkpoint_every=1, log_every=100, eval_every=0)
    assert r["max_length_query"] == 32
    assert r["max_length_answer"] == 48
    assert r["max_length"] == 48          # what the registry reports, since evaluation caps both
    assert CT.MAX_LENGTH_ANSWER > CT.MAX_LENGTH_QUERY


def test_the_cap_passed_to_embed_reaches_the_tokenizer(stage):
    """A recorded cap that never reaches the tokenizer would leave every answer truncated at the
    tokenizer's own default, which is the failure the asymmetric caps exist to avoid."""
    mlm, _, _ = stage
    run = json.loads((PT.model_dir(mlm) / "_run.json").read_text())
    tok = CT.hf_tokenizer(run["tokenizer"], 512)
    long_text = "margin freight inventory guidance pricing volume backlog " * 60
    seen = {}

    class Recorder:
        config = None

        def __call__(self, input_ids=None, attention_mask=None, **kw):
            seen["len"] = int(input_ids.shape[1])
            return type("O", (), {"last_hidden_state": torch.zeros(input_ids.shape[0],
                                                                   input_ids.shape[1], 8)})()

    CT.embed(Recorder(), tok, [long_text], "cpu", 16, None)
    assert seen["len"] == 16
    CT.embed(Recorder(), tok, [long_text], "cpu", 64, None)
    assert seen["len"] == 64


def test_the_trainer_sends_the_answer_cap_to_the_answer_side(stage, monkeypatch):
    """Recording the two caps and honoring them inside `embed` is not enough: the trainer has to hand
    the right one to the right side. Swapping them capped every answer at the question length, which
    the other tests in this file all passed."""
    mlm, train_pairs, _ = stage
    calls = []
    real = CT.embed

    def spy(model, tok, texts, device, max_length, dtype, pooling="mean"):
        calls.append((max_length, len(texts)))
        return real(model, tok, texts, device, max_length, dtype, pooling)

    monkeypatch.setattr(CT, "embed", spy)
    CT.train(mlm, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=1,
             calls_per_batch=2, pairs_per_call=2, microbatch=4, max_length_query=24, max_length_answer=56,
             compute_dtype="float32", checkpoint_every=1, log_every=100, eval_every=0)
    caps = [c for c, _ in calls]
    assert caps[0] == 24, f"the question side got {caps[0]}, expected the query cap"
    assert caps[1] == 56, f"the answer side got {caps[1]}, expected the answer cap"


def test_gradient_checkpointing_is_on_by_default_and_recorded(stage):
    """Measured on the card 17 of 30 scavenger nodes carry: it takes 128 anchors from OOM to 2.24 GiB
    at three quarters of the throughput. Here the batch size is the negative count, so that trade
    goes the opposite way from the masked language modeling stage's."""
    mlm, train_pairs, _ = stage
    r = CT.train(mlm, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=1,
                 calls_per_batch=2, pairs_per_call=2, microbatch=4, max_length_query=24, max_length_answer=48,
                 compute_dtype="float32", checkpoint_every=1, log_every=100, eval_every=0)
    assert r["gradient_checkpointing"] is True
    r2 = CT.train(mlm, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=1,
                  calls_per_batch=2, pairs_per_call=2, microbatch=4, max_length_query=24, max_length_answer=48,
                  compute_dtype="float32", checkpoint_every=1, log_every=100, eval_every=0,
                  name="ecalls-ret-nockpt", gradient_checkpointing=False)
    assert r2["gradient_checkpointing"] is False


# --- gradient caching: the negative count must not depend on the card ---------------------------

def test_the_cached_gradient_equals_the_full_batch_gradient(stage):
    """The whole justification for gradient caching. If the assembled gradient were only an
    approximation, the batch would not be a real hyperparameter and every claim about holding the
    negative count constant across cards would be false.

    Method: Gao, Zhang, Han, Callan (2021), doi 10.18653/v1/2021.repl4nlp-1.31.
    """
    mlm, _, _ = stage
    run = json.loads((PT.model_dir(mlm) / "_run.json").read_text())
    tok = CT.hf_tokenizer(run["tokenizer"], 64)
    q = [f"what drove margin in segment {i}" for i in range(8)]
    a = [f"pricing and freight moved segment {i} materially this quarter" for i in range(8)]
    ids = np.array([f"a{i}" for i in range(8)])

    def naive_grads():
        model, _ = CT.load_backbone(mlm, run["vocab_size"], "cpu")
        model.train()
        model.zero_grad(set_to_none=True)
        Q = CT.embed(model, tok, q, "cpu", 32, None)
        A = CT.embed(model, tok, a, "cpu", 48, None)
        CT.info_nce(Q, A, ids, 0.05).backward()
        return {k: v.grad.clone() for k, v in model.named_parameters() if v.grad is not None}

    def cached_grads(mb):
        model, _ = CT.load_backbone(mlm, run["vocab_size"], "cpu")
        model.train()
        model.zero_grad(set_to_none=True)
        CT.cached_step(model, tok, q, a, ids, "cpu", 32, 48, None, 0.05, mb, None)
        return {k: v.grad.clone() for k, v in model.named_parameters() if v.grad is not None}

    ref = naive_grads()
    for mb in (2, 3, 8):
        got = cached_grads(mb)
        assert set(got) == set(ref), "a different set of parameters received gradient"
        worst = max((ref[k] - got[k]).abs().max().item() /
                    max(1e-12, ref[k].abs().max().item()) for k in ref)
        assert worst < 1e-4, f"microbatch {mb}: relative gradient error {worst:.2e}"


def test_the_cached_loss_does_not_depend_on_the_microbatch(stage):
    """A loss that moved with the microbatch would mean the similarity matrix was being computed on
    pieces rather than on the whole batch, which is exactly the thing being avoided."""
    mlm, _, _ = stage
    run = json.loads((PT.model_dir(mlm) / "_run.json").read_text())
    tok = CT.hf_tokenizer(run["tokenizer"], 64)
    q = [f"what drove margin in segment {i}" for i in range(9)]
    a = [f"pricing and freight moved segment {i} this quarter" for i in range(9)]
    ids = np.array([f"a{i}" for i in range(9)])
    losses = []
    for mb in (2, 4, 9):
        model, _ = CT.load_backbone(mlm, run["vocab_size"], "cpu")
        model.train(); model.zero_grad(set_to_none=True)
        losses.append(float(CT.cached_step(model, tok, q, a, ids, "cpu", 32, 48, None, 0.05, mb, None)))
    assert max(losses) - min(losses) < 1e-5, losses


def test_a_ragged_final_microbatch_is_handled(stage):
    """9 anchors in microbatches of 4 leaves a piece of 1, and seeding its backward pass with a
    mis-sliced gradient would either raise or silently apply the wrong vector."""
    mlm, _, _ = stage
    run = json.loads((PT.model_dir(mlm) / "_run.json").read_text())
    tok = CT.hf_tokenizer(run["tokenizer"], 64)
    q = [f"question {i} on margin" for i in range(9)]
    a = [f"answer {i} on pricing and volume" for i in range(9)]
    ids = np.array([f"a{i}" for i in range(9)])
    model, _ = CT.load_backbone(mlm, run["vocab_size"], "cpu")
    model.train(); model.zero_grad(set_to_none=True)
    loss = CT.cached_step(model, tok, q, a, ids, "cpu", 32, 48, None, 0.05, 4, None)
    assert float(loss) > 0
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in model.parameters())


def test_the_microbatch_scales_with_the_card_but_the_batch_does_not():
    """The batch is the hyperparameter; the microbatch is what fits. The masked language modeling
    stage holds tokens per step fixed for the same reason."""
    assert CT.auto_microbatch("cpu", 512) <= 8
    assert CT.auto_microbatch("cpu", 4) == 4          # never larger than the batch
    assert CT.CALLS_PER_BATCH * CT.PAIRS_PER_CALL == 512


# --- the ablation path ---------------------------------------------------------------------------

def test_exactly_one_backbone_source_is_required(stage):
    """Both would leave it ambiguous which weights the recorded cutoff belongs to, and neither
    leaves nothing to train."""
    mlm, train_pairs, _ = stage
    with pytest.raises(ValueError, match="exactly one"):
        CT.train(mlm, train_pairs, cutoff="2010-12-31", backbone="chrono-bert-v1-20101231",
                 device="cpu", total_steps=1)
    with pytest.raises(ValueError, match="exactly one"):
        CT.train(None, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=1)


def test_an_atemporal_backbone_is_refused_for_the_ablation():
    """A retriever inherits its backbone's cutoff, so a backbone without one cannot produce a
    vintage. bge-small is exactly this case, which is why it is the reference and not a starting
    point."""
    with pytest.raises(ValueError, match="atemporal|no known training cutoff"):
        CT.load_registered_backbone("bge-small-en-v1.5", "cpu")


def test_the_ablation_gets_a_different_name_than_the_vintage():
    """The two are the same stage on the same pairs and differ only in the backbone, so sharing a
    name would have one overwrite the other and destroy the comparison."""
    assert CT.run_name("2010-12-31", "ecalls-mlm") == "ecalls-ret-20101231"
    assert CT.run_name("2010-12-31", "registered") == "ecalls-abl-20101231"
    assert CT.run_name("2010-12-31", "ecalls-mlm") != CT.run_name("2010-12-31", "registered")


def test_embed_refuses_an_unknown_pooling_rule(stage):
    """Falling back to mean for a CLS-trained retriever discards where its sentence meaning was put."""
    mlm, _, _ = stage
    run = json.loads((PT.model_dir(mlm) / "_run.json").read_text())
    tok = CT.hf_tokenizer(run["tokenizer"], 32)
    model, _ = CT.load_backbone(mlm, run["vocab_size"], "cpu")
    with pytest.raises(ValueError, match="pooling must be one of"):
        CT.embed(model, tok, ["margin and freight"], "cpu", 32, None, "first-token-ish")


def test_an_oom_halves_the_microbatch_and_leaves_the_batch_alone(stage, monkeypatch, capsys):
    """The masked language modeling stage had to halve the microbatch *and* double accumulation to
    keep the gradient fixed. Here the microbatch does not enter the gradient at all, so recovery is
    free, and what must not change is the anchor count."""
    mlm, train_pairs, _ = stage
    calls = {"n": 0}
    real = CT.cached_step

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return real(*a, **kw)

    monkeypatch.setattr(CT, "cached_step", flaky)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    r = CT.train(mlm, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=1,
                 calls_per_batch=2, pairs_per_call=2, microbatch=4,
                 max_length_query=24, max_length_answer=48, compute_dtype="float32",
                 checkpoint_every=1, log_every=100, eval_every=0, name="ecalls-ret-oom")
    assert "microbatch -> 2" in capsys.readouterr().out
    assert r["microbatch"] == 2
    assert r["anchors_per_batch"] == 4, "the batch, and so the negative count, must not have moved"
    assert r["final_step"] == 1


# --- one trainer per model directory -------------------------------------------------------------

def test_a_second_trainer_on_the_same_model_directory_is_refused(stage):
    """Two identical ablation arrays ran concurrently over the same fifteen directories on
    2026-10-01, and nothing in the artifacts said so: every run record reported 4,000 steps because
    whichever process finished last wrote one. The per-writer temp name keeps a single file from
    being spliced; it does not stop two trainers taking turns on one model."""
    from ecalls.ingest import manifest as M

    mlm, train_pairs, _ = stage
    d = CT.model_dir("ecalls-ret-20101231")
    M.claim_dir(d, label="a pretend concurrent run")
    # a foreign claim, so the lock must not carry this process's SLURM job id
    lock = json.loads((d / "_lock.json").read_text())
    lock["slurm_job"] = "999999"
    (d / "_lock.json").write_text(json.dumps(lock))
    with pytest.raises(SystemExit, match="already being written"):
        CT.train(mlm, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=1,
                 calls_per_batch=2, pairs_per_call=2, microbatch=4,
                 max_length_query=24, max_length_answer=48, compute_dtype="float32",
                 checkpoint_every=1, log_every=100, eval_every=0)


def test_a_stale_lock_does_not_block_a_restart(stage):
    """A preempted scavenger task cannot clean up after itself, so a lock nobody is refreshing has
    to expire or the vintage can never be retrained."""
    import os as _os
    from ecalls.ingest import manifest as M

    mlm, train_pairs, _ = stage
    d = CT.model_dir("ecalls-ret-20101231")
    M.claim_dir(d, label="abandoned")
    lock = json.loads((d / "_lock.json").read_text())
    lock["slurm_job"] = "999999"
    (d / "_lock.json").write_text(json.dumps(lock))
    _os.utime(d / "_lock.json", (0, 0))          # long abandoned
    r = CT.train(mlm, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=1,
                 calls_per_batch=2, pairs_per_call=2, microbatch=4,
                 max_length_query=24, max_length_answer=48, compute_dtype="float32",
                 checkpoint_every=1, log_every=100, eval_every=0)
    assert r["final_step"] == 1


def test_a_requeue_of_the_same_job_reclaims_its_own_lock(stage, monkeypatch):
    """Requeue keeps the job id, so the resumed run must be allowed through its own fresh lock."""
    from ecalls.ingest import manifest as M

    mlm, train_pairs, _ = stage
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    d = CT.model_dir("ecalls-ret-20101231")
    M.claim_dir(d, label="first attempt")
    r = CT.train(mlm, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=1,
                 calls_per_batch=2, pairs_per_call=2, microbatch=4,
                 max_length_query=24, max_length_answer=48, compute_dtype="float32",
                 checkpoint_every=1, log_every=100, eval_every=0)
    assert r["final_step"] == 1


def test_the_checkpoint_temp_name_is_unique_per_writer():
    """A fixed `.tmp` is what let one array rename a file out from under the other."""
    from ecalls.ingest import manifest as M

    a = M.temp_sibling(pathlib.Path("/x/checkpoint.pt"))
    b = M.temp_sibling(pathlib.Path("/x/checkpoint.pt"))
    assert a != b
    assert a.name.endswith(".tmp") and "checkpoint.pt" in a.name


# --- the domain-adaptive path ----------------------------------------------------------------------

TINY = dict(hidden=32, layers=2, heads=2, intermediate=64, context=128)


@pytest.fixture
def dapt(stage, monkeypatch):
    """A finished continued-pretraining run on a tiny stand-in for ChronoBERT's architecture.

    `load_mlm_backbone` is replaced by a builder of the same tiny architecture at seed 0, and the
    run's checkpoint holds the weights of a seed-1 build. A loader that reads the checkpoint returns
    seed-1 weights, and one that silently keeps the freshly built backbone returns seed-0 weights,
    which is the failure that would make this path score like the ablation.
    """
    from types import SimpleNamespace

    from ecalls.chunk import registry as R
    from ecalls.train import model as MD

    _, train_pairs, ho_pairs = stage
    tok = CT.hf_tokenizer("ct-tok", 64)
    vocab = len(tok)
    monkeypatch.setattr(PT, "load_mlm_backbone",
                        lambda model_id, device: (MD.build_mlm(vocab, seed=0, **TINY), None, {}))
    entry = SimpleNamespace(source="huggingface:fake/chrono", pooling="mean")
    monkeypatch.setattr(R, "load", lambda *a, **k: SimpleNamespace(by_name=lambda name: entry))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)

    name = "ecalls-dapt-20101231"
    d = PT.model_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    continued = MD.build_mlm(vocab, seed=1, **TINY)
    torch.save({"model": continued.state_dict()}, d / "checkpoint.pt")
    run = {"name": name, "stage": "dapt", "backbone": "chrono-bert-v1-20101231",
           "cutoff": "2010-12-31", "tokenizer": "registry:chrono-bert-v1-20241231",
           "total_steps": 10, "final_step": 10}
    (d / "_run.json").write_text(json.dumps(run) + "\n")
    return name, train_pairs, ho_pairs, continued


def test_the_dapt_backbone_carries_the_continued_weights_not_the_fresh_ones(dapt):
    name, _, _, continued = dapt
    body, tok, run, pooling = CT.load_dapt_backbone(name, "cpu")
    want = continued.model.state_dict()
    got = body.state_dict()
    assert set(got) == set(want)
    for k in want:
        assert torch.equal(got[k], want[k]), k
    assert pooling == "mean"


@pytest.mark.parametrize("final_step, total_steps", [(4250, 12207), (None, 12207)])
def test_an_unfinished_dapt_run_is_refused(dapt, final_step, total_steps):
    """A preempted continued run leaves a valid checkpoint part way through its budget. A retriever
    built on it would record a backbone that saw a third of the corpus as if it saw all of it."""
    name = dapt[0]
    f = PT.model_dir(name) / "_run.json"
    run = json.loads(f.read_text())
    run.update(final_step=final_step, total_steps=total_steps)
    f.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="stopped at step"):
        CT.load_dapt_backbone(name, "cpu")


def test_a_dapt_run_without_its_record_is_refused(dapt):
    """The trainer writes `_run.json` only after its last step, so its absence means unfinished."""
    name = dapt[0]
    (PT.model_dir(name) / "_run.json").unlink()
    with pytest.raises(FileNotFoundError, match="has not finished"):
        CT.load_dapt_backbone(name, "cpu")


def test_the_contrastive_stage_trains_from_a_dapt_backbone_under_its_own_name(dapt):
    name, train_pairs, ho_pairs, _ = dapt
    r = CT.train(name, train_pairs, cutoff="2010-12-31", device="cpu", total_steps=1,
                 calls_per_batch=2, pairs_per_call=2, microbatch=4, max_length_query=32,
                 max_length_answer=48, compute_dtype="float32", checkpoint_every=1, log_every=100,
                 eval_every=0)
    assert r["name"] == "ecalls-dapt-ret-20101231"
    assert r["backbone_kind"] == "dapt"
    assert r["backbone"] == name
    assert r["tokenizer"] == "registry:chrono-bert-v1-20241231"
    assert (CT.model_dir(r["name"]) / "hf" / "config.json").exists()


def test_the_three_starting_points_and_any_seed_get_distinct_names():
    names = {CT.run_name("2010-12-31", k, s)
             for k in ("ecalls-mlm", "registered", "dapt") for s in (0, 1, 2)}
    assert len(names) == 9
    assert CT.run_name("2010-12-31", "dapt") == "ecalls-dapt-ret-20101231"
    assert CT.run_name("2010-12-31", "ecalls-mlm", 1) == "ecalls-seed1-ret-20101231"
    with pytest.raises(ValueError, match="unknown backbone kind"):
        CT.run_name("2010-12-31", "mystery")


def test_no_seed_or_dapt_name_falls_inside_a_registered_family_prefix():
    """The registry resolves a family by `model_id` prefix and ties at one cutoff raise. A seed run
    whose name started with `ecalls-ret` would tie with the deliverable and break `for_date` for
    every caller."""
    for kind in ("ecalls-mlm", "registered", "dapt"):
        for s in (1, 2):
            n = CT.run_name("2010-12-31", kind, s)
            assert not n.startswith(("ecalls-ret", "ecalls-abl", "ecalls-dapt-ret")), n
    assert not CT.run_name("2010-12-31", "dapt").startswith(("ecalls-ret", "ecalls-abl"))
    # and the date stays the last field, which is how `families()` splits a name
    assert CT.run_name("2010-12-31", "ecalls-mlm", 2).rsplit("-", 1)[1] == "20101231"


def test_the_record_gives_the_batch_actually_drawn_across_a_resume(stage):
    """Each fixture call has 6 pairs, so asking for 10 per call draws 6 from each: 2 calls give 12
    anchors per batch against a maximum of 20. The count survives a resume, so a preempted run reports
    the same mean as one that ran straight through."""
    mlm, train_pairs, _ = stage
    kw = dict(cutoff="2010-12-31", device="cpu", calls_per_batch=2, pairs_per_call=10, microbatch=4,
              max_length_query=32, max_length_answer=48, compute_dtype="float32", checkpoint_every=1,
              log_every=100, eval_every=0)
    CT.train(mlm, train_pairs, total_steps=1, **kw)
    r = CT.train(mlm, train_pairs, total_steps=3, **kw)
    assert r["final_step"] == 3
    assert r["anchors_per_batch"] == 20
    assert r["mean_anchors_per_batch"] == 12.0


# --- the reference path for a backbone with no known cutoff (D34) ---------------------------------

def test_a_backbone_with_no_cutoff_is_refused_unless_it_is_a_reference():
    with pytest.raises(ValueError, match="no known training cutoff"):
        CT.check_reference("atemporal", reference=False, name="ref-bge-small-ft-20241231")


def test_a_reference_must_carry_the_ref_prefix_so_it_never_registers_as_a_vintage():
    with pytest.raises(ValueError, match="ref-"):
        CT.check_reference("atemporal", reference=True, name="ecalls-abl-20241231")
    with pytest.raises(ValueError, match="ref-"):
        CT.check_reference("atemporal", reference=True, name=None)
    CT.check_reference("atemporal", reference=True, name="ref-bge-small-ft-20241231")


def test_a_vintage_backbone_passes_the_guard_as_before():
    CT.check_reference("vintage", reference=False, name=None)


def test_no_vintage_family_pattern_matches_a_reference_name():
    import fnmatch, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import register_vintages as RV
    vintage_patterns = [p for p in RV.PATTERNS if p != "ref-*"]
    assert not any(fnmatch.fnmatch("ref-bge-small-ft-20241231", p) for p in vintage_patterns)
