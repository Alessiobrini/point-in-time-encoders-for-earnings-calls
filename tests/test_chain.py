"""The chained vintages (D29) and the 2x budget check.

A chained segment starts from the model of the previous year end and trains once over that year's new
calls. Each property tested here fails quietly when broken: an increment that overlaps or skips text, a
parent from the wrong year, a held-out call reaching training, a schedule that does not restart, and a
name that collides with an existing family in the registry or on disk.
"""
import fnmatch
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from ecalls.chunk import registry as R
from ecalls.train import contrastive as CT
from ecalls.train import holdout as HO
from ecalls.train import pretrain as PT
from ecalls.train import pretrain_data as PD
from ecalls.train import tokenizer as T

_p = Path(__file__).resolve().parents[1] / "scripts" / "register_vintages.py"
_spec = importlib.util.spec_from_file_location("register_vintages", _p)
RV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RV)

YEARS = list(range(2010, 2025))


# --- the increment, on shard offsets alone -------------------------------------------------------

def _meta(seed: int = 7, empty_year: int | None = None) -> dict:
    """A packed corpus's sidecar with four quarterly shards per year, 2010 to 2024, uneven sizes."""
    rng = np.random.default_rng(seed)
    offs, end = [], 0
    for y in YEARS:
        for q in (1, 2, 3, 4):
            end += 0 if y == empty_year else int(rng.integers(5, 50))
            offs.append({"shard": f"{y}Q{q}", "end": end})
    return {"name": "synthetic", "cutoff": "2024-12-31", "context": 64, "n_sequences": end,
            "shard_offsets": offs}


def test_each_increment_is_the_difference_of_two_from_scratch_prefixes():
    """The increment must come from the same function that sized every from-scratch vintage, so a
    segment can never see a sequence the from-scratch vintage of its year could not."""
    m = _meta()
    for t in YEARS[1:]:
        first, end = PD.increment_bounds(m, f"{t - 1}-12-31", f"{t}-12-31")
        assert first == PD.sequences_up_to(m, f"{t - 1}-12-31")
        assert end == PD.sequences_up_to(m, f"{t}-12-31")


def test_consecutive_increments_meet_with_no_overlap_and_no_gap():
    m = _meta()
    bounds = [PD.increment_bounds(m, f"{t - 1}-12-31", f"{t}-12-31") for t in YEARS[1:]]
    for (_, end_a), (first_b, _) in zip(bounds, bounds[1:]):
        assert end_a == first_b


def test_the_2010_window_plus_every_increment_is_the_2024_window():
    m = _meta()
    seen = list(range(PD.sequences_up_to(m, "2010-12-31")))
    for t in YEARS[1:]:
        first, end = PD.increment_bounds(m, f"{t - 1}-12-31", f"{t}-12-31")
        seen.extend(range(first, end))
    assert seen == list(range(PD.sequences_up_to(m, "2024-12-31")))
    assert len(seen) == len(set(seen)) == m["n_sequences"]


def test_an_empty_increment_is_refused():
    """A segment with nothing to train on would still write a record and register as a vintage
    identical to its parent."""
    with pytest.raises(ValueError, match="no sequence after"):
        PD.increment_bounds(_meta(empty_year=2013), "2012-12-31", "2013-12-31")


def test_an_increment_past_the_packed_corpus_is_refused():
    with pytest.raises(ValueError, match="was packed to"):
        PD.increment_bounds(_meta(), "2024-12-31", "2025-12-31")


def test_an_increment_that_ends_before_it_starts_is_refused():
    with pytest.raises(ValueError, match="must start before"):
        PD.increment_bounds(_meta(), "2012-12-31", "2012-12-31")


# --- a real packed corpus over three years -------------------------------------------------------

def _ids_for(year_idx: int) -> tuple[list[int], list[int]]:
    """Two held-out and fifteen trainable calls per year, from disjoint id ranges."""
    r = range(1 + 200 * year_idx, 200 * (year_idx + 1))
    held = [k for k in r if HO.is_holdout(k)][:2]
    trained = [k for k in r if not HO.is_holdout(k)][:15]
    assert len(held) == 2 and len(trained) == 15
    return held, trained


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A packed corpus with 2010, 2011 and 2012 shards, packed with the default held-out split."""
    from ecalls.embed import encode as E
    from ecalls.train import pairs as PR

    for mod in (PD, T, PR):
        monkeypatch.setattr(mod.P, "CHUNKS_DIR", tmp_path / "chunks")
        monkeypatch.setattr(mod.P, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(mod.P, "chunks_shard", lambda s: tmp_path / "chunks" / f"chunks_{s}.parquet")
    monkeypatch.setattr(PD, "PRETRAIN_DIR", tmp_path / "processed" / "pretrain")
    monkeypatch.setattr(PR, "PAIRS_DIR", tmp_path / "processed" / "pairs")
    monkeypatch.setattr(T, "TOKENIZER_DIR", tmp_path / "processed" / "tokenizers")
    monkeypatch.setattr(PT, "MODELS_DIR", tmp_path / "processed" / "models")
    monkeypatch.setattr(CT, "MODELS_DIR", tmp_path / "processed" / "models")
    monkeypatch.setattr(E.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    PD.P.CHUNKS_DIR.mkdir(parents=True)

    calls = {}
    for i, shard in enumerate(("2010Q1", "2011Q2", "2012Q3")):
        held, trained = _ids_for(i)
        calls[shard] = {"held": held, "trained": trained}
        ids = sorted(held + trained)
        pl.DataFrame({"keydevid": ids, "chunk_id": [f"{k}-00" for k in ids],
                      "text": [" ".join([f"w{k}x"] * 20) for k in ids],
                      "n_chars": [100] * len(ids)}).write_parquet(PD.P.chunks_shard(shard))
    T.fit("2010-12-31", vocab_size=800, name="chain-tok")
    w = PD.pack_window("2012-12-31", "chain-tok", context=64, quiet=True)
    return SimpleNamespace(window=w["name"], tok="chain-tok", meta=w, calls=calls)


def _decoded(tok, arr) -> str:
    # Spaces stripped, because digit splitting renders "w27x" as "w 2 7 x" on decode. See
    # test_pretrain_data for the two earlier versions of this check that this artifact broke.
    return " ".join(tok.decode([int(x) for x in row]) for row in np.asarray(arr)).replace(" ", "")


def test_the_loaded_increment_is_the_slice_between_the_two_prefixes(corpus):
    full, _ = PD.load_window(corpus.window)
    arr, m = PD.load_increment(corpus.window, "2010-12-31", "2011-12-31")
    first = PD.sequences_up_to(corpus.meta, "2010-12-31")
    end = PD.sequences_up_to(corpus.meta, "2011-12-31")
    assert (m["first_sequence"], m["last_sequence"]) == (first, end - 1)
    assert m["n_sequences"] == len(arr) == end - first
    assert m["n_tokens"] == len(arr) * 64
    assert np.array_equal(np.asarray(arr), np.asarray(full[first:end]))


def test_an_increment_holds_its_own_year_only_and_no_held_out_call(corpus):
    """Held out of both stages, in every segment. The benchmark draws its queries from those calls."""
    tok = T.load(corpus.tok)
    arr, _ = PD.load_increment(corpus.window, "2010-12-31", "2011-12-31")
    flat = _decoded(tok, arr)
    year = corpus.calls["2011Q2"]
    assert not [k for k in year["held"] if f"w{k}x" in flat], "a held-out call reached the increment"
    assert [k for k in year["trained"] if f"w{k}x" in flat], "no trainable call either, so this proves nothing"
    others = corpus.calls["2010Q1"]["trained"] + corpus.calls["2012Q3"]["trained"]
    assert not [k for k in others if f"w{k}x" in flat], "text from another year reached the increment"


# --- the segment, end to end ---------------------------------------------------------------------

def _base(corpus) -> dict:
    return PT.train(corpus.window, corpus.tok, cutoff="2010-12-31", max_steps=1, batch_sequences=2,
                    accum=1, device="cpu", checkpoint_every=100, log_every=100)


def _segment(corpus, year: int, parent: str, **kw) -> dict:
    return PT.train(corpus.window, corpus.tok, cutoff=f"{year}-12-31", init_from=parent,
                    batch_sequences=2, accum=1, device="cpu", checkpoint_every=100,
                    log_every=100, **kw)


def test_a_segment_trains_once_over_its_increment_and_records_its_lineage(corpus, monkeypatch):
    base = _base(corpus)
    loaded = {}
    real = PT.load_parent_weights

    def spy(model, parent):
        real(model, parent)
        want = torch.load(PT.model_dir(parent) / "checkpoint.pt", weights_only=False)["model"]
        loaded["equal"] = all(torch.equal(v, want[k]) for k, v in model.state_dict().items())
        loaded["parent"] = parent

    monkeypatch.setattr(PT, "load_parent_weights", spy)
    r = _segment(corpus, 2011, base["name"])

    first = PD.sequences_up_to(corpus.meta, "2010-12-31")
    end = PD.sequences_up_to(corpus.meta, "2011-12-31")
    run = json.loads((PT.model_dir(r["name"]) / "_run.json").read_text())
    assert run["name"] == "ecalls-chain-mlm-20111231"
    assert run["stage"] == "chain-mlm" and run["family"] == "chain-mlm"
    assert run["init_from"] == "ecalls-mlm-20101231"
    assert run["init_from_cutoff"] == run["start_cutoff"] == "2010-12-31"
    assert run["cutoff"] == "2011-12-31"
    assert (run["first_sequence"], run["last_sequence"]) == (first, end - 1)
    assert run["n_sequences_available"] == end - first
    # One pass: the budget is the increment, and whole steps never repeat a sequence.
    assert run["token_budget"] == (end - first) * 64
    assert run["total_steps"] == run["token_budget"] // run["tokens_per_step"]
    assert run["final_step"] == run["total_steps"]
    assert run["tokens_seen"] <= run["token_budget"]
    assert 0.9 < run["epochs_equivalent"] <= 1.0
    # Fresh optimizer and counters: the segment's tokens are its own, not the parent's plus its own.
    assert run["tokens_seen"] == run["final_step"] * run["tokens_per_step"]
    assert run["warmup_steps"] == int(run["total_steps"] * 0.06)
    assert loaded == {"equal": True, "parent": "ecalls-mlm-20101231"}
    window_file = json.loads((PT.model_dir(r["name"]) / "_window.json").read_text())
    assert window_file["init_from"] == "ecalls-mlm-20101231"
    assert window_file["first_sequence"] == first


def test_each_segment_restarts_its_own_schedule(corpus, monkeypatch):
    base = _base(corpus)
    calls = []
    real = PT.lr_at
    monkeypatch.setattr(PT, "lr_at", lambda s, t, p, w: calls.append((s, t, w)) or real(s, t, p, w))
    r = _segment(corpus, 2011, base["name"])
    assert [c[0] for c in calls] == list(range(r["total_steps"]))
    assert {c[1] for c in calls} == {r["total_steps"]}
    assert {c[2] for c in calls} == {r["warmup_steps"]}


def test_the_schedule_warms_up_then_decays_to_the_floor_at_the_last_step():
    """At the size of the largest real increment, 274,035 sequences, which is 2,140 steps."""
    total = 274_035 * 768 // PT.TOKENS_PER_STEP
    warmup = int(total * 0.06)
    peak = 6e-4
    lrs = [PT.lr_at(s, total, peak, warmup) for s in range(total)]
    assert lrs[0] < lrs[warmup // 2] < lrs[warmup - 1] == pytest.approx(peak)
    assert all(a >= b for a, b in zip(lrs[warmup:], lrs[warmup + 1:])), "not monotone after warmup"
    assert lrs[-1] == pytest.approx(0.1 * peak, rel=1e-4)
    assert PT.lr_at(total, total, peak, warmup) == pytest.approx(0.1 * peak)


def test_a_second_segment_continues_from_the_first_and_meets_it(corpus):
    base = _base(corpus)
    s11 = _segment(corpus, 2011, base["name"])
    s12 = _segment(corpus, 2012, s11["name"])
    assert s12["name"] == "ecalls-chain-mlm-20121231"
    assert s12["init_from"] == s11["name"]
    assert s12["first_sequence"] == s11["last_sequence"] + 1
    assert s12["last_sequence"] + 1 == PD.sequences_up_to(corpus.meta, "2012-12-31")


# --- the parent ----------------------------------------------------------------------------------

def _fake_parent(name: str, **over) -> None:
    d = PT.model_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    run = {"name": name, "stage": "mlm", "cutoff": "2010-12-31", "total_steps": 10, "final_step": 10,
           **over}
    (d / "_run.json").write_text(json.dumps(run) + "\n")


@pytest.mark.parametrize("parent_cutoff, match", [
    ("2009-12-31", "must continue from the model of 2010-12-31"),
    ("2010-06-30", "must continue from the model of 2010-12-31"),
    ("2011-12-31", "at or after"),
    ("2012-12-31", "at or after"),
])
def test_a_parent_from_any_date_but_the_previous_year_end_is_refused(corpus, parent_cutoff, match):
    _fake_parent("p", cutoff=parent_cutoff)
    with pytest.raises(ValueError, match=match):
        PT.chain_parent("p", "2011-12-31")
    with pytest.raises(ValueError, match=match):
        _segment(corpus, 2011, "p")
    assert not PT.model_dir("ecalls-chain-mlm-20111231").exists(), "refused after claiming a directory"


@pytest.mark.parametrize("final_step", [4, None])
def test_an_unfinished_parent_is_refused(corpus, final_step):
    _fake_parent("p", final_step=final_step)
    with pytest.raises(ValueError, match="stopped at step"):
        _segment(corpus, 2011, "p")


def test_a_parent_without_a_run_record_is_refused(corpus):
    """The trainer writes `_run.json` only after its last step, so a checkpoint alone is unfinished."""
    d = PT.model_dir("p")
    d.mkdir(parents=True)
    (d / "checkpoint.pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="has not finished"):
        _segment(corpus, 2011, "p")


def test_a_domain_adapted_parent_is_refused(corpus):
    _fake_parent("p", stage="dapt")
    with pytest.raises(ValueError, match="stage 'dapt'"):
        _segment(corpus, 2011, "p")


def test_a_segment_cannot_skip_its_predecessor(corpus):
    """2012 from the 2010 base would leave the 2011 calls unseen while the record claimed a chain."""
    base = _base(corpus)
    with pytest.raises(ValueError, match="must continue from the model of 2011-12-31"):
        _segment(corpus, 2012, base["name"])


@pytest.mark.parametrize("field, value, match", [
    ("holdout_salt", "ecalls-holdout-v0", "holdout_salt"),
    ("holdout_fraction", 0.05, "holdout_fraction"),
    ("tokenizer", "another-tok", "tokenizer"),
    ("window", "another-window", "Sequence indices mean different text"),
    ("n_sequences_available", 3, "skip or repeat"),
])
def test_a_parent_trained_on_a_different_split_or_corpus_is_refused(corpus, field, value, match):
    base = _base(corpus)
    f = PT.model_dir(base["name"]) / "_run.json"
    run = json.loads(f.read_text())
    run[field] = value
    f.write_text(json.dumps(run))
    with pytest.raises(ValueError, match=match):
        _segment(corpus, 2011, base["name"])


def test_a_segment_refuses_a_token_budget(corpus):
    base = _base(corpus)
    with pytest.raises(ValueError, match="one pass over its increment"):
        _segment(corpus, 2011, base["name"], token_budget=PT.DEFAULT_TOKEN_BUDGET)


def test_training_on_a_held_out_pack_is_refused(corpus):
    ho = PD.pack_window("2012-12-31", corpus.tok, context=64, quiet=True, holdout_only=True)
    with pytest.raises(ValueError, match="held-out pack"):
        PT.train(ho["name"], corpus.tok, cutoff="2010-12-31", max_steps=1, batch_sequences=2,
                 accum=1, device="cpu")


# --- names ---------------------------------------------------------------------------------------

def test_the_family_tag_names_the_chain_and_the_budget_multiple():
    assert PT.family_tag() == "mlm"
    assert PT.family_tag(PT.DEFAULT_TOKEN_BUDGET) == "mlm"
    assert PT.family_tag(2 * PT.DEFAULT_TOKEN_BUDGET) == "mlm2x"
    assert PT.family_tag(1_500_000_000) is None
    assert PT.family_tag(backbone="chrono-bert-v1-20101231") == "dapt"
    assert PT.family_tag(init_from="ecalls-mlm-20101231") == "chain-mlm"
    assert PT.run_name("2011-12-31", "chain-mlm") == "ecalls-chain-mlm-20111231"
    assert PT.run_name("2024-12-31", PT.family_tag(2_400_000_000)) == "ecalls-mlm2x-20241231"
    assert PT.previous_year_end("2011-12-31") == "2010-12-31"


def test_the_2x_run_gets_its_own_directory_by_default(corpus):
    r = PT.train(corpus.window, corpus.tok, cutoff="2010-12-31", max_steps=1, batch_sequences=2,
                 accum=1, device="cpu", token_budget=2 * PT.DEFAULT_TOKEN_BUDGET)
    assert r["name"] == "ecalls-mlm2x-20101231"
    assert r["family"] == "mlm2x" and r["stage"] == "mlm"


def test_a_budget_with_no_family_needs_a_name(corpus):
    with pytest.raises(ValueError, match="Pass --name"):
        PT.train(corpus.window, corpus.tok, cutoff="2010-12-31", max_steps=1, batch_sequences=2,
                 accum=1, device="cpu", token_budget=1_500_000_000)


def test_a_2x_run_cannot_resume_into_the_1x_vintage(corpus):
    """Named by hand onto the 1x directory, it would extend the finished deliverable to 2x steps."""
    base = _base(corpus)
    with pytest.raises(ValueError, match="continue a different experiment"):
        PT.train(corpus.window, corpus.tok, cutoff="2010-12-31", max_steps=2, batch_sequences=2,
                 accum=1, device="cpu", token_budget=2 * PT.DEFAULT_TOKEN_BUDGET, name=base["name"])


def test_the_contrastive_names_for_the_new_families():
    assert CT.run_name("2011-12-31", "chain-mlm") == "ecalls-chain-ret-20111231"
    assert CT.run_name("2024-12-31", "mlm2x") == "ecalls-2x-ret-20241231"
    assert CT.run_name("2011-12-31", "chain-mlm", seed=1) == "ecalls-seed1-chain-ret-20111231"
    assert len(set(CT.RUN_TAGS.values())) == len(CT.RUN_TAGS)
    assert set(CT.FAMILY_KINDS.values()) <= set(CT.RUN_TAGS)


def test_no_family_prefix_captures_another_family():
    """The registry resolves a family by `model_id` prefix, so `ecalls-ret2x-*` would tie with
    `ecalls-ret-*` at 2024 and make every `family="ecalls-ret"` query from 2025 on raise."""
    prefixes = [f"ecalls-{tag}" for tag in CT.RUN_TAGS.values()]
    for a in prefixes:
        for b in prefixes:
            if a != b:
                assert not f"{b}-20241231".startswith(a), f"{a} captures {b}"
    reg = R.Registry([R.Entry(m, R.VINTAGE, f"local:{m}", 384, "2024-12-31")
                      for m in ("ecalls-ret-20241231", "ecalls-2x-ret-20241231",
                                "ecalls-chain-ret-20241231", "ecalls-rpl-ret-20241231")])
    assert reg.for_date("2025-06-30", family="ecalls-rpl-ret").model_id == "ecalls-rpl-ret-20241231"
    assert reg.for_date("2025-06-30", family="ecalls-ret").model_id == "ecalls-ret-20241231"
    assert reg.for_date("2025-06-30", family="ecalls-2x-ret").model_id == "ecalls-2x-ret-20241231"
    assert reg.for_date("2025-06-30", family="ecalls-chain-ret").model_id == "ecalls-chain-ret-20241231"


@pytest.mark.parametrize("record, kind", [
    ({"stage": "chain-mlm", "family": "chain-mlm"}, "chain-mlm"),
    ({"stage": "mlm", "family": "mlm2x"}, "mlm2x"),
    ({"stage": "mlm", "family": "mlm"}, "ecalls-mlm"),
    ({"stage": "mlm"}, "ecalls-mlm"),           # a record written before `family` existed
    ({}, "ecalls-mlm"),                         # the fifteen vintages, written before `stage`
    ({"stage": "dapt"}, "dapt"),
])
def test_the_backbone_kind_is_read_from_the_record(tmp_path, monkeypatch, record, kind):
    monkeypatch.setattr(PT, "MODELS_DIR", tmp_path)
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "_run.json").write_text(json.dumps(record))
    assert CT._backbone_kind("m") == kind


def test_an_unfinished_chain_backbone_is_refused_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(PT, "MODELS_DIR", tmp_path)
    (tmp_path / "ecalls-chain-mlm-20111231").mkdir()
    with pytest.raises(FileNotFoundError, match="has not finished"):
        CT._backbone_kind("ecalls-chain-mlm-20111231")


# --- the contrastive stage on a chained backbone -------------------------------------------------

def _pairs(name: str, calls, holdout: bool) -> None:
    from ecalls.train import pairs as PR

    d = PR.pairs_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"keydevid": c, "query_id": f"q{c}-{i}", "answer_id": f"a{c}-{i}",
             "text": f"what drove the margin in segment {i} of call {c}",
             "answer_text": f"pricing and freight moved segment {i} in call {c}",
             "shard": "2010Q1" if c < 200 else "2011Q2"}
            for c in calls for i in range(4)]
    pl.DataFrame(rows).write_parquet(d / "pairs.parquet")
    (d / "_meta.json").write_text(json.dumps(
        {"name": name, "cutoff": "2011-12-31", "holdout_only": holdout, "n_pairs": len(rows),
         "holdout_fraction": 0.1, "holdout_salt": "ecalls-holdout-v1"}) + "\n")


CT_SMALL = dict(device="cpu", total_steps=1, calls_per_batch=2, pairs_per_call=2, microbatch=4,
                max_length_query=32, max_length_answer=48, compute_dtype="float32",
                checkpoint_every=1, log_every=100, eval_every=0)


def test_the_contrastive_stage_names_and_records_a_chained_retriever(corpus):
    base = _base(corpus)
    seg = _segment(corpus, 2011, base["name"])
    _pairs("qa-pairs-20111231", corpus.calls["2010Q1"]["trained"] + corpus.calls["2011Q2"]["trained"],
           holdout=False)
    r = CT.train(seg["name"], "qa-pairs-20111231", cutoff="2011-12-31", **CT_SMALL)
    assert r["name"] == "ecalls-chain-ret-20111231"
    assert r["backbone_kind"] == "chain-mlm"
    assert r["backbone_lineage"]["init_from"] == "ecalls-mlm-20101231"
    assert r["backbone_lineage"]["first_sequence"] == seg["first_sequence"]
    assert (CT.model_dir(r["name"]) / "hf" / "config.json").exists()


def test_the_contrastive_stage_refuses_an_unfinished_chained_backbone(corpus):
    base = _base(corpus)
    seg = _segment(corpus, 2011, base["name"])
    f = PT.model_dir(seg["name"]) / "_run.json"
    run = json.loads(f.read_text())
    run["final_step"] = run["total_steps"] - 1
    f.write_text(json.dumps(run))
    _pairs("qa-pairs-20111231", corpus.calls["2011Q2"]["trained"], holdout=False)
    with pytest.raises(ValueError, match="stopped at step"):
        CT.train(seg["name"], "qa-pairs-20111231", cutoff="2011-12-31", **CT_SMALL)


# --- registration --------------------------------------------------------------------------------

def test_the_registration_patterns_pick_up_the_new_families():
    for name in ("ecalls-chain-ret-20111231", "ecalls-chain-ret-20241231", "ecalls-2x-ret-20241231",
                 "ecalls-rpl-ret-20101231"):
        assert any(fnmatch.fnmatch(name, p) for p in RV.PATTERNS), name
    # and the pretraining directories, which are not retrievers, stay out
    for name in ("ecalls-chain-mlm-20111231", "ecalls-mlm2x-20241231", "ecalls-rpl-mlm-20101231"):
        assert not any(fnmatch.fnmatch(name, p) for p in RV.PATTERNS), name


def test_every_backbone_kind_has_a_registry_note():
    assert set(CT.RUN_TAGS) <= set(RV.NOTES)


def test_a_chained_retriever_registers_as_a_vintage_with_its_lineage(tmp_path):
    d = tmp_path / "ecalls-chain-ret-20111231"
    (d / "hf").mkdir(parents=True)
    (d / "hf" / "config.json").write_text(json.dumps({"hidden_size": 384}))
    lineage = {"family": "chain-mlm", "init_from": "ecalls-mlm-20101231",
               "init_from_cutoff": "2010-12-31", "start_cutoff": "2010-12-31",
               "first_sequence": 394072, "last_sequence": 567398, "token_budget": 133115136,
               "tokens_seen": 133038080}
    run = {"name": d.name, "stage": "contrastive", "cutoff": "2011-12-31", "pooling": "mean",
           "normalized": True, "backbone_kind": "chain-mlm", "backbone": "ecalls-chain-mlm-20111231",
           "backbone_lineage": lineage, "n_pairs": 1000, "n_calls": 50, "temperature": 0.05,
           "max_length": 512, "tokenizer": "ecalls-bpe-32768-20101231", "vocab_size": 32768}
    (d / "_run.json").write_text(json.dumps(run))
    e = RV.entry_for(d)
    assert e.kind == R.VINTAGE and e.training_cutoff == "2011-12-31"
    assert e.notes.startswith("CHAINED")
    assert e.extra["backbone_init_from"] == "ecalls-mlm-20101231"
    assert (e.extra["backbone_first_sequence"], e.extra["backbone_last_sequence"]) == (394072, 567398)
    assert e.extra["backbone_kind"] == "chain-mlm"


# --- the replay control (D31) ---------------------------------------------------------------------

def _replay(corpus, parent: str, target: str, cutoff: str = "2011-12-31", **kw) -> dict:
    return PT.train(corpus.window, corpus.tok, cutoff=cutoff, init_from=parent, replay_of=target,
                    batch_sequences=2, accum=1, device="cpu", checkpoint_every=100, log_every=100, **kw)


@pytest.fixture
def chain3(corpus):
    base = _base(corpus)
    s11 = _segment(corpus, 2011, base["name"])
    s12 = _segment(corpus, 2012, s11["name"])
    return SimpleNamespace(base=base, s11=s11, s12=s12)


def test_a_replay_matches_the_next_segment_in_compute_and_rereads_its_parents_last_year(corpus, chain3):
    r = _replay(corpus, chain3.s11["name"], chain3.s12["name"])
    run = json.loads((PT.model_dir(r["name"]) / "_run.json").read_text())
    assert run["name"] == "ecalls-rpl-mlm-20111231"
    assert run["stage"] == run["family"] == "rpl-mlm"
    assert run["cutoff"] == "2011-12-31", "a replay reads no later text, so it keeps the parent's cutoff"
    assert run["init_from"] == chain3.s11["name"] and run["replay_of"] == chain3.s12["name"]
    # Compute matched to the segment it controls for, step for step.
    assert run["token_budget"] == chain3.s12["token_budget"]
    assert run["total_steps"] == chain3.s12["total_steps"] and run["final_step"] == run["total_steps"]
    # The slice is the parent's own last year, ending exactly where the parent ends.
    assert (run["first_sequence"], run["last_sequence"]) == (chain3.s11["first_sequence"],
                                                             chain3.s11["last_sequence"])
    assert np.isclose(run["epochs_equivalent"],
                      run["tokens_seen"] / (run["n_sequences_available"] * 64))


def test_a_replay_trains_on_no_text_after_its_parent(corpus, chain3, monkeypatch):
    seen = {}
    real = PD.load_increment

    def spy(name, start, end):
        arr, m = real(name, start, end)
        seen["text"] = _decoded(T.load(corpus.tok), arr)
        return arr, m

    monkeypatch.setattr(PD, "load_increment", spy)
    _replay(corpus, chain3.s11["name"], chain3.s12["name"])
    later = corpus.calls["2012Q3"]["trained"]
    assert not [k for k in later if f"w{k}x" in seen["text"]], "a replay read text after its cutoff"
    assert [k for k in corpus.calls["2011Q2"]["trained"] if f"w{k}x" in seen["text"]]


def test_a_replay_of_a_segment_from_another_parent_is_refused(corpus, chain3):
    with pytest.raises(ValueError, match="same weights"):
        _replay(corpus, chain3.base["name"], chain3.s12["name"], cutoff="2010-12-31")


def test_a_replay_must_match_a_chained_segment(corpus, chain3):
    with pytest.raises(ValueError, match="matches a chained segment"):
        _replay(corpus, chain3.s11["name"], chain3.base["name"])


def test_a_replay_keeps_its_parents_cutoff(corpus, chain3):
    with pytest.raises(ValueError, match="keeps its parent's cutoff"):
        _replay(corpus, chain3.s11["name"], chain3.s12["name"], cutoff="2012-12-31")
    assert not PT.model_dir("ecalls-rpl-mlm-20121231").exists()


def test_a_replay_takes_no_budget_of_its_own(corpus, chain3):
    with pytest.raises(ValueError, match="takes its budget"):
        _replay(corpus, chain3.s11["name"], chain3.s12["name"], token_budget=10_000)


def test_a_replay_slice_that_ends_after_its_parent_is_refused(corpus, chain3):
    parent = json.loads((PT.model_dir(chain3.s11["name"]) / "_run.json").read_text())
    wmeta = {"first_sequence": parent["first_sequence"], "last_sequence": parent["last_sequence"] + 3,
             "vocab_size": parent["vocab_size"], "context": parent["context"],
             "holdout_salt": parent["holdout_salt"], "holdout_fraction": parent["holdout_fraction"]}
    with pytest.raises(ValueError, match="only text its parent had read"):
        PT.check_parent_matches(parent, corpus.window, wmeta, corpus.tok, replay=True)


def test_a_replay_cannot_resume_into_the_segment_it_controls_for(corpus, chain3):
    with pytest.raises(ValueError, match="continue a different experiment"):
        _replay(corpus, chain3.s11["name"], chain3.s12["name"], name=chain3.s12["name"])


def test_the_replay_names():
    assert PT.family_tag(init_from="ecalls-chain-mlm-20111231", replay=True) == "rpl-mlm"
    assert PT.run_name("2010-12-31", "rpl-mlm") == "ecalls-rpl-mlm-20101231"
    assert CT.run_name("2010-12-31", "rpl-mlm") == "ecalls-rpl-ret-20101231"
