"""Verification for the domain-adaptive pretraining array, run on the cluster with RUN_DIR set.

What a valid result means here, as distinct from a finished job. The array exists to separate the
finance domain from the 4.4x capacity and roughly 380x pretraining-volume advantage the ablation's
backbone had (L15), so a run that stopped early, trained on the wrong vocabulary, or inherited the
wrong cutoff answers a different question than the one asked and must fail rather than be reported.

`RUN_DIR` holds one `task_<i>_run.json` per finished task, copied there by the command. The model
itself stays in `data/processed/models/`, because the registry resolves a `local:` source by the
registered name and a timestamped directory would break that contract. The copy is what makes this
test possible without reaching into the canonical location.
"""
import json
import os
import pathlib

import pytest

YEARS = (2010, 2017, 2024)
RUN_DIR = pathlib.Path(os.environ.get("RUN_DIR", ""))
MLM_LOSS_CEILING = 6.0      # random guessing over a 50,368 vocabulary is about 10.8

# Skipped when RUN_DIR is absent, which is the ordinary local `pytest` run, and NOT skipped when it is
# set to an empty directory, which is the case that must fail. The distinction matters: a file that
# fails in the normal suite teaches everyone to read failures as noise, and a file that passes on an
# empty run dir verifies nothing. `cluster_run.py verify` always sets RUN_DIR.
pytestmark = pytest.mark.skipif(
    "RUN_DIR" not in os.environ,
    reason="verification for a cluster run; RUN_DIR is set by cluster_run.py verify")


def records():
    return sorted(RUN_DIR.glob("task_*_run.json"))


def test_the_run_dir_is_set_and_not_empty():
    """A test that passes on an empty run dir verifies nothing, which is the failure mode this
    project has hit four times in other guises."""
    assert RUN_DIR and RUN_DIR.is_dir(), f"RUN_DIR is not a directory: {RUN_DIR!r}"
    assert records(), f"no task_*_run.json under {RUN_DIR}"


def test_every_vintage_produced_a_record():
    got = {json.loads(p.read_text())["cutoff"] for p in records()}
    want = {f"{y}-12-31" for y in YEARS}
    assert got == want, f"missing {sorted(want - got)}, unexpected {sorted(got - want)}"


@pytest.mark.parametrize("path", records(), ids=lambda p: p.name)
def test_the_run_reached_its_full_token_budget(path):
    """A preempted task that resumed must still have finished. Reporting a short run as a result is
    how the vintages would differ by training length rather than by what they saw."""
    r = json.loads(path.read_text())
    assert r["final_step"] == r["total_steps"], \
        f"{r['name']} stopped at {r['final_step']} of {r['total_steps']}"
    assert r["tokens_seen"] >= 0.99 * r["token_budget"], \
        f"{r['name']} saw {r['tokens_seen']:,} of {r['token_budget']:,} tokens"


@pytest.mark.parametrize("path", records(), ids=lambda p: p.name)
def test_the_loss_is_finite_and_better_than_guessing(path):
    r = json.loads(path.read_text())
    loss = r["final_loss"]
    assert loss == loss, f"{r['name']} final loss is NaN"
    assert 0 < loss < MLM_LOSS_CEILING, \
        f"{r['name']} final loss {loss} is outside (0, {MLM_LOSS_CEILING})"


@pytest.mark.parametrize("path", records(), ids=lambda p: p.name)
def test_it_continued_from_the_matching_vintage_not_a_fresh_model(path):
    """The whole design is that these start from ChronoBERT's checkpoint of the same year. A fresh
    model, or one continued from a different year, would make the comparison meaningless while still
    producing a plausible loss."""
    r = json.loads(path.read_text())
    assert r.get("stage") == "dapt", f"{r['name']} recorded stage {r.get('stage')!r}"
    year = r["cutoff"][:4]
    assert r.get("backbone") == f"chrono-bert-v1-{year}1231", \
        f"{r['name']} continued from {r.get('backbone')!r}"
    assert r.get("backbone_cutoff") == r["cutoff"], \
        f"{r['name']} backbone cutoff {r.get('backbone_cutoff')} against vintage {r['cutoff']}"


@pytest.mark.parametrize("path", records(), ids=lambda p: p.name)
def test_it_used_the_borrowed_vocabulary(path):
    """Training these weights on our own 32,768 vocabulary would give every token id a different
    meaning than it had during their pretraining, and would still converge to something."""
    r = json.loads(path.read_text())
    assert r["tokenizer"] == "registry:chrono-bert-v1-20241231", \
        f"{r['name']} used tokenizer {r['tokenizer']!r}"
    assert r["vocab_size"] == 50368, f"{r['name']} vocab {r['vocab_size']}"
    assert r["architecture"]["total"] > 100e6, \
        f"{r['name']} has {r['architecture']['total']:,} parameters, not a 149 M backbone"


@pytest.mark.parametrize("path", records(), ids=lambda p: p.name)
def test_the_window_prefix_grows_with_the_vintage(path):
    """Each vintage reads a prefix of one packed corpus, so a later cutoff must see more sequences.
    Equal counts would mean the prefix was not applied and every vintage trained on everything."""
    r = json.loads(path.read_text())
    assert r["n_sequences_available"] > 0
    by_year = {json.loads(p.read_text())["cutoff"][:4]: json.loads(p.read_text())["n_sequences_available"]
               for p in records()}
    years = sorted(by_year)
    if len(years) > 1:
        counts = [by_year[y] for y in years]
        assert counts == sorted(counts) and len(set(counts)) == len(counts), \
            f"sequence counts do not increase with the cutoff: {by_year}"
