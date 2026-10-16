import importlib.util
import pathlib

import numpy as np
import pytest
import torch

from ecalls.diagnostics import chronogpt as CG
from ecalls.diagnostics.rag_lookahead import answer_logprob, score_sequences, top_k, vintage

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "chronogpt_original.py"


def test_top_k_orders_best_first_and_excludes():
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [1.0, 0.0]])
    assert top_k(Q, D, k=2).tolist() == [[3, 0], [1, 2]]
    # the excluded column is never returned, and -1 excludes nothing
    assert top_k(Q, D, k=2, exclude=np.array([3, -1])).tolist() == [[0, 2], [1, 2]]


def test_vintage_follows_the_chain_rule():
    assert vintage("chain", 2010) == "ecalls-ret-20101231"
    assert vintage("chain", 2015) == "ecalls-chain-ret-20151231"
    assert vintage("scratch", 2010) == "ecalls-ret-20101231"


def test_answer_logprob_scores_the_answer_one_position_earlier():
    # vocab 4, one row: prompt [0, 1], answer [2, 3]; logits at t predict the token at t+1
    T, V = 4, 4
    logits = torch.full((1, T, V), -10.0)
    logits[0, 1, 2] = 10.0   # position 1 predicts token 2 (first answer token)
    logits[0, 2, 3] = 10.0   # position 2 predicts token 3 (second answer token)
    ids = torch.tensor([[0, 1, 2, 3]])
    good = answer_logprob(logits, ids, [2], [2])[0]
    assert good > -1e-3
    # an off-by-one alignment would read positions 2 and 3 and score badly
    shifted = torch.roll(logits, shifts=1, dims=1)
    assert answer_logprob(shifted, ids, [2], [2])[0] < -5


class _Causal(torch.nn.Module):
    """Logits at t depend only on the token at t, so padding after a sequence changes nothing."""
    def __init__(self, V=11):
        super().__init__()
        torch.manual_seed(0)
        self.table = torch.randn(V, V)

    def forward(self, ids):
        return self.table[ids]


def test_batched_scores_equal_one_at_a_time_and_keep_order():
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(9):
        p, a = int(rng.integers(2, 6)), int(rng.integers(1, 5))
        seqs.append((rng.integers(0, 11, p + a).tolist(), p, a))
    m = _Causal()
    together = score_sequences(m, seqs, "cpu", token_budget=20)
    alone = np.array([score_sequences(m, [s], "cpu")[0] for s in seqs])
    np.testing.assert_allclose(together, alone, rtol=1e-6)
    assert len(set(np.round(alone, 6))) > 1   # the sequences really differ


def _original_module():
    spec = importlib.util.spec_from_file_location("chronogpt_original", FIXTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not FIXTURE.exists(), reason="original model code not present")
def test_vendored_model_matches_the_original_on_a_batch():
    orig = _original_module()
    cfg = dict(vocab_size=50, num_layers=6, num_heads=2, model_dim=32)
    torch.manual_seed(1)
    ref = orig.ChronoGPT(**cfg)
    for p in ref.parameters():          # the original zero-initializes some weights
        torch.nn.init.normal_(p, std=0.2)
    ours = CG.ChronoGPT(**cfg)
    ours.load_state_dict(ref.state_dict())
    ref, ours = ref.to(torch.bfloat16).eval(), ours.to(torch.bfloat16).eval()
    ids = torch.randint(0, 50, (3, 17))
    np.testing.assert_allclose(ours(ids).numpy(), ref(ids).numpy(), rtol=0, atol=0)
    # and the logits are not trivially constant
    assert ours(ids).std().item() > 0.1


def test_prompt_template_matches_the_original():
    src = FIXTURE.read_text()
    assert "### Instruction:" in src and CG.template("x").startswith("\n\n### Instruction:\n")
    assert CG.SYSTEM_PROMPT in src
