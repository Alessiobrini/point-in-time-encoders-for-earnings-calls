"""The contrastive stage: turn a pretrained vintage into a retriever.

Masked language modeling produces a capable language model and a weak retriever, which this project
measured on its own corpus rather than taking from the literature. ChronoBERT is a 149 M MLM-only
encoder and it scores 0.2158 within-call recall@1 where term counting scores 0.4804 and a 33 M
contrastively-trained model scores 0.5378. The reason is that the
masked objective never asks whether two passages mean the same thing, so no whole-text vector is ever
trained and the ones we construct afterwards sit in a narrow cone where unrelated text still scores
high. This stage optimizes the geometry retrieval actually uses.

Four choices carry the design.

**Batches are grouped by call, which is what makes the hard negatives free.** Each batch draws
`calls_per_batch` calls and up to `pairs_per_call` question-answer pairs from each. Every other answer
in the batch is a negative, and the ones from the same call share the company, the quarter, the
speakers and the topic, so separating them requires reading the question rather than recognizing the
firm. Random negatives would be separable on vocabulary alone and teach a shallow rule. The same
in-call pool is what the benchmark scores against, so the training signal and the evaluation are the
same task rather than two related ones.

**A repeated answer in a batch is masked out of the denominator.** Two questions in a call can share
a gold answer, and counting that answer as a negative for the other question trains the model to push
apart two texts that belong together.

**The loss is one-directional, question to answer.** Retrieval here ranks answers for a question and
never the reverse, so the symmetric form would spend half the gradient on a direction nothing uses.

**The batch is a hyperparameter and gradient caching is what allows that.** The batch size here is
the negative count, so scaling it to the allocated card would mean a vintage that drew a 2080 Ti
trained against a different loss than one that drew an A5000, and `scavenger-gpu` gives whatever is
free. The full batch's gradient is instead reassembled from microbatches, exactly rather than
approximately, so only the microbatch varies with the card. See `cached_step` and D23.

**float16 with a gradient scaler, not bfloat16.** L13 measured bfloat16 at 17.0 k tokens/s against
float16's 59.4 k on a 2080 Ti, and 17 of 30 scavenger nodes are 2080 Ti. The masked language modeling
stage stayed on bfloat16 because its loss is a sum over many token positions and the wider exponent
is worth the throughput. This stage is where float16 is worth trying: the pooled embeddings are cast
back to float32 before the similarity matrix, so the part with the narrow dynamic range, a matrix of
cosines divided by a temperature of 0.05, never runs in half precision.

Run:
    python -m ecalls.train.contrastive --mlm ecalls-mlm-20101231 --pairs qa-pairs-20241231 \
        --cutoff 2010-12-31 --require-gpu
    python -m ecalls.train.contrastive --mlm ecalls-chain-mlm-20111231 --pairs qa-pairs-20241231 \
        --cutoff 2011-12-31 --require-gpu          # writes ecalls-chain-ret-20111231
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

from ecalls.config import paths as P
from ecalls.ingest import manifest as M
from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.train import model as MD
from ecalls.train import pairs as PR
from ecalls.train import pretrain as PT
from ecalls.train import tokenizer as TK

MODELS_DIR = P.PROCESSED_DIR / "models"

TEMPERATURE = 0.05          # in the 0.01 to 0.1 range the SimCSE line of work uses
# Measured over 40,000 sampled pairs, and the two sides are not alike: questions average 83 tokens
# and answers 187. One cap of 256 truncates 0.88% of questions but 22.46% of answers and throws away
# 16.8% of all answer tokens. So the caps are asymmetric. 256 on the question side truncates 0.88%,
# 512 on the answer side truncates 3.46%. 512 is also what `bge-small-en-v1.5` accepts, and it scored
# 0.5378 with that budget, so a shorter answer cap here would rig the comparison against our own
# model rather than against the corpus. See D22.
MAX_LENGTH_QUERY = 256
MAX_LENGTH_ANSWER = 512
MAX_LENGTH = MAX_LENGTH_ANSWER      # what the registry reports, since evaluation caps both sides
# The batch is a hyperparameter here, not a memory setting, and gradient caching is what lets it be
# one. 32 calls x 16 pairs gives each anchor 15 in-call hard negatives, which matches the benchmark's
# within-call pool median of about 15, plus 496 cross-call negatives.
CALLS_PER_BATCH = 32
PAIRS_PER_CALL = 16
TOTAL_STEPS = 4000          # held fixed across vintages, as the token budget is in the MLM stage
PEAK_LR = 2e-5              # a fine-tune on a pretrained backbone, not a fresh run
WARMUP_FRAC = 0.05
# How many sequences of one side fit on the card at a time. Measured on a 2080 Ti with the D22 caps
# and checkpointing on: 128 anchors took 3.51 GiB, 256 took 6.27 and 512 took 8.18 of 10.58, so about
# 48 sequences per GiB at the answer cap. Halved for headroom, since batch content varies in length.
SEQ_PER_GB = 24.0
MAX_MICROBATCH = 256


def model_dir(name: str) -> Path:
    return MODELS_DIR / name


# `chain-mlm` is the chained family of D29 and `mlm2x` the from-scratch 2024 vintage at twice the
# budget. The chain has no retriever of its own for 2010, because its base is the from-scratch
# `ecalls-mlm-20101231` and the contrastive stage on it is `ecalls-ret-20101231`, so that one model
# serves as the 2010 member of both families.
#
# The tag goes in front for the 2x run, `ecalls-2x-ret-*` rather than `ecalls-ret2x-*`, because the
# registry resolves a family by `model_id` prefix. `ecalls-ret2x-20241231` starts with `ecalls-ret`,
# so it would tie with `ecalls-ret-20241231` and make `for_date(..., family="ecalls-ret")` raise for
# every date from 2025 on. The seed replicates are named the same way for the same reason.
RUN_TAGS = {"ecalls-mlm": "ret", "registered": "abl", "dapt": "dapt-ret",
            "chain-mlm": "chain-ret", "mlm2x": "2x-ret", "rpl-mlm": "rpl-ret"}
# The pretraining run's `family` field, mapped to the backbone kind above. A record written before
# the field existed is read from its stage, which then means `mlm` or `dapt`.
FAMILY_KINDS = {"mlm": "ecalls-mlm", "dapt": "dapt", "chain-mlm": "chain-mlm", "mlm2x": "mlm2x",
                "rpl-mlm": "rpl-mlm"}
LINEAGE_KEYS = ("family", "init_from", "init_from_cutoff", "replay_of", "start_cutoff", "first_sequence",
                "last_sequence", "token_budget", "tokens_seen")


def run_name(cutoff: str, backbone_kind: str = "ecalls-mlm", seed: int = 0) -> str:
    """`ecalls-ret-YYYYMMDD` for ours, `ecalls-abl-YYYYMMDD` for an ablation on a foreign backbone,
    `ecalls-dapt-ret-YYYYMMDD` for the same stage on a backbone continued on our corpus,
    `ecalls-chain-ret-YYYYMMDD` on a chained segment and `ecalls-2x-ret-YYYYMMDD` on the 2x budget run.

    No two may share a name. They are the same contrastive stage on the same pairs and differ
    only in what they start from, which is the whole point of running them, so one overwriting another
    would destroy the comparison.

    **A seed other than 0 goes in front of the tag, never after the date.** The registry resolves a
    family by `model_id` prefix and reads the date as the last field. `ecalls-ret-20101231-s1` would
    match the family `ecalls-ret`, tie with seed 0 at the same cutoff, and make `for_date` raise for
    every caller, the premise test among them. `ecalls-seed1-ret-20101231` is a family of its own. It
    also means a seed run that forgets `--name` cannot resume into the deliverable's finished
    checkpoint and overwrite its export.
    """
    if backbone_kind not in RUN_TAGS:
        raise ValueError(f"unknown backbone kind {backbone_kind!r}, expected one of {sorted(RUN_TAGS)}")
    tag = RUN_TAGS[backbone_kind]
    if seed:
        tag = f"seed{seed}-{tag}"
    return f"ecalls-{tag}-{cutoff.replace('-', '')}"


def load_backbone(mlm_name: str, vocab: int, device, seed: int = 0):
    """The pretrained encoder body, with the masked language modeling head dropped.

    Loading the full MLM model and taking its `.model` submodule rather than remapping keys into a
    bare encoder: the prefixes differ between the two classes and a silent partial load would leave
    a randomly initialized layer in place, which shows up as a mediocre retriever rather than an error.
    """
    import torch

    d = PT.model_dir(mlm_name)
    f = d / "checkpoint.pt"
    if not f.exists():
        raise FileNotFoundError(
            f"no pretrained checkpoint at {f}. The contrastive stage starts from a masked language "
            f"modeling run, not from scratch.")
    mlm = MD.build_mlm(vocab, seed=seed)
    ck = torch.load(f, map_location="cpu", weights_only=False)
    mlm.load_state_dict(ck["model"], strict=True)     # strict, so a partial load raises
    body = mlm.model                      # ModernBertModel inside ModernBertForMaskedLM
    run = json.loads((d / "_run.json").read_text())
    return body.to(device), run


def load_dapt_backbone(dapt_name: str, device):
    """A ChronoBERT checkpoint continued on our corpus, with the masked language modeling head dropped.

    `load_backbone` cannot read it, because it builds our 33.8 M architecture and this is ChronoBERT's
    149 M one. The architecture is therefore rebuilt from the registered backbone the continued run
    started from, and the continued weights are loaded into it strictly, so a partial load raises
    rather than leaving ChronoBERT's original layers in place, which would score like the ablation and
    look like a finding.

    The tokenizer is the registered one the corpus was packed with. Our own BPE wrapper cannot be
    used, since this model's vocabulary is ChronoBERT's 50,368 tokens.

    Refuses an unfinished run. A preempted continued-pretraining job leaves a valid checkpoint part
    way through its budget, and a retriever built on it would carry a backbone that saw a fraction of
    the corpus while its record said otherwise.
    """
    import torch
    from transformers import AutoTokenizer

    from ecalls.chunk import registry as R

    d = PT.model_dir(dapt_name)
    f = d / "_run.json"
    if not f.exists():
        raise FileNotFoundError(
            f"{dapt_name} has no _run.json, so its continued pretraining has not finished. The "
            "trainer writes it only after the last step.")
    run = json.loads(f.read_text())
    if run.get("stage") != "dapt":
        raise ValueError(f"{dapt_name} is stage {run.get('stage')!r}, not 'dapt'")
    if run.get("final_step") != run.get("total_steps"):
        raise ValueError(
            f"{dapt_name} stopped at step {run.get('final_step')} of {run.get('total_steps')}. The "
            "comparison holds the token budget fixed, so a partial run is a different experiment.")
    model, _, _ = PT.load_mlm_backbone(run["backbone"], "cpu")
    ck = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    tok_entry = R.load().by_name(run["tokenizer"].removeprefix(TK.REGISTRY_PREFIX))
    tok = AutoTokenizer.from_pretrained(tok_entry.source.removeprefix("huggingface:"))
    if len(tok) != model.config.vocab_size:
        raise ValueError(
            f"{dapt_name}: tokenizer {run['tokenizer']} has {len(tok):,} entries but the model "
            f"expects {model.config.vocab_size:,}")
    backbone_entry = R.load().by_name(run["backbone"])
    return model.model.to(device), tok, run, backbone_entry.pooling


REFERENCE_PREFIX = "ref-"


def check_reference(kind: str, reference: bool, name: str | None) -> None:
    """The guard on a backbone with no known cutoff (D34).

    The stage refuses an ATEMPORAL backbone, because its retriever could not carry a cutoff. A reviewer
    asked how much of our lead over general retrievers comes from fine-tuning on the calls, which only a
    general retriever fine-tuned on the same pairs answers, so `reference` lets one through. Its name
    must start with `ref-`, which no family pattern in `scripts/register_vintages.py` matches, so it can
    never be registered as a vintage or picked by `for_date`.
    """
    if kind == "vintage":
        return
    if not reference:
        raise ValueError(
            f"the backbone is registered as {kind}, so it has no known training cutoff and a "
            "retriever built on it could not be registered as a vintage either. Pass reference=True "
            "for a reference model that is never a vintage.")
    if not (name or "").startswith(REFERENCE_PREFIX):
        raise ValueError(f"a reference model needs a name starting with {REFERENCE_PREFIX!r}, got {name!r}")


def load_registered_backbone(model_id: str, device, compute_dtype: str = "float32",
                             reference: bool = False, name: str | None = None):
    """A registered encoder as the backbone, which is how the ablation is run.

    Beating `bge-small-en-v1.5` does not by itself show that finance pretraining helps, because our
    model also gets a contrastive stage on in-domain pairs and trains on exactly the benchmark's task
    while bge-small gets neither. Running this same stage, on these same pairs, from a general-domain
    backbone with a known cutoff separates the three: ours against that isolates the pretraining, and
    that against bge-small isolates the in-domain pairs.

    The backbone must be a `VINTAGE` entry, because the result has to carry a cutoff like ours does.
    Capacity is not matched, ChronoBERT being 149 M against our 33.8 M, which is stated rather than
    corrected: if the smaller finance-pretrained model still wins, that is the stronger result.
    """
    from ecalls.chunk import registry as R
    from ecalls.embed.encode import load_encoder

    entry = R.load().by_name(model_id)
    check_reference(entry.kind, reference, name)
    tok, model, dev = load_encoder(entry, device, compute_dtype=compute_dtype)
    return model.train(), tok, entry


def _stage_of(mlm_name: str) -> str | None:
    f = PT.model_dir(mlm_name) / "_run.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("stage")


def _backbone_kind(mlm_name: str) -> str:
    """Which family a pretraining run belongs to, read from its own record.

    A chained segment or a 2x run without its record is refused here, because the record is written
    only after the last step and both families exist to be compared at a finished state. A plain
    vintage keeps its earlier behavior and fails later on the missing file.
    """
    f = PT.model_dir(mlm_name) / "_run.json"
    if not f.exists():
        if mlm_name.startswith(("ecalls-chain-mlm-", "ecalls-mlm2x-", "ecalls-rpl-mlm-")):
            PT.require_finished(mlm_name)
        return "ecalls-mlm"
    run = json.loads(f.read_text())
    family = run.get("family") or run.get("stage") or "mlm"
    if family not in FAMILY_KINDS:
        raise ValueError(f"{mlm_name} is family {family!r}, which has no contrastive stage")
    return FAMILY_KINDS[family]


# --- batching ------------------------------------------------------------------------------------

class CallSampler:
    """Draws calls without replacement within an epoch, so no call is oversampled early in training.

    State round-trips through the checkpoint for the same reason the pretraining sampler's does: a
    scavenger job is preempted and restarted, and restarting the sampler would re-show the calls it
    had already used while never reaching the tail.
    """

    def __init__(self, n_calls: int, per_batch: int, seed: int = 0):
        self.n, self.per_batch, self.seed = n_calls, per_batch, seed
        self.epoch, self.pos = 0, 0
        self._order = self._shuffle(0)

    def _shuffle(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, epoch])
        return rng.permutation(self.n)

    def next(self) -> np.ndarray:
        """`per_batch` call indices, wrapping into the next epoch rather than returning a short batch.

        A short final batch would have fewer negatives than every other, which changes the loss scale
        rather than just the sample, so the batch is filled from the next epoch instead.
        """
        out = []
        while len(out) < self.per_batch:
            if self.pos >= self.n:
                self.epoch += 1
                self._order = self._shuffle(self.epoch)
                self.pos = 0
            take = min(self.per_batch - len(out), self.n - self.pos)
            out.extend(self._order[self.pos:self.pos + take].tolist())
            self.pos += take
        return np.array(out)

    def state(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos, "seed": self.seed, "n": self.n}

    def load_state(self, s: dict) -> None:
        if s["n"] != self.n:
            raise ValueError(
                f"the checkpoint was written against {s['n']:,} calls but this run has {self.n:,}. "
                "Resuming would sample a different set than the run being resumed.")
        self.epoch, self.pos = s["epoch"], s["pos"]
        self._order = self._shuffle(self.epoch)


def group_pairs(df: pl.DataFrame) -> tuple[list[np.ndarray], np.ndarray]:
    """Row indices per call, and the call ids, both in a fixed order.

    Calls with a single pair are kept. They contribute no in-call hard negative of their own but they
    are still valid anchors against the rest of the batch, and dropping them would bias the training
    set toward long Q&A sessions.
    """
    df = df.with_row_index("_row")
    g = df.group_by("keydevid", maintain_order=True).agg(pl.col("_row"))
    groups = [np.asarray(x, dtype=np.int64) for x in g["_row"].to_list()]
    return groups, g["keydevid"].to_numpy()


def draw_batch(groups, call_idx: np.ndarray, pairs_per_call: int, rng) -> np.ndarray:
    """Row indices for one batch: up to `pairs_per_call` pairs from each of the chosen calls."""
    rows = []
    for c in call_idx:
        g = groups[c]
        if len(g) > pairs_per_call:
            g = rng.choice(g, size=pairs_per_call, replace=False)
        rows.extend(g.tolist())
    return np.array(rows, dtype=np.int64)


# --- the loss ------------------------------------------------------------------------------------

def info_nce(q, a, answer_ids: np.ndarray, temperature: float = TEMPERATURE):
    """Cross-entropy over cosine similarities, question to answer, with duplicates masked out.

    `q` and `a` are L2-normalized and float32, so the matrix product is a matrix of cosines. The
    diagonal is the gold pair. A candidate that repeats another anchor's gold answer is masked to
    -inf off the diagonal, because counting it as a negative asks the model to separate two texts
    that belong together.
    """
    import torch

    sim = (q @ a.T) / temperature
    n = sim.shape[0]
    same = torch.from_numpy(answer_ids[:, None] == answer_ids[None, :]).to(sim.device)
    off_diagonal_duplicate = same & ~torch.eye(n, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(off_diagonal_duplicate, float("-inf"))
    target = torch.arange(n, device=sim.device)
    return torch.nn.functional.cross_entropy(sim, target)


def embed(model, tok, texts, device, max_length: int, dtype, pooling: str = "mean"):
    """Pooled, L2-normalized vectors in float32, with the pooling done outside autocast.

    Mean pooling divides by the real token count, so in float16 a long sequence sums several hundred
    values before dividing. The sum is taken in float32 to keep that exact, and the similarity matrix
    that follows is float32 for the same reason.

    `pooling` comes from the backbone rather than being fixed, because the ablation runs this same
    stage on a downloaded encoder and the rule is a property of how that model was trained. Ours is
    mean, since a model with no contrastive stage has no trained `CLS` position.
    """
    import torch

    from ecalls.embed.encode import POOLERS

    if pooling not in POOLERS:
        raise ValueError(f"pooling must be one of {sorted(POOLERS)}, got {pooling!r}")
    enc = tok(list(texts), padding=True, truncation=True, max_length=max_length,
              return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
        hidden = model(**enc).last_hidden_state
    vec = POOLERS[pooling](hidden.float(), enc["attention_mask"])
    return torch.nn.functional.normalize(vec, p=2, dim=1)


# --- gradient caching: the negative count must not depend on the card ---------------------------

def cached_step(model, tok, q_texts, a_texts, answer_ids, device, max_length_query: int,
                max_length_answer: int, dtype, temperature: float, microbatch: int, scaler,
                pooling: str = "mean"):
    """One optimizer step's worth of gradient for the full batch, in microbatch-sized pieces.

    This is the contrastive analogue of the rule the masked language modeling stage follows: tokens
    per step is the hyperparameter and the microbatch is only how much fits on the allocated card.
    Here **the batch size is the negative count**, so scaling it to the device would mean a vintage
    that drew a 2080 Ti trained against a different loss than one that drew an A5000, and
    `scavenger-gpu` hands out whatever is free. A difference between two vintages would then be partly
    a difference in which GPU they happened to get, which is the L14 confound in a worse form because
    nothing in the run record would show it.

    The method is gradient caching, from Gao, Zhang, Han and Callan (2021), *Scaling Deep Contrastive
    Learning Batch Size under Memory Limited Setup*, doi 10.18653/v1/2021.repl4nlp-1.31. Three passes:

    1. Embed everything with no gradient, giving the full matrices at the cost of one microbatch.
    2. Compute the loss on those matrices and backpropagate **only into them**, caching one gradient
       vector per embedding. The similarity matrix is the only thing held at full batch size, and it
       is n by n floats rather than n sequences of activations.
    3. Re-embed each microbatch with gradients on and seed its backward pass with the cached vectors.

    The result is the exact gradient of the full-batch loss, not an approximation, which is what makes
    the batch a real hyperparameter. Under a gradient scaler the loss is scaled in pass 2, so the
    cached vectors are already scaled and pass 3 propagates them into the parameters unchanged.
    """
    import torch

    def in_pieces(texts, cap, grad=None):
        out = []
        for i in range(0, len(texts), microbatch):
            piece = texts[i:i + microbatch]
            if grad is None:
                with torch.no_grad():
                    out.append(embed(model, tok, piece, device, cap, dtype, pooling))
            else:
                v = embed(model, tok, piece, device, cap, dtype, pooling)
                v.backward(gradient=grad[i:i + len(piece)])
        return torch.cat(out) if grad is None else None

    Q = in_pieces(q_texts, max_length_query).detach().requires_grad_(True)
    A = in_pieces(a_texts, max_length_answer).detach().requires_grad_(True)
    loss = info_nce(Q, A, answer_ids, temperature)
    (scaler.scale(loss) if scaler is not None else loss).backward()
    gQ, gA = Q.grad.detach(), A.grad.detach()
    del Q, A

    in_pieces(q_texts, max_length_query, gQ)
    in_pieces(a_texts, max_length_answer, gA)
    return loss.detach()


# --- device sizing -------------------------------------------------------------------------------

def auto_microbatch(device: str, batch: int, target_per_gb: float = SEQ_PER_GB) -> int:
    """How many sequences to embed at once. This is the only thing that varies with the card.

    The masked language modeling stage learned the cost of a fixed value the expensive way: a
    microbatch of 32 that a probe ran comfortably on a 49 GiB card killed twelve of fifteen tasks on
    10.58 GiB cards. The difference here is that shrinking this number does not change the gradient at
    all, because gradient caching reassembles the full batch's gradient from the pieces, so it can be
    scaled freely and a small card costs time rather than a different loss.
    """
    if device != "cuda":
        return min(batch, 8)
    import torch

    gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return max(4, min(batch, MAX_MICROBATCH, int(gb * target_per_gb)))


# --- checkpointing -------------------------------------------------------------------------------

def save_checkpoint(d: Path, model, opt, scaler, sampler, step: int, meta: dict) -> None:
    import torch

    d.mkdir(parents=True, exist_ok=True)
    # A per-writer temp name, for the reason `manifest.atomic_write_table` documents: a fixed
    # `.tmp` is safe for one writer and destructive for two. Two concurrent ablation arrays proved
    # it here, one dying with its temp file renamed out from under it at step 3,800 of 4,000.
    tmp = M.temp_sibling(d / "checkpoint.pt")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "sampler": sampler.state(), "step": step,
                "anchors_drawn": meta.get("anchors_drawn")}, tmp)
    os.replace(tmp, d / "checkpoint.pt")
    M.refresh_claim(d)          # a live run must not look abandoned to the next claimant
    M.atomic_write_json({**meta, "step": step, "updated_at": utc_now()}, d / "_progress.json")


def load_checkpoint(d: Path, model, opt, scaler, sampler, device) -> tuple[int, int | None]:
    """The step to resume from and the anchors drawn so far, `None` for a checkpoint that predates
    the count."""
    import torch

    f = d / "checkpoint.pt"
    if not f.exists():
        return 0, 0
    ck = torch.load(f, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    sampler.load_state(ck["sampler"])
    return int(ck["step"]), ck.get("anchors_drawn")


# --- the held-out progress check -----------------------------------------------------------------

def holdout_recall(model, tok, df: pl.DataFrame, device, max_length: int, dtype,
                   max_queries: int = 512, batch: int = 32, pooling: str = "mean") -> dict:
    """Within-call recall@1 on held-out calls, for watching a run rather than for reporting.

    **This is not the benchmark number and must not be compared against 0.5378.** The candidate pool
    here is the gold answers of the sampled queries, while `diagnostics.bow_vs_dense` pools every
    executive turn in the call above the length floor, which is a larger and harder pool. The two
    also differ in population, since 0.5378 was measured with no holdout filter. This function exists
    so a six-hour run reports whether it is learning at all; the comparison runs through
    `bow_vs_dense --holdout-only`, which builds one pool and scores every model against it.

    Ties are
    broken pessimistically, matching `diagnostics.bow_vs_dense.rank_metrics`: a gold answer tied with
    a distractor counts as a miss, because on this corpus adjacent neighbours differ by about 1e-4 and
    an optimistic tie would report a win for reordering indistinguishable text.
    """
    import torch

    from ecalls.diagnostics.bow_vs_dense import rank_metrics

    sub = df.head(max_queries)
    if sub.height == 0:
        raise ValueError("the held-out pair set is empty, so the progress check measures nothing")
    q_text, a_text = sub["text"].to_list(), sub["answer_text"].to_list()
    call = sub["keydevid"].to_numpy()

    was_training = model.training
    model.eval()
    with torch.no_grad():
        Q = torch.cat([embed(model, tok, q_text[i:i + batch], device, max_length, dtype, pooling)
                       for i in range(0, len(q_text), batch)])
        A = torch.cat([embed(model, tok, a_text[i:i + batch], device, max_length, dtype, pooling)
                       for i in range(0, len(a_text), batch)])
    if was_training:
        model.train()
    S = (Q @ A.T).float().cpu().numpy()
    gold = np.arange(len(q_text))
    mask = call[:, None] == call[None, :]        # candidates from the same call
    return {"within_call": rank_metrics(S, gold, mask),
            "cross_call": rank_metrics(S, gold, None),
            "n_queries": len(q_text)}


# --- training ------------------------------------------------------------------------------------

def train(mlm: str | None, pairs: str, cutoff: str | None = None,
          holdout_pairs: str | None = None, backbone: str | None = None,
          total_steps: int = TOTAL_STEPS, calls_per_batch: int = CALLS_PER_BATCH,
          microbatch: int | None = None,
          pairs_per_call: int = PAIRS_PER_CALL, temperature: float = TEMPERATURE,
          peak_lr: float = PEAK_LR, warmup_frac: float = WARMUP_FRAC,
          max_length_query: int = MAX_LENGTH_QUERY, max_length_answer: int = MAX_LENGTH_ANSWER,
          seed: int = 0, device: str | None = None, compute_dtype: str = "float16",
          checkpoint_every: int = 200, log_every: int = 25, eval_every: int = 500,
          name: str | None = None, require_gpu: bool = False,
          gradient_checkpointing: bool = True, reference: bool = False) -> dict:
    """Run the contrastive stage for one vintage. Idempotent under preemption."""
    import torch

    from ecalls.embed.encode import resolve_device

    df, pmeta = PR.load_pairs(pairs)
    if pmeta.get("holdout_only"):
        raise ValueError(
            f"{pairs} is the held-out pair set. Training on it would train on the calls the "
            "benchmark scores, and the resulting number would be memorization.")
    # The pair set is written shard by shard in calendar order and carries the shard on every row, so
    # a vintage's pairs are a subset selected by shard rather than a separate build.
    vintage_cutoff = cutoff or pmeta["cutoff"]
    if cutoff:
        from ecalls.embed.encode import shards_up_to
        allowed = set(shards_up_to(cutoff, sorted(set(df["shard"].to_list()))))
        if not allowed:
            raise ValueError(f"no shard of {pairs} ends on or before {cutoff}")
        df = df.filter(pl.col("shard").is_in(allowed))
        if df.height == 0:
            raise ValueError(f"no pair of {pairs} falls inside the window ending {cutoff}")

    device = resolve_device(device, require_gpu=require_gpu)
    dt = {"float32": None, "float16": torch.float16, "bfloat16": torch.bfloat16}[compute_dtype]
    if device == "cpu":
        dt = None
    if microbatch is None:
        microbatch = auto_microbatch(device, calls_per_batch * pairs_per_call)

    # The tokenizer comes from the backbone's own record, not from a default. Every vintage reuses the
    # one fit on the earliest window (D20), and reading it from the run record means a mismatch
    # between the two stages raises here rather than producing a model with a shifted vocabulary.
    if (mlm is None) == (backbone is None):
        raise ValueError(
            "pass exactly one of mlm= (a vintage pretrained here) or backbone= (a registered "
            "encoder, for the ablation). Passing both would leave it ambiguous which weights the "
            "recorded cutoff belongs to.")

    backbone_kind = "registered"
    lineage = None
    mlm_kind = _backbone_kind(mlm) if mlm is not None else None
    if mlm_kind == "dapt":
        # Continued from ChronoBERT on our corpus. Everything but the starting weights matches the
        # ablation, which is the comparison this exists for, so the pooling is the ablation's too.
        model, tok, run_meta_mlm, pooling = load_dapt_backbone(mlm, device)
        tokenizer_name = run_meta_mlm["tokenizer"]
        vocab = int(model.config.vocab_size)
        backbone_cutoff = run_meta_mlm.get("cutoff")
        backbone_id = mlm
        backbone_kind = "dapt"
    elif mlm is not None:
        if mlm_kind != "ecalls-mlm":
            # The chained and 2x families are compared against the finished vintages, so a
            # preempted segment must not pass for one. The plain vintages keep their earlier path.
            PT.require_finished(mlm)
        run_meta_mlm = json.loads((PT.model_dir(mlm) / "_run.json").read_text())
        tokenizer_name = run_meta_mlm["tokenizer"]
        vocab = run_meta_mlm["vocab_size"]
        backbone_cutoff = run_meta_mlm.get("cutoff")
        backbone_id = mlm
        pooling = "mean"      # no contrastive stage yet, so no trained CLS position
        tok = hf_tokenizer(tokenizer_name, max(max_length_query, max_length_answer))
        model, _ = load_backbone(mlm, vocab, device, seed=seed)
        backbone_kind = mlm_kind
        # Copied so the retriever's own record, and through it the registry, states which parent a
        # chained backbone continued and which slice of the corpus it saw.
        lineage = {k: run_meta_mlm.get(k) for k in LINEAGE_KEYS}
    else:
        model, tok, entry = load_registered_backbone(backbone, device, compute_dtype="float32",
                                                     reference=reference, name=name)
        if reference:
            backbone_kind = "reference"
        tokenizer_name = f"registry:{backbone}"
        vocab = int(getattr(model.config, "vocab_size", 0))
        backbone_cutoff = entry.training_cutoff
        backbone_id = backbone
        pooling = entry.pooling
    if backbone_cutoff and backbone_cutoff != vintage_cutoff:
        raise ValueError(
            f"{backbone_id} was pretrained to {backbone_cutoff} but this stage is training the "
            f"{vintage_cutoff} vintage. A retriever inherits its backbone's cutoff, so the two must "
            "agree or the recorded cutoff is a fiction.")
    # Measured on a 2080 Ti, the card 17 of 30 scavenger nodes carry: 64 anchors took 6.14 GiB
    # without checkpointing and 128 would not fit, while checkpointed 128 anchors took 2.24 GiB at
    # 202 anchor/s against 266. In this stage the batch size *is* the negative count, so paying a
    # quarter of the throughput for several times the negatives is the right trade, and it is the
    # opposite of the call the masked language modeling stage makes, where the batch is only memory.
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    name = name or run_name(vintage_cutoff, backbone_kind, seed)
    d = model_dir(name)
    # Refuse a second trainer on this model rather than taking turns with it. Two identical arrays
    # over these fifteen directories is how this was found, and nothing in the artifacts said so.
    M.claim_dir(d, label=f"contrastive {name}")

    groups, call_ids = group_pairs(df)
    sampler = CallSampler(len(groups), calls_per_batch, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6,
                            weight_decay=0.01)
    scaler = torch.amp.GradScaler(device) if dt is torch.float16 else None
    warmup = int(total_steps * warmup_frac)

    answer_ids = df["answer_id"].to_numpy()
    q_all, a_all = df["text"].to_list(), df["answer_text"].to_list()

    meta = {
        "name": name, "stage": "contrastive", "backbone": backbone_id, "pairs": pairs,
        "backbone_kind": backbone_kind, "backbone_lineage": lineage,
        "pooling_from_backbone": pooling,
        "cutoff": vintage_cutoff, "pairs_cutoff": pmeta["cutoff"],
        "tokenizer": tokenizer_name, "vocab_size": vocab,
        "n_pairs": df.height, "n_calls": len(groups),
        "total_steps": total_steps, "calls_per_batch": calls_per_batch,
        # The most anchors a batch can hold. A call with fewer pairs gives all it has, so the batch
        # actually drawn is smaller, and `mean_anchors_per_batch` in the final record says by how much
        # (audit of 2026-10-13: 405 at the 2010 cutoff, about 368 at 2024).
        "pairs_per_call": pairs_per_call, "anchors_per_batch": calls_per_batch * pairs_per_call,
        "microbatch": microbatch,
        "temperature": temperature, "peak_lr": peak_lr, "warmup_steps": warmup,
        "max_length_query": max_length_query, "max_length_answer": max_length_answer,
        "max_length": max(max_length_query, max_length_answer),
        "compute_dtype": compute_dtype, "grad_scaler": scaler is not None,
        "pooling": pooling, "normalized": True, "loss": "info_nce_question_to_answer",
        "gradient_checkpointing": gradient_checkpointing,
        "seed": seed, "device": device, "reference": reference,
        "holdout_salt": pmeta.get("holdout_salt"), "holdout_fraction": pmeta.get("holdout_fraction"),
        "git_sha": git_sha(),
    }

    step, anchors_drawn = load_checkpoint(d, model, opt, scaler, sampler, device)
    if step:
        print(f"resuming {name} at step {step}/{total_steps}", flush=True)
    else:
        print(f"{name}: {total_steps:,} steps of {calls_per_batch} calls x {pairs_per_call} pairs "
              f"= {calls_per_batch * pairs_per_call} anchors in microbatches of {microbatch} "
              f"on {device}, {df.height:,} pairs over {len(groups):,} calls", flush=True)

    ho = None
    if holdout_pairs:
        hdf, hmeta = PR.load_pairs(holdout_pairs)
        if not hmeta.get("holdout_only"):
            raise ValueError(
                f"{holdout_pairs} is a training pair set, not a held-out one. The progress check "
                "would score the model on calls it is training on.")
        if cutoff:
            hdf = hdf.filter(pl.col("shard").is_in(allowed))
        ho = hdf

    rng = np.random.default_rng([seed, step])
    model.train()
    t0, hist = time.time(), []
    evals = []

    while step < total_steps:
        lr = PT.lr_at(step, total_steps, peak_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        rows = draw_batch(groups, sampler.next(), pairs_per_call, rng)
        if anchors_drawn is not None:
            anchors_drawn += len(rows)
            meta["anchors_drawn"] = anchors_drawn
        opt.zero_grad(set_to_none=True)
        # The gradient is assembled from microbatches but is the full batch's, exactly, so the
        # number of negatives is identical on every card. See `cached_step`.
        # Halve the microbatch and retry on out-of-memory. This is free of consequence here in a way
        # it is not in the masked language modeling stage: the microbatch does not enter the
        # gradient, so a task that shrinks it computes the same step as one that did not, and the
        # ablation runs a 149 M backbone through a budget calibrated on 33.8 M.
        while True:
            try:
                loss = cached_step(model, tok, [q_all[i] for i in rows], [a_all[i] for i in rows],
                                   answer_ids[rows], device, max_length_query, max_length_answer,
                                   dt, temperature, microbatch, scaler, pooling)
                break
            except torch.cuda.OutOfMemoryError:
                if microbatch <= 1:
                    raise
                microbatch = max(1, microbatch // 2)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                meta["microbatch"] = microbatch
                print(f"  OOM at step {step}, microbatch -> {microbatch} "
                      f"(the gradient is unchanged)", flush=True)
        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        step += 1
        hist.append(float(loss))

        if step % log_every == 0:
            rate = step / max(1e-9, time.time() - t0)
            print(f"  step {step}/{total_steps}  loss {np.mean(hist[-log_every:]):.4f}  "
                  f"lr {lr:.2e}  {rate:.2f} step/s  epoch {sampler.epoch}", flush=True)
        if ho is not None and eval_every and step % eval_every == 0:
            r = holdout_recall(model, tok, ho, device, max_length_answer, dt, pooling=pooling)
            evals.append({"step": step, **r})
            print(f"  step {step}: held-out within-call R@1 "
                  f"{r['within_call']['recall_at_1']:.4f}  MRR {r['within_call']['mrr']:.4f}",
                  flush=True)
        if step % checkpoint_every == 0:
            save_checkpoint(d, model, opt, scaler, sampler, step, meta)

    save_checkpoint(d, model, opt, scaler, sampler, step, meta)
    final = (holdout_recall(model, tok, ho, device, max_length_answer, dt, pooling=pooling)
             if ho is not None else None)
    if final:
        evals.append({"step": step, **final})
    export_hf(d, model, tok, meta)

    out = {**meta, "final_step": step, "final_loss": float(np.mean(hist[-50:])) if hist else None,
           "epochs_equivalent": sampler.epoch + sampler.pos / max(1, sampler.n),
           "holdout_evals": evals,
           "mean_anchors_per_batch": (anchors_drawn / step if anchors_drawn is not None and step else None),
           "final_holdout_recall_at_1": (final["within_call"]["recall_at_1"] if final else None),
           "finished_at": utc_now()}
    M.atomic_write_json(out, d / "_run.json")
    M.release_dir(d)
    return out


# --- export --------------------------------------------------------------------------------------

def hf_tokenizer(name: str, max_length: int = MAX_LENGTH):
    """Our byte-level BPE wrapped as a transformers tokenizer.

    The rest of the pipeline calls a tokenizer the transformers way, `tok(texts, padding=True, ...)`,
    so wrapping is what lets `embed/encode.py`, the benchmark and the index use these models with no
    special case. The wrapper adds no vocabulary and fits nothing: it is the same file the vintages
    were pretrained with.
    """
    from transformers import PreTrainedTokenizerFast

    path = TK.tokenizer_path(name) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"no tokenizer at {path}")
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path), model_max_length=max_length,
        pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]")


def export_hf(d: Path, model, tok, meta: dict) -> Path:
    """Write the model where `AutoModel.from_pretrained` can read it.

    The registry resolves a model by name to a source, and this is what makes `local:` sources work,
    so the trained vintages flow through the same embedding pass, benchmark and index as the
    downloaded ones with no special case. It is also the releasable artifact: the corpus cannot be
    redistributed but weights can.
    """
    out = d / "hf"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "_provenance.json").write_text(json.dumps(
        {k: meta[k] for k in ("name", "stage", "backbone", "pairs", "cutoff", "tokenizer",
                              "pooling", "normalized", "temperature", "max_length",
                              "max_length_query", "max_length_answer", "git_sha")
         if k in meta}, indent=2) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g0 = ap.add_mutually_exclusive_group(required=True)
    g0.add_argument("--mlm", help="a vintage pretrained here, by model directory name")
    g0.add_argument("--backbone", help="a registered encoder instead, for the ablation that "
                                      "separates finance pretraining from the contrastive stage")
    ap.add_argument("--pairs", required=True, help="the training pair set")
    ap.add_argument("--holdout-pairs", default=None, help="the held-out pair set for the progress check")
    ap.add_argument("--cutoff", default=None,
                    help="restrict the pairs to this vintage's window; one pair set serves all")
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--calls-per-batch", type=int, default=CALLS_PER_BATCH,
                    help="a hyperparameter, not a memory setting: it is the negative count")
    ap.add_argument("--microbatch", type=int, default=None,
                    help="how many sequences are embedded at once; scaled from the card by default "
                         "and it does not change the gradient")
    ap.add_argument("--pairs-per-call", type=int, default=PAIRS_PER_CALL)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--lr", type=float, default=PEAK_LR)
    ap.add_argument("--max-length-query", type=int, default=MAX_LENGTH_QUERY)
    ap.add_argument("--max-length-answer", type=int, default=MAX_LENGTH_ANSWER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--compute-dtype", default="float16", choices=("float32", "float16", "bfloat16"))
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--name", default=None)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false",
                    help="faster per step but several times fewer negatives fit; see the probe")
    ap.set_defaults(gradient_checkpointing=True)
    ap.add_argument("--reference", action="store_true",
                    help="allow a backbone with no known cutoff, for a reference model named ref-* that "
                         "is never a vintage (D34)")
    args = ap.parse_args(argv)

    r = train(args.mlm, args.pairs, args.cutoff, args.holdout_pairs, args.backbone, args.steps,
              args.calls_per_batch, args.microbatch, args.pairs_per_call, args.temperature,
              args.lr, WARMUP_FRAC,
              args.max_length_query, args.max_length_answer, args.seed, args.device,
              args.compute_dtype,
              args.checkpoint_every, args.log_every, args.eval_every, args.name, args.require_gpu,
              args.gradient_checkpointing, reference=args.reference)
    print(f"\n{r['name']}: loss {r['final_loss']:.4f} after {r['final_step']:,} steps, "
          f"{r['epochs_equivalent']:.2f} epochs")
    if r.get("final_holdout_recall_at_1") is not None:
        print(f"  held-out within-call recall@1 {r['final_holdout_recall_at_1']:.4f}")
        print("  NOT comparable to bge-small's 0.5378. That number is unfiltered and comes from the "
              "benchmark's\n  own candidate pool, every executive turn in the call over the length "
              "floor, while this is\n  held-out calls scored against the sampled queries' gold "
              "answers. For the comparison run\n  `bow_vs_dense --holdout-only` on both models, "
              "which builds one pool for all scorers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
