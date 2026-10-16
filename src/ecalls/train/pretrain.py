"""Masked language modeling for one vintage, on a fixed token budget, survivable under preemption.

Three properties matter beyond "it trains".

**The budget is in tokens, identical for every vintage.** Section 5 item 2 forbids training a later
vintage longer because it had more text, since that confounds data volume with training length. So every
vintage sees the same token count and the later ones see a smaller fraction of their own corpus: at a 1.2 B
budget the 2010 vintage makes four passes over its 0.303 B tokens while the 2024 vintage covers about a
third of its corpus once.

**Checkpointing is a first-class feature, not an afterthought.** These runs sit on `scavenger-gpu`, where a
task can be preempted at any moment, and a training run that restarts from step zero on preemption will
never finish. The checkpoint carries the model, the optimizer, the scheduler, the step counter and the
sampler's random state, because resuming with a fresh random state would re-show the same sequences and
silently change the effective budget.

**Masking is dynamic.** The mask is drawn fresh for every batch rather than fixed once at preprocessing,
which is one of the clearest reversals of the original BERT recipe. The ratio is 30% per MosaicBERT rather
than the original 15%.

**A chained segment is the one exception to the fixed budget (D29).** It starts from the finished model
of the previous year end, trains for one pass over that year's new calls only, and anneals its own
schedule, after He, Lv, Manela and Wu's ChronoBERT. Its budget is set by the increment because the
chained family exists to measure what the training history adds, and the from-scratch vintages remain
the fixed-budget comparison.
"""
import argparse
import json
import os
import math
import time
from pathlib import Path

import numpy as np

from ecalls.config import paths as P
from ecalls.ingest import manifest as M
from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.train import model as MD
from ecalls.train import pretrain_data as PD
from ecalls.train import tokenizer as TK

MASK_RATIO = 0.30        # MosaicBERT, against the original BERT's 0.15
REPLACE_MASK = 0.80      # of the chosen positions, the share replaced by [MASK]
REPLACE_RANDOM = 0.10    # replaced by a random token; the remaining 10% are left unchanged
MODELS_DIR = P.PROCESSED_DIR / "models"


# The real hyperparameter is tokens per step, which section 5 requires identical across vintages because
# it is what the gradient is computed over. The microbatch is not a hyperparameter, it is how many
# sequences fit on whichever card scavenger allocated, and accumulation makes up the difference. Holding
# tokens per step fixed while varying the microbatch gives the same result on a 10 GB card and a 49 GB one.
TOKENS_PER_STEP = 98_304          # 32 sequences of 768 at the size the probe ran
SEQ_PER_GB = 0.75                 # measured: batch 32 at context 768 OOMs an 11 GB card and fits 49 GB
MAX_MICROBATCH = 64


def auto_microbatch(device: str, context: int, target_tokens: int = TOKENS_PER_STEP) -> int:
    """Sequences per forward pass, scaled to the allocated device.

    **Measured the hard way:** twelve of fifteen pretraining tasks died within thirty seconds on 10.58 GiB
    cards at a microbatch of 32, while the throughput probe had run that batch comfortably on a 49 GiB RTX
    6000 Ada. Training holds activations for the backward pass and optimizer state, so its memory demand is
    several times inference's, and the inference budget from `embed/encode.py` does not transfer.

    Returns a power-of-two microbatch that divides the token target evenly, so accumulation lands on a
    whole number and the effective batch is exactly the same on every card.
    """
    from ecalls.embed.encode import gpu_memory_gb

    gb = gpu_memory_gb(device)
    cap = target_tokens // context
    if not gb:
        return min(8, cap)                       # CPU or MPS: small, since neither is used for a real run
    want = max(1, min(int(gb * SEQ_PER_GB), MAX_MICROBATCH, cap))
    mb = 1
    while mb * 2 <= want:
        mb *= 2
    return mb


DEFAULT_TOKEN_BUDGET = 1_200_000_000     # D21, identical across the from-scratch vintages
CHAIN_STAGE = "chain-mlm"
# The stages a chained segment may continue from. A dapt run has ChronoBERT's architecture and
# vocabulary, so its weights cannot load into ours, and it is refused by name rather than by the
# strict load failing later with a less useful message.
CHAIN_PARENT_STAGES = ("mlm", CHAIN_STAGE)
# The compute-matched control for the chain (D31): chained vintage Y continued on its own last year for
# as many tokens as segment Y+1 read, so it matches Y+1's compute and reads no later text.
REPLAY_STAGE = "rpl-mlm"


def model_dir(name: str) -> Path:
    return MODELS_DIR / name


def run_name(cutoff: str, tag: str = "mlm") -> str:
    return f"ecalls-{tag}-{cutoff.replace('-', '')}"


def family_tag(token_budget: int | None = None, backbone: str | None = None,
               init_from: str | None = None, replay: bool = False) -> str | None:
    """The name tag of a run: `mlm`, `dapt`, `chain-mlm`, or `mlm2x` for twice the standard budget.

    The budget enters the name because the run directory is also the resume point. A 2.4 B run
    named `ecalls-mlm-20241231` would find that vintage's finished checkpoint, resume it at step
    12,207 and train on to 24,414, overwriting the deliverable. Returns None for a budget that is
    not a whole multiple of the standard one, and the caller must then supply a name.
    """
    if replay:
        return REPLAY_STAGE
    if init_from:
        return CHAIN_STAGE
    if backbone:
        return "dapt"
    budget = DEFAULT_TOKEN_BUDGET if token_budget is None else token_budget
    if budget == DEFAULT_TOKEN_BUDGET:
        return "mlm"
    if budget % DEFAULT_TOKEN_BUDGET == 0:
        return f"mlm{budget // DEFAULT_TOKEN_BUDGET}x"
    return None


def previous_year_end(cutoff: str) -> str:
    """The cutoff a chained segment must continue from: 31 December of the year before its own."""
    import datetime as dt
    return dt.date(dt.date.fromisoformat(cutoff).year - 1, 12, 31).isoformat()


def require_finished(name: str) -> dict:
    """The run record of a finished pretraining run, or an error saying why it is not finished.

    The trainer writes `_run.json` only after its last step, so a missing record means unfinished. A
    record whose last step falls short of its total is refused as well, since a preempted run leaves
    a valid checkpoint part way through its schedule, before the learning rate has annealed.
    """
    f = model_dir(name) / "_run.json"
    if not f.exists():
        raise FileNotFoundError(
            f"{name} has no _run.json, so its pretraining has not finished. The trainer writes it "
            "only after the last step.")
    run = json.loads(f.read_text())
    if run.get("final_step") is None or run.get("final_step") != run.get("total_steps"):
        raise ValueError(
            f"{name} stopped at step {run.get('final_step')} of {run.get('total_steps')}, so its "
            "schedule has not annealed and anything continued from it starts mid-run.")
    return run


def chain_parent(init_from: str, cutoff: str) -> dict:
    """Check that `init_from` may start the chained segment ending at `cutoff`, and return its record.

    The point-in-time contract for a chain is that segment t starts from exactly the model of t-1. A
    parent at or after the segment's own cutoff would carry later text into an earlier model, and a
    parent further back would leave a year of text unseen while the record claimed a continuous
    history. Only the previous year end passes.
    """
    run = require_finished(init_from)
    if run.get("stage", "mlm") not in CHAIN_PARENT_STAGES:
        raise ValueError(
            f"{init_from} is stage {run.get('stage')!r}. A chained segment continues one of our own "
            f"masked language models, stage {' or '.join(CHAIN_PARENT_STAGES)}.")
    parent_cutoff = run.get("cutoff")
    if not parent_cutoff:
        raise ValueError(f"{init_from} records no cutoff, so the segment cannot place it in time")
    if parent_cutoff >= cutoff:
        raise ValueError(
            f"{init_from} has cutoff {parent_cutoff}, at or after this segment's {cutoff}. Continuing "
            "from it would carry later text into an earlier vintage.")
    want = previous_year_end(cutoff)
    if parent_cutoff != want:
        raise ValueError(
            f"{init_from} has cutoff {parent_cutoff}, but the segment ending {cutoff} must continue "
            f"from the model of {want}. Any other parent leaves a gap in the chain's history.")
    return run


def replay_target(replay_of: str, init_from: str) -> dict:
    """The record of the chained segment a replay control matches, checked (D31).

    The control continues the same parent as that segment and takes its token budget, so the segment
    must be finished, must be a chained segment, and must have started from `init_from`.
    """
    run = require_finished(replay_of)
    if run.get("stage") != CHAIN_STAGE:
        raise ValueError(f"{replay_of} is stage {run.get('stage')!r}. A replay control matches a "
                         f"chained segment, stage {CHAIN_STAGE!r}.")
    if run.get("init_from") != init_from:
        raise ValueError(
            f"{replay_of} continued {run.get('init_from')!r}, and this control continues "
            f"{init_from!r}. The two must start from the same weights to differ only in their text.")
    return run


def check_parent_matches(parent: dict, window: str, wmeta: dict, tokenizer_name: str,
                         replay: bool = False) -> None:
    """Refuse a parent trained on another packed corpus, vocabulary, context or held-out split.

    A different tokenizer gives every token id another meaning. A different held-out split is the
    quieter failure: the parent could have trained on calls this segment's split holds out, and the
    benchmark draws its queries from those calls. The parent must also end on the sequence just
    before this segment's first, so the chain's text has no gap and no overlap.
    """
    if parent.get("window") != window:
        raise ValueError(
            f"{parent.get('name')} trained on {parent.get('window')!r} and this segment reads "
            f"{window!r}. Sequence indices mean different text in two packed corpora.")
    if parent.get("stage", "mlm") == CHAIN_STAGE:
        parent_end = parent.get("last_sequence", -2) + 1
    else:
        parent_end = parent.get("n_sequences_available")
    if replay:
        # A replay rereads the parent's own last year, so its slice must end where the parent ends.
        # Ending later would read text after the parent's cutoff, which is the leak it controls for.
        if parent_end != wmeta["last_sequence"] + 1:
            raise ValueError(
                f"{parent.get('name')} ends before sequence {parent_end} but this replay's slice ends "
                f"before {wmeta['last_sequence'] + 1}. A replay reads only text its parent had read.")
    elif parent_end != wmeta["first_sequence"]:
        raise ValueError(
            f"{parent.get('name')} ends before sequence {parent_end} but this segment starts at "
            f"{wmeta['first_sequence']}. The chain would skip or repeat text.")
    checks = {"tokenizer": tokenizer_name, "vocab_size": wmeta["vocab_size"],
              "context": wmeta["context"], "holdout_salt": wmeta.get("holdout_salt"),
              "holdout_fraction": wmeta.get("holdout_fraction")}
    for k, v in checks.items():
        if parent.get(k) != v:
            raise ValueError(
                f"{parent.get('name')} has {k}={parent.get(k)!r} but this segment has {v!r}. A chain "
                "holds the tokenizer, context and held-out split fixed from its base onward.")


def load_parent_weights(model, parent: str) -> None:
    """Load the parent's final weights strictly, so a partial load raises rather than leaving noise."""
    import torch

    ck = torch.load(model_dir(parent) / "checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)


# Fields that decide what a run trains on. A directory resumed under different values would continue
# one experiment as another, which is how a 2x budget run could extend a finished 1x vintage.
RESUME_KEYS = ("stage", "token_budget", "init_from", "first_sequence", "last_sequence", "replay_of")


def check_resume(d: Path, meta: dict) -> None:
    f = d / "_progress.json"
    if not f.exists():
        return
    prior = json.loads(f.read_text())
    for k in RESUME_KEYS:
        if k in prior and prior[k] != meta.get(k):
            raise ValueError(
                f"{d.name} holds a run with {k}={prior[k]!r}, and this one asks for "
                f"{meta.get(k)!r}. Resuming would continue a different experiment. Use another --name.")


def load_mlm_backbone(model_id: str, device):
    """A registered encoder with its masked language modeling head, for continued pretraining.

    `AutoModelForMaskedLM` rather than `AutoModel`, because this stage trains the head. Dropping it
    and letting transformers initialise a fresh one would throw away the part of the model that
    knows the vocabulary, which is most of what continued pretraining is meant to build on.
    """
    from transformers import AutoModelForMaskedLM

    from ecalls.chunk import registry as R

    entry = R.load().by_name(model_id)
    if entry.kind != R.VINTAGE:
        raise ValueError(
            f"{model_id} is registered as {entry.kind}, so it has no known training cutoff and "
            "anything continued from it could not be registered as a vintage either.")
    if not entry.source.startswith("huggingface:"):
        raise ValueError(f"{model_id}: expected a huggingface source, got {entry.source!r}")
    m = AutoModelForMaskedLM.from_pretrained(entry.source.removeprefix("huggingface:")).to(device)
    c = m.config
    arch = {"hidden": c.hidden_size, "layers": c.num_hidden_layers,
            "heads": c.num_attention_heads, "intermediate": c.intermediate_size,
            "context": c.max_position_embeddings, "vocab": c.vocab_size,
            "total": sum(p.numel() for p in m.parameters())}
    return m, entry.training_cutoff, arch


class Sampler:
    """Draws sequence indices without replacement within an epoch, reproducibly.

    Without replacement inside an epoch so the budget means what it says: with replacement, a 4 epoch
    budget would show roughly 2% of sequences never and others four times over. The generator's state is
    part of the checkpoint for the same reason.
    """

    def __init__(self, n: int, seed: int = 0):
        self.n = n
        self.seed = seed
        self.epoch = 0
        self.pos = 0
        self._order = None
        self._reshuffle()

    def _reshuffle(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self._order = rng.permutation(self.n)
        self.pos = 0

    def next_batch(self, size: int) -> np.ndarray:
        out = []
        while len(out) < size:
            take = min(size - len(out), self.n - self.pos)
            out.extend(self._order[self.pos:self.pos + take].tolist())
            self.pos += take
            if self.pos >= self.n:
                self.epoch += 1
                self._reshuffle()
        return np.asarray(out)

    def state(self) -> dict:
        return {"n": self.n, "seed": self.seed, "epoch": self.epoch, "pos": self.pos}

    def load_state(self, s: dict) -> None:
        self.seed, self.epoch = s["seed"], s["epoch"]
        self._reshuffle()
        self.pos = s["pos"]


def mask_batch(ids, mask_id: int, vocab_size: int, special_ids, generator,
               ratio: float = MASK_RATIO):
    """Dynamic masking, returning the masked input and the labels.

    Special tokens are never masked. Predicting a `[CLS]` the model put there itself teaches nothing, and
    masking `[PAD]` would ask it to predict padding.
    """
    import torch

    labels = ids.clone()
    special = torch.zeros_like(ids, dtype=torch.bool)
    for sid in special_ids:
        special |= ids == sid

    prob = torch.full(ids.shape, ratio, device=ids.device)
    prob.masked_fill_(special, 0.0)
    chosen = torch.bernoulli(prob, generator=generator).bool()
    labels[~chosen] = -100        # the loss ignores these positions

    draw = torch.rand(ids.shape, device=ids.device, generator=generator)
    to_mask = chosen & (draw < REPLACE_MASK)
    to_random = chosen & (draw >= REPLACE_MASK) & (draw < REPLACE_MASK + REPLACE_RANDOM)
    out = ids.clone()
    out[to_mask] = mask_id
    out[to_random] = torch.randint(vocab_size, (int(to_random.sum()),), device=ids.device,
                                   generator=generator, dtype=out.dtype)
    return out, labels


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warmup then cosine decay to a tenth of peak.

    The schedule is part of the fixed budget: a checkpoint taken mid-schedule has not annealed, which is
    the reason D11 rejects mid-run checkpoints as vintages.
    """
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


def save_checkpoint(d: Path, model, opt, sampler, step: int, tokens: int, meta: dict) -> None:
    import torch

    d.mkdir(parents=True, exist_ok=True)
    # A per-writer temp name, for the reason `manifest.atomic_write_table` documents: a fixed
    # `.tmp` is safe for one writer and destructive for two. Two concurrent ablation arrays proved
    # it here, one dying with its temp file renamed out from under it at step 3,800 of 4,000.
    tmp = M.temp_sibling(d / "checkpoint.pt")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "sampler": sampler.state(), "step": step, "tokens": tokens,
                "torch_rng": torch.get_rng_state()}, tmp)
    os.replace(tmp, d / "checkpoint.pt")
    M.refresh_claim(d)          # a live run must not look abandoned to the next claimant
    M.atomic_write_json({**meta, "step": step, "tokens_seen": tokens, "updated_at": utc_now()},
                        d / "_progress.json")


def load_checkpoint(d: Path, model, opt, sampler, device):
    import torch

    f = d / "checkpoint.pt"
    if not f.exists():
        return 0, 0
    ck = torch.load(f, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    sampler.load_state(ck["sampler"])
    torch.set_rng_state(ck["torch_rng"].cpu() if hasattr(ck["torch_rng"], "cpu")
                        else ck["torch_rng"])
    return int(ck["step"]), int(ck["tokens"])


def check_borrowed_tokenizer(tokenizer_name: str, vintage_cutoff: str, backbone: str | None,
                             lookup=None, fingerprint_of=None) -> dict:
    """Refuse a borrowed vocabulary registered after the vintage, unless it is the backbone's own.

    The packer checks a borrowed tokenizer's cutoff against the cutoff it packs at, but one packed
    corpus serves every vintage as a prefix, so a corpus packed at 2024 with ChronoBERT's 2024
    vocabulary reached the 2010 and 2017 DAPT runs without any check (audit of 2026-10-13). A later
    vocabulary could carry later text. It is allowed only when its serialization is identical to the
    vocabulary of the backbone, whose cutoff `train` already requires to equal the vintage's, so every
    text gets the ids it would have had at the time. Returns the fields for the run record.
    """
    if not tokenizer_name.startswith(TK.REGISTRY_PREFIX):
        return {}
    if lookup is None:
        from ecalls.chunk import registry as R
        lookup = R.load().by_name
    if fingerprint_of is None:
        fingerprint_of = lambda name: TK.fingerprint(TK.load_any(name))  # noqa: E731
    entry = lookup(tokenizer_name.removeprefix(TK.REGISTRY_PREFIX))
    tok_cutoff = str(entry.training_cutoff) if entry.training_cutoff else None
    out = {"tokenizer_cutoff": tok_cutoff, "tokenizer_equivalent_to": None}
    if tok_cutoff is None or tok_cutoff <= vintage_cutoff:
        return out
    own = f"{TK.REGISTRY_PREFIX}{backbone}" if backbone else None
    if own is None or fingerprint_of(tokenizer_name) != fingerprint_of(own):
        raise ValueError(
            f"{tokenizer_name} is registered with cutoff {tok_cutoff}, after this vintage's "
            f"{vintage_cutoff}, and its vocabulary differs from that of the backbone {backbone}. A "
            "later vocabulary can carry later text into an earlier vintage. Pack with "
            f"--tokenizer registry:{backbone}.")
    out["tokenizer_equivalent_to"] = own
    return out


def train(window: str, tokenizer_name: str, cutoff: str | None = None,
          backbone: str | None = None, token_budget: int | None = None,
          batch_sequences: int | None = None, accum: int | None = None, peak_lr: float = 6e-4,
          tokens_per_step: int = TOKENS_PER_STEP,
          warmup_frac: float = 0.06, seed: int = 0, device: str | None = None,
          checkpoint_every: int = 200, log_every: int = 25, name: str | None = None,
          max_steps: int | None = None, compute_dtype: str = "bfloat16",
          require_gpu: bool = False, init_from: str | None = None,
          replay_of: str | None = None) -> dict:
    """Pretrain one vintage. Idempotent under preemption: rerunning resumes from the checkpoint.

    `init_from` makes the run a chained segment (D29): it starts from that model's final weights with
    a fresh optimizer, trains once over the sequences between the parent's cutoff and `cutoff`, and
    takes its budget from that increment rather than from `token_budget`.

    `replay_of` makes it the compute-matched control of D31 instead: it starts from `init_from` as
    the chained segment `replay_of` did, takes that segment's token budget, and trains on the parent's
    own last year, so its cutoff stays the parent's.
    """
    import torch

    from ecalls.embed.encode import resolve_device

    parent_run = None
    if replay_of is not None:
        if init_from is None or backbone is not None or token_budget is not None:
            raise ValueError("a replay control needs --init-from and takes its budget from "
                             "--replay-of, so --backbone and --token-budget do not apply")
        parent_run = require_finished(init_from)
        if parent_run.get("stage", "mlm") not in CHAIN_PARENT_STAGES:
            raise ValueError(f"{init_from} is stage {parent_run.get('stage')!r}, not a model of ours "
                             f"a chain continues ({' or '.join(CHAIN_PARENT_STAGES)})")
        if cutoff != parent_run.get("cutoff"):
            raise ValueError(
                f"a replay control keeps its parent's cutoff {parent_run.get('cutoff')}, got {cutoff}. "
                "It reads no text after it.")
        target = replay_target(replay_of, init_from)
        arr, wmeta = PD.load_increment(window, previous_year_end(cutoff), cutoff)
        check_parent_matches(parent_run, window, wmeta, tokenizer_name, replay=True)
        token_budget = target["token_budget"]
    elif init_from is not None:
        if backbone is not None:
            raise ValueError("a chained segment continues our own model, so pass init_from or "
                             "backbone, not both")
        if cutoff is None:
            raise ValueError("a chained segment needs --cutoff, the end of its increment")
        if token_budget is not None:
            raise ValueError(
                "a chained segment's budget is one pass over its increment, so --token-budget "
                "does not apply to it")
        parent_run = chain_parent(init_from, cutoff)
        arr, wmeta = PD.load_increment(window, parent_run["cutoff"], cutoff)
        check_parent_matches(parent_run, window, wmeta, tokenizer_name)
        token_budget = len(arr) * wmeta["context"]
    else:
        arr, wmeta = PD.load_window(window, cutoff=cutoff)
        token_budget = DEFAULT_TOKEN_BUDGET if token_budget is None else token_budget
    if wmeta.get("holdout_only"):
        raise ValueError(
            f"{window} is a held-out pack. Training on it would train on the calls every vintage is "
            "evaluated on.")
    # One packed corpus serves every vintage as a calendar-order prefix, so the file's own cutoff is
    # 2024-12-31 whichever vintage is training. What a vintage may not see past is the slice cutoff.
    # Recording the file's cutoff here labelled all fifteen run records 2024-12-31, which is the one
    # field a point-in-time model cannot afford to get wrong.
    window_cutoff = wmeta["cutoff"]
    vintage_cutoff = wmeta.get("sliced_to_cutoff") or window_cutoff
    tok = TK.load_any(tokenizer_name)
    tok_record = check_borrowed_tokenizer(tokenizer_name, vintage_cutoff, backbone)
    vocab = wmeta["vocab_size"]
    context = wmeta["context"]
    device = resolve_device(device, require_gpu=require_gpu)
    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[compute_dtype]
    if device == "cpu":
        dt = torch.float32

    # The microbatch adapts to the card and accumulation compensates, so tokens per step, which is the
    # hyperparameter, is identical everywhere. Twelve of fifteen tasks died on 10.58 GiB cards at a fixed
    # microbatch of 32 that the probe had run comfortably on 49 GiB.
    if batch_sequences is None:
        batch_sequences = auto_microbatch(device, context, tokens_per_step)
    if accum is None:
        accum = max(1, tokens_per_step // (batch_sequences * context))

    family = family_tag(token_budget, backbone, init_from, replay=replay_of is not None)
    stage = REPLAY_STAGE if replay_of else (CHAIN_STAGE if init_from else ("dapt" if backbone else "mlm"))
    if name is None:
        if family is None:
            raise ValueError(
                f"a budget of {token_budget:,} tokens is not a whole multiple of the standard "
                f"{DEFAULT_TOKEN_BUDGET:,}, so it has no default name. Pass --name.")
        name = run_name(vintage_cutoff, family)
    d = model_dir(name)
    # Checked before the claim, so a refused resume leaves no lock on the directory it refused.
    check_resume(d, {"stage": stage, "replay_of": replay_of,
                     "token_budget": token_budget, "init_from": init_from,
                     "first_sequence": wmeta.get("first_sequence", 0),
                     "last_sequence": wmeta.get("last_sequence", len(arr) - 1)})
    M.claim_dir(d, label=f"mlm {name}")   # one trainer per model directory, see manifest.claim_dir
    if init_from is not None:
        model = MD.build_mlm(vocab, seed=seed).to(device)
        load_parent_weights(model, init_from)
        arch = parent_run.get("architecture") or MD.describe(vocab)
        backbone_cutoff = None
    elif backbone is None:
        model = MD.build_mlm(vocab, seed=seed).to(device)
        arch = MD.describe(vocab)
        backbone_cutoff = None
    else:
        # Domain-adaptive pretraining: continue masked language modeling from a registered model
        # rather than starting from noise. Gururangan et al. (2020), doi
        # 10.18653/v1/2020.acl-main.740. The ablation showed a general backbone given our
        # contrastive stage beats our own finance backbone, but at 4.4x the parameters and about
        # 380x the pretraining tokens, so that comparison cannot separate domain from scale. This
        # one can: same weights, same capacity, same contrastive stage afterwards, and the only
        # difference is whether the finance corpus was seen.
        model, backbone_cutoff, arch = load_mlm_backbone(backbone, device)
        if tok.get_vocab_size() != int(model.config.vocab_size):
            raise ValueError(
                f"the packed corpus was tokenized with a {tok.get_vocab_size():,} token "
                f"vocabulary but {backbone} expects {model.config.vocab_size:,}. Continuing from\n"
                f"            these weights would give every token id a different meaning than it "
                f"had during pretraining. Repack with --tokenizer registry:{backbone}.")
        if backbone_cutoff and backbone_cutoff != vintage_cutoff:
            raise ValueError(
                f"{backbone} was pretrained to {backbone_cutoff} but this run is the "
                f"{vintage_cutoff} vintage. Continued pretraining inherits the backbone's cutoff, "
                "so the two must agree.")
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6,
                            weight_decay=0.01)
    # float16 needs a gradient scaler and bfloat16 does not, because bf16 carries fp32's exponent
    # range while fp16's smallest normal is about 6e-5 and MLM gradients go below it. Without this a
    # float16 run underflows to zero gradient and still looks like it is training, which is the
    # quietest way to waste a day on a 149 M model. Tied to the dtype rather than offered as a flag,
    # so the pair cannot be got wrong.
    scaler = torch.amp.GradScaler(device.split(":")[0]) if dt is torch.float16 else None
    sampler = Sampler(len(arr), seed=seed)

    tokens_per_step = batch_sequences * accum * context
    total_steps = max_steps or max(1, token_budget // tokens_per_step)
    warmup = int(total_steps * warmup_frac)

    meta = {
        "name": name, "window": window, "cutoff": vintage_cutoff,
        "window_cutoff": window_cutoff,
        "tokenizer": tokenizer_name, **tok_record, "vocab_size": vocab, "context": context,
        "token_budget": token_budget, "total_steps": total_steps,
        "tokens_per_step": tokens_per_step, "batch_sequences": batch_sequences,
        "grad_accum": accum, "peak_lr": peak_lr, "warmup_steps": warmup, "seed": seed,
        "mask_ratio": MASK_RATIO, "compute_dtype": compute_dtype, "device": device,
        "grad_scaler": scaler is not None,
        "architecture": arch, "backbone": backbone, "backbone_cutoff": backbone_cutoff,
        "stage": stage, "family": family, "replay_of": replay_of,
        # The chain's lineage. `start_cutoff` is the parent's cutoff, and the two sequence indices
        # are absolute positions in the packed corpus, inclusive at both ends. A from-scratch run
        # reads a prefix, so its first sequence is 0 and it has no parent.
        "init_from": init_from, "init_from_cutoff": parent_run["cutoff"] if parent_run else None,
        "start_cutoff": wmeta.get("sliced_from_cutoff"),
        "first_sequence": wmeta.get("first_sequence", 0),
        "last_sequence": wmeta.get("last_sequence", len(arr) - 1),
        "holdout_salt": wmeta.get("holdout_salt"), "holdout_fraction": wmeta.get("holdout_fraction"),
        "n_sequences_available": len(arr), "git_sha": git_sha(),
    }
    step, tokens = load_checkpoint(d, model, opt, sampler, device)
    if step:
        print(f"resuming {name} at step {step}/{total_steps}, {tokens/1e9:.3f} B tokens seen",
              flush=True)
    else:
        MD.save_config(d, vocab, {k: meta[k] for k in
                                  ("window", "cutoff", "window_cutoff", "tokenizer",
                                   "token_budget", "total_steps", "init_from", "replay_of", "start_cutoff",
                                   "first_sequence", "last_sequence")},
                       seed=seed)
        print(f"{name}: {total_steps:,} steps of {tokens_per_step:,} tokens on {device}, "
              f"{meta['architecture']['total']/1e6:.1f} M params", flush=True)

    mask_id = tok.token_to_id("[MASK]")
    specials = [tok.token_to_id(s) for s in TK.SPECIALS]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + step)          # advanced by step so a resume does not repeat masks
    model.train()
    t0, hist = time.time(), []

    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        lr = lr_at(step, total_steps, peak_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        done_micro = 0
        while done_micro < accum:
            idx = np.sort(sampler.next_batch(batch_sequences))
            batch = torch.from_numpy(np.asarray(arr[idx], dtype=np.int64)).to(device)
            inp, labels = mask_batch(batch, mask_id, vocab, specials, gen)
            try:
                with torch.autocast(device_type=device.split(":")[0], dtype=dt,
                                    enabled=(device != "cpu")):
                    out = model(input_ids=inp, labels=labels)
                step_loss = out.loss / accum
                (scaler.scale(step_loss) if scaler is not None else step_loss).backward()
            except torch.OutOfMemoryError:
                # Halve the microbatch and double accumulation, which keeps tokens per step and so the
                # gradient identical. The calibration is one measurement on two card sizes, so it will be
                # wrong somewhere, and this makes being wrong cost a step instead of the run.
                if batch_sequences == 1:
                    raise
                del batch, inp, labels
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                batch_sequences //= 2
                accum *= 2
                opt.zero_grad(set_to_none=True)
                loss_sum, done_micro = 0.0, 0
                print(f"    OOM: microbatch now {batch_sequences}, accum {accum}, "
                      f"tokens/step unchanged at {batch_sequences * accum * context:,}", flush=True)
                continue
            loss_sum += float(out.loss.detach()) / accum
            done_micro += 1
        if scaler is not None:
            scaler.unscale_(opt)            # clip real gradients, not scaled ones
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        step += 1
        tokens += tokens_per_step
        hist.append(loss_sum)

        if step % log_every == 0:
            el = time.time() - t0
            tps = tokens_per_step * log_every / max(el, 1e-9)
            print(f"  step {step:,}/{total_steps:,}  loss {np.mean(hist[-log_every:]):.4f}  "
                  f"lr {lr:.2e}  {tps/1e3:.1f} k tok/s  "
                  f"eta {(total_steps-step)*tokens_per_step/max(tps,1)/3600:.2f} h", flush=True)
            t0 = time.time()
        if step % checkpoint_every == 0 or step == total_steps:
            save_checkpoint(d, model, opt, sampler, step, tokens, meta)

    final = {**meta, "final_step": step, "tokens_seen": tokens,
             "final_loss": float(np.mean(hist[-50:])) if hist else None,
             "epochs_equivalent": tokens / max(wmeta["n_tokens"], 1),
             "finished_at": utc_now()}
    M.atomic_write_json(final, d / "_run.json")
    M.release_dir(d)
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", required=True, help="a packed corpus name from pretrain_data")
    ap.add_argument("--backbone", default=None,
                    help="continue masked language modeling from this registered model instead of "
                         "starting from scratch; the packed corpus must use its vocabulary")
    ap.add_argument("--cutoff", default=None,
                    help="train on the prefix of the packed corpus belonging to this vintage. One "
                         "packed corpus serves every vintage, since the array is in calendar order")
    ap.add_argument("--tokenizer", default="ecalls-bpe-32768-20101231")
    ap.add_argument("--token-budget", type=int, default=None,
                    help=f"tokens to train on, {DEFAULT_TOKEN_BUDGET:,} when omitted. A whole multiple "
                         "of the standard budget gets its own family name, ecalls-mlm2x-* for twice it")
    ap.add_argument("--init-from", default=None,
                    help="a chained segment (D29): continue from this finished model, whose cutoff "
                         "must be the previous year end, and train once over the text after it")
    ap.add_argument("--replay-of", default=None,
                    help="the compute-matched control of D31: continue --init-from on its own last "
                         "year for as many tokens as this chained segment read, keeping its cutoff")
    ap.add_argument("--batch-sequences", type=int, default=None,
                    help="sequences per forward pass. Omit it to scale from the allocated GPU's memory, "
                         "which is the right default since accumulation keeps tokens per step fixed")
    ap.add_argument("--accum", type=int, default=None,
                    help="gradient accumulation steps. Derived from --tokens-per-step when omitted")
    ap.add_argument("--tokens-per-step", type=int, default=TOKENS_PER_STEP,
                    help="the real hyperparameter, identical across vintages and across cards")
    ap.add_argument("--peak-lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--compute-dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override the budget, for a throughput probe rather than a real run")
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    r = train(args.window, args.tokenizer, args.cutoff, args.backbone, args.token_budget,
              args.batch_sequences, args.accum, args.peak_lr, args.tokens_per_step,
              seed=args.seed, device=args.device, compute_dtype=args.compute_dtype,
              max_steps=args.max_steps, checkpoint_every=args.checkpoint_every,
              log_every=args.log_every, name=args.name, require_gpu=args.require_gpu,
              init_from=args.init_from, replay_of=args.replay_of)
    print(f"\n{r['name']}: {r['final_step']:,} steps, {r['tokens_seen']/1e9:.3f} B tokens "
          f"({r['epochs_equivalent']:.2f} epochs), final loss {r['final_loss']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
