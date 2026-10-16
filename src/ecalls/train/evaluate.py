"""Masked language modeling loss for a trained vintage, on text it never saw.

This exists because the fifteen training losses are not comparable with each other. Each one is
measured on that vintage's own corpus, and a later vintage's corpus is larger and more varied, so a
higher loss there is what equal skill on harder text looks like as much as it is a less converged
model. The training numbers rise from 1.7763 at 2010 to 1.8153 at 2024 (L14), and reading that as a
convergence ranking assumes the thing it is supposed to establish.

One held-out set inside every vintage's window fixes that. Every model is scored on the same
sequences, so the remaining differences are differences between the models.

Two further sources of noise are removed rather than averaged away.

**The mask pattern is fixed and does not depend on the model.** `mask_seed` seeds a generator per
batch from the batch index alone, so vintage A and vintage B are asked to predict the same tokens in
the same positions. Drawing fresh masks per model would put a few thousandths of loss between two
identical models, which is the size of the effect being measured.

**Evaluation runs in float32 whatever the model trained in.** The differences here are in the third
decimal place and bfloat16 has about three decimal digits of mantissa, so the arithmetic would be
the same size as the signal.

Run:
    python -m ecalls.train.evaluate --window <packed-holdout-set>
    python -m ecalls.train.evaluate --window <packed-holdout-set> --model ecalls-mlm-20101231
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.train import model as MD
from ecalls.train import pretrain as PT
from ecalls.train import pretrain_data as PD
from ecalls.train import tokenizer as TK

MASK_SEED = 1234           # fixed, and deliberately not the training seed
EVAL_SEQUENCES = 20_000    # measured: the loss is stable to 1e-3 well below this, see the report
BATCH = 16


def load_trained(d: Path, vocab: int, device):
    """The MLM model with its trained weights, in eval mode.

    Reads `checkpoint.pt` rather than a saved `model` directory, because that is what the trainer
    writes and it is the only file guaranteed to exist for a preempted run.
    """
    import torch

    f = d / "checkpoint.pt"
    if not f.exists():
        raise FileNotFoundError(f"no checkpoint at {f}")
    m = MD.build_mlm(vocab)
    ck = torch.load(f, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), int(ck["step"]), int(ck["tokens"])


def eval_indices(n_available: int, n_sequences: int) -> np.ndarray:
    """Evenly spaced sequence indices spanning the whole set, not its first `n`.

    **A packed corpus is in calendar order**, so taking a prefix evaluates the oldest text rather
    than a sample of the set. That is not a performance detail. It silently made two evaluation sets
    give byte-identical losses across all fifteen vintages, because the 2010 held-out set is the
    leading slice of the full-span one and a prefix of each is the same text. The defect announced
    itself only because fifteen numbers matching to four decimals is impossible.

    Evenly spaced rather than random, so the sample covers the span uniformly and is reproducible
    without carrying a seed.
    """
    n = min(n_sequences, n_available)
    if n <= 0:
        raise ValueError(f"nothing to evaluate: {n_available} sequences available")
    if n == n_available:
        return np.arange(n_available)
    return np.linspace(0, n_available - 1, n).astype(np.int64)


def mlm_loss(model, arr, mask_id: int, vocab: int, special_ids, device,
             n_sequences: int = EVAL_SEQUENCES, batch: int = BATCH,
             mask_seed: int = MASK_SEED) -> dict:
    """Mean cross-entropy over masked positions, token-weighted rather than batch-weighted.

    Averaging the per-batch means would weight a batch with few masked positions equally with a full
    one. The masked count varies because special tokens are never chosen.
    """
    import torch

    idx = eval_indices(len(arr), n_sequences)
    total, counted = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(idx), batch):
            take = idx[start:start + batch]
            ids = torch.from_numpy(np.asarray(arr[take], dtype=np.int64)).to(device)
            # Seeded from the position in the iteration alone, so every model sees an identical mask
            # pattern over identical sequences.
            gen = torch.Generator(device=device)
            gen.manual_seed(mask_seed + start)
            x, labels = PT.mask_batch(ids, mask_id, vocab, special_ids, gen)
            out = model(input_ids=x, labels=labels)
            k = int((labels != -100).sum())
            total += float(out.loss) * k
            counted += k
    if not counted:
        raise ValueError("no position was masked, so the loss is undefined")
    return {"loss": total / counted, "n_sequences": int(len(idx)),
            "n_available": int(len(arr)), "n_masked_positions": counted,
            "first_index": int(idx[0]), "last_index": int(idx[-1]),
            "mask_seed": mask_seed}


def evaluate(window: str, models: list[str] | None = None, device: str | None = None,
             n_sequences: int = EVAL_SEQUENCES, batch: int = BATCH,
             require_gpu: bool = False) -> dict:
    """Score every named model, or every MLM vintage on disk, against one held-out set."""
    import torch

    from ecalls.embed.encode import resolve_device

    arr, wmeta = PD.load_window(window)
    if not wmeta.get("holdout_only"):
        raise ValueError(
            f"{window} is a training pack, not a held-out one. Scoring a model on the text it "
            f"trained on measures memorization. Build the evaluation set with "
            f"`python -m ecalls.train.pretrain_data --cutoff <date> --holdout-only`.")
    vocab = wmeta["vocab_size"]
    tok = TK.load(wmeta["tokenizer"])
    mask_id = tok.token_to_id("[MASK]")
    specials = [tok.token_to_id(s) for s in TK.SPECIALS]
    device = resolve_device(device, require_gpu=require_gpu)

    dirs = ([PT.model_dir(m) for m in models] if models
            else sorted(p for p in (PT.MODELS_DIR).glob("ecalls-mlm-*") if (p / "checkpoint.pt").exists()))
    if not dirs:
        raise SystemExit(f"no trained MLM vintage found under {PT.MODELS_DIR}")

    print(f"{window}: {len(arr):,} held-out sequences, scoring {min(n_sequences, len(arr)):,} "
          f"of them on {device}", flush=True)
    rows = []
    for d in dirs:
        run = json.loads((d / "_run.json").read_text()) if (d / "_run.json").exists() else {}
        m, step, tokens = load_trained(d, vocab, device)
        with torch.autocast(device_type=device, enabled=False):      # float32, see the docstring
            r = mlm_loss(m, arr, mask_id, vocab, specials, device, n_sequences, batch)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        rows.append({"model": d.name, "cutoff": run.get("cutoff"),
                     "train_loss": run.get("final_loss"), "step": step, "tokens_seen": tokens,
                     "n_sequences_trained_on": run.get("n_sequences_available"),
                     "epochs_equivalent": run.get("epochs_equivalent"), **r})
        print(f"  {d.name}  held-out {r['loss']:.4f}  train {run.get('final_loss', float('nan')):.4f}",
              flush=True)

    out = {"window": window, "window_cutoff": wmeta["cutoff"],
           "holdout_fraction": wmeta.get("holdout_fraction"),
           "holdout_salt": wmeta.get("holdout_salt"),
           "n_held_out_sequences": len(arr), "device": device,
           "models": rows, "git_sha": git_sha(), "ts": utc_now()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", required=True, help="a packed set built with --holdout-only")
    ap.add_argument("--model", action="append", default=None,
                    help="repeatable; default is every MLM vintage on disk")
    ap.add_argument("--n-sequences", type=int, default=EVAL_SEQUENCES)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--device", default=None)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    args = ap.parse_args(argv)

    r = evaluate(args.window, args.model, args.device, args.n_sequences, args.batch,
                 args.require_gpu)
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    rows = sorted(r["models"], key=lambda x: x["model"])
    print(f"\n{'model':<24}{'cutoff':<12}{'held-out':>10}{'train':>10}{'epochs':>9}")
    for x in rows:
        print(f"{x['model']:<24}{str(x['cutoff']):<12}{x['loss']:>10.4f}"
              f"{x['train_loss']:>10.4f}{x['epochs_equivalent']:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
