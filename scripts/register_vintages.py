"""Register the trained retriever vintages so the rest of the pipeline can resolve them by date.

This is the step that makes the fifteen models part of the point-in-time contract rather than fifteen
directories. Once registered, `registry.for_date(d)` returns the newest vintage whose cutoff is at or
before `d`, `index.check_point_in_time` refuses a query whose model is too new, and `embed/encode.py`
loads the exported weights through a `local:` source. None of that works from a path.

Every field is read from the model's own `_run.json` rather than passed in, so a mismatch between
what was trained and what is registered cannot be introduced here. The cutoff most of all: it was
already wrong once, when the trainer recorded the packed corpus's cutoff instead of the vintage's.

Run on the cluster, where the models are:
    PYTHONPATH=src python3 scripts/register_vintages.py [--apply] [--stage contrastive]
"""
import argparse
import json
import re
import sys

from ecalls.chunk import registry as R
from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.train import contrastive as CT


NOTES = {
    "ecalls-mlm": ("Finance vintage retriever trained on this corpus. Masked language modeling on the "
                   "{cutoff} window ({backbone}), then a contrastive stage on "),
    "registered": ("ABLATION, not a deliverable. The same contrastive stage from the general-domain "
                   "backbone {backbone}, which isolates how much of the gain over bge-small comes "
                   "from finance pretraining rather than from the contrastive stage. Trained on "),
    "dapt": ("DOMAIN-ADAPTIVE comparison, not a deliverable. ChronoBERT continued by masked language "
             "modeling on this corpus to {cutoff} ({backbone}), then the same contrastive stage as "
             "the ablation, which it differs from only in having seen the finance corpus. Trained on "),
    "chain-mlm": ("CHAINED vintage, a comparison for the training history (D29), not the headline. "
                  "Masked language modeling continued year by year from ecalls-mlm-20101231, one "
                  "pass over each year's new calls, to {cutoff} ({backbone}), then the same "
                  "contrastive stage on "),
    "mlm2x": ("BUDGET CHECK, not a deliverable. The {cutoff} vintage pretrained from scratch at twice "
              "the standard token budget ({backbone}), then the same contrastive stage on "),
    "reference": ("REFERENCE, never a vintage (D34). {backbone}, whose training cutoff is unknown, "
                  "fine-tuned by the same contrastive stage on "),
    "rpl-mlm": ("REPLAY CONTROL, not a deliverable (D31). The chained {cutoff} vintage continued on its "
                "own last year for as many tokens as the next chained segment read ({backbone}), so it "
                "matches that segment's compute with no later text, then the same contrastive stage on "),
}

# Every family the contrastive stage writes. Seed replicates carry the seed before the tag, so they
# form families of their own and never tie with the seed-0 models in `for_date` (see `run_name`).
# The chained family starts at 2011, since its 2010 member is `ecalls-ret-20101231` itself.
# The pretraining-seed replicates of the chain (D34) carry pseed<S>- in front of every name, and only
# their retrievers match, since their pretraining directories end in -mlm-.
PATTERNS = ("ecalls-ret-*", "ecalls-abl-*", "ecalls-dapt-ret-*", "ecalls-seed*-*",
            "ecalls-chain-ret-*", "ecalls-2x-ret-*", "ecalls-rpl-ret-*", "ecalls-pseed*-ret-*",
            "ref-*")


def pretrain_seed_note(name: str) -> str:
    """The note prefix of a pretraining-seed replicate. `run["seed"]` is the contrastive seed, which
    these runs leave at 0, so the pretraining seed is read from the name."""
    m = re.match(r"ecalls-pseed(\d+)-", name)
    return (f"PRETRAINING SEED {m.group(1)} REPLICATE of the chained family (D34), for the "
            "pretraining-seed check only. " if m else "")


def entry_for(d) -> R.Entry:
    run = json.loads((d / "_run.json").read_text())
    if run.get("stage") != "contrastive":
        raise ValueError(f"{d.name}: stage is {run.get('stage')!r}, not 'contrastive'. A model with "
                         "no contrastive stage is a weak retriever and must not be registered as one.")
    if not (d / "hf" / "config.json").exists():
        raise FileNotFoundError(f"{d.name}: no exported model under {d / 'hf'}")
    cfg = json.loads((d / "hf" / "config.json").read_text())
    # A reference model (D34) starts from a backbone with no known cutoff, so it is registered without
    # one: ATEMPORAL entries are never returned by `for_date` and fail `require_point_in_time`.
    reference = bool(run.get("reference"))
    if reference != run["name"].startswith("ref-"):
        raise ValueError(f"{d.name}: a reference model and only a reference model is named ref-*")
    return R.Entry(
        model_id=run["name"],
        kind=R.ATEMPORAL if reference else R.VINTAGE,
        source=f"local:{run['name']}",
        dim=int(cfg["hidden_size"]),
        training_cutoff=None if reference else run["cutoff"],
        pooling=run["pooling"],
        normalized=bool(run["normalized"]),
        notes=(
            pretrain_seed_note(run["name"])
            + (f"SEED {run.get('seed')} REPLICATE, for the training-seed variation only. " if run.get("seed") else "")
            + NOTES[run.get("backbone_kind", "ecalls-mlm")].format(**run)
            + (f"{run['n_pairs']:,} question-answer pairs over {run['n_calls']:,} calls at "
               f"temperature {run['temperature']}. Held-out calls excluded from both stages.")),
        registered_at=utc_now(),
        git_sha=git_sha(),
        extra={"stage": "contrastive", "backbone": run["backbone"],
               "backbone_kind": run.get("backbone_kind", "ecalls-mlm"),
               "max_seq_len": run["max_length"], "temperature": run["temperature"],
               "n_pairs": run["n_pairs"], "n_calls": run["n_calls"],
               "final_holdout_recall_at_1": run.get("final_holdout_recall_at_1"),
               "tokenizer": run["tokenizer"], "vocab_size": run["vocab_size"],
               "seed": run.get("seed", 0),
               # The parent and the slice of the corpus a chained backbone saw. Absent for a
               # record written before the field existed, which then describes a prefix run.
               **({"backbone_" + k: v for k, v in run["backbone_lineage"].items()}
                  if run.get("backbone_lineage") else {})},
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--replace", action="store_true", help="overwrite an existing entry of the same id")
    args = ap.parse_args(argv)

    # Both families. `ecalls-abl-*` is the ablation, the same contrastive stage from a foreign
    # backbone, and it is a legitimate vintage because it inherits that backbone's cutoff. Two
    # families now share every cutoff, which `for_date` refuses rather than resolving by sort order,
    # so registering them cannot silently shadow the deliverable.
    dirs = sorted({p for pat in PATTERNS
                   for p in CT.MODELS_DIR.glob(pat) if (p / "_run.json").exists()})
    if not dirs:
        sys.exit(f"no trained retriever under {CT.MODELS_DIR}")
    print(f"{len(dirs)} trained retriever(s) on disk\n")

    reg = R.load()
    added = 0
    for d in dirs:
        e = entry_for(d)
        try:
            existing = reg.by_name(e.model_id)
        except Exception:      # noqa: BLE001 -- not registered yet is the normal case
            existing = None
        if existing and not args.replace:
            print(f"{e.model_id}: already registered, cutoff {existing.training_cutoff}")
            continue
        print(f"{e.model_id}: {e.kind}, cutoff {e.training_cutoff}, dim {e.dim}, "
              f"pooling {e.pooling}, source {e.source}")
        reg = reg.register(e, replace=args.replace)
        added += 1

    if args.apply and added:
        path = reg.save()
        print(f"\nwrote {path} with {len(reg)} entries")
    else:
        print(f"\n{added} entry/entries would be added; re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
