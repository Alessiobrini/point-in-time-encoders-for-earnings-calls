"""Is each trained model's checkpoint the same weights as its exported copy?

Written because two identical ablation arrays, 57156367 and 57156375, ran concurrently over the same
fifteen model directories. `save_checkpoint` used a fixed temp name, `checkpoint.pt.tmp`, so two
writers could and did splice one file: one task died with that temp file already renamed out from
under it. The project had fixed exactly this once before, in the embedding stage, by putting the
host, the pid and a random token in the temp name.

A directory written twice can still look complete. Every `_run.json` reports 4,000 steps because
whichever process finished last wrote one. So completeness is not the question; agreement is.

`checkpoint.pt` and `hf/model.safetensors` are written from the same model object milliseconds apart
at the end of a run. Two processes resuming from a shared checkpoint diverge, because the call
sampler and the mask draws differ once their step counts do, so if one process wrote the checkpoint
and the other wrote the export, the two files disagree. Byte-level agreement across every parameter
is therefore evidence that one run produced both.

Run:
    PYTHONPATH=src python3 scripts/verify_model_integrity.py [--pattern 'ecalls-abl-*']
"""
import argparse
import json
import sys

from ecalls.train import contrastive as CT


def check(d, atol: float = 0.0) -> dict:
    import torch
    from transformers import AutoModel

    run = json.loads((d / "_run.json").read_text())
    ck = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
    exported = AutoModel.from_pretrained(d / "hf")

    ckw = ck["model"]
    exw = dict(exported.state_dict())
    only_ck = sorted(set(ckw) - set(exw))
    only_ex = sorted(set(exw) - set(ckw))
    shared = sorted(set(ckw) & set(exw))

    worst, worst_key = 0.0, None
    for k in shared:
        a, b = ckw[k].float(), exw[k].float()
        if a.shape != b.shape:
            return {"name": d.name, "ok": False, "reason": f"shape mismatch at {k}"}
        m = (a - b).abs().max().item()
        if m > worst:
            worst, worst_key = m, k

    ok = (not only_ck and not only_ex and worst <= atol)
    return {"name": d.name, "ok": ok, "step": run.get("final_step"),
            "ckpt_step": int(ck.get("step", -1)),
            "n_params_compared": len(shared), "only_in_checkpoint": only_ck[:3],
            "only_in_export": only_ex[:3], "worst_abs_diff": worst, "worst_key": worst_key,
            "backbone": run.get("backbone"), "cutoff": run.get("cutoff")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pattern", default="ecalls-abl-*")
    args = ap.parse_args(argv)

    dirs = sorted(p for p in CT.MODELS_DIR.glob(args.pattern)
                  if (p / "_run.json").exists() and (p / "hf" / "config.json").exists())
    if not dirs:
        sys.exit(f"nothing matching {args.pattern} under {CT.MODELS_DIR}")

    bad = []
    print("%-24s %6s %6s %8s %14s %s" % ("name", "run", "ckpt", "params", "worst diff", "verdict"))
    for d in dirs:
        r = check(d)
        if not r["ok"]:
            bad.append(r["name"])
        print("%-24s %6s %6s %8s %14.3e %s" % (
            r["name"], r.get("step"), r.get("ckpt_step"), r.get("n_params_compared"),
            r.get("worst_abs_diff", float("nan")), "OK" if r["ok"] else "MISMATCH"))
        if not r["ok"]:
            print("    reason:", r.get("reason") or
                  f"worst {r['worst_abs_diff']:.3e} at {r['worst_key']}, "
                  f"only_ck={r['only_in_checkpoint']} only_ex={r['only_in_export']}")

    print()
    if bad:
        print(f"{len(bad)} of {len(dirs)} disagree between checkpoint and export: {bad}")
        print("Those weights cannot be attributed to one run and must be retrained.")
        return 1
    print(f"all {len(dirs)} agree exactly: one run produced both files in each directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
