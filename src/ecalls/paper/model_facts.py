"""Parameter counts and dimensions of one model per family, measured from the loaded encoders.

Table 2 states each family's size, and a size typed from a model card can be the size of a different
checkpoint or include a head we drop. This loads the encoder body exactly as the benchmark does
(`encode.load_encoder`) and counts its parameters. Run on the cluster, where the weights are:

    python -m ecalls.paper.model_facts
"""
import json

from ecalls.chunk import registry as R
from ecalls.config import paths as P
from ecalls.embed import encode as E
from ecalls.ingest.manifest import git_sha, utc_now

MODELS = ("bge-small-en-v1.5", "bge-base-en-v1.5", "bge-large-en-v1.5", "e5-base-v2", "finbert-pretrain",
          "chrono-bert-v1-20241231", "ecalls-abl-20241231", "ecalls-dapt-ret-20241231",
          "ecalls-ret-20241231", "ecalls-chain-ret-20241231")
OUT = P.REPORTS_DIR / "paper" / "model_facts.json"


def facts(model_id: str) -> dict:
    entry = R.load().by_name(model_id)
    tok, model, _ = E.load_encoder(entry, device="cpu", compute_dtype="float32")
    cfg = model.config
    return {"params": int(sum(p.numel() for p in model.parameters())), "dim": int(cfg.hidden_size),
            "layers": int(getattr(cfg, "num_hidden_layers", 0)), "vocab": int(cfg.vocab_size),
            "max_seq_len": int(E.effective_max_length(entry)), "kind": entry.kind,
            "cutoff": entry.training_cutoff}


def main() -> int:
    out = {m: facts(m) for m in MODELS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"models": out, "git_sha": git_sha(), "ts": utc_now()}, indent=2) + "\n")
    for m, f in out.items():
        print(f"{m:<28} {f['params']/1e6:7.1f}M  dim {f['dim']}  layers {f['layers']}  vocab {f['vocab']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
