"""Embed the chunk layer with a registered model, one quarterly shard at a time.

Three properties this module is built around, each of which is a lesson the ingest stages paid for:

1. **The model comes from the registry, never from a path.** `--model` is a registered `model_id` and
   the run refuses an unknown one, so an embedding set cannot exist on disk without a recorded
   training cutoff. The whole point-in-time discipline rests on that record.
2. **Resume compares coverage, not existence.** A part file counts as done only when it holds an
   embedding for every chunk_id in its shard. Three separate silent losses in the ingest stages came
   from treating a file's existence as completeness, and an interrupted embedding run is exactly the
   same shape: the part file is there and short.
3. **The sidecar is written from the run, not from the plan.** `_meta.json` records the model, the
   dimension, the pooling, the dtype, the max_length, the device and the torch version actually used,
   so a later reader never has to infer how these vectors were made.

`max_length` defaults to 768, which is **measured** rather than conventional: over 18,000 sampled
chunks the longest was 710 ChronoBERT tokens, so 768 truncates nothing. The reflexive 512 for a
BERT-family encoder would truncate 18.2% of chunks and drop 2.9% of the corpus characters.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa

from ecalls.chunk import registry as R
from ecalls.config import paths as P
from ecalls.ingest import manifest as M

MAX_LENGTH = 768        # measured: nothing in the chunk layer exceeds 710 ChronoBERT tokens
DEFAULT_BATCH = 64
STORE_DTYPE = "float16"   # halves 37 GB to 19 GB at the full corpus; see DECISIONS D13
COMPUTE_DTYPE = "float16"  # measured 1.66x faster than float32 on MPS at a cosine cost under 1e-4
CHARS_PER_GB = 30_000      # padded chars per GB of device memory; 24 GB cards ran ~770k, 11 GB OOMed
MAX_CHAR_BUDGET = 1_200_000  # past here the batcher is not the bottleneck and the batches get unwieldy

_PA_DTYPE = {"float16": pa.float16(), "float32": pa.float32()}
_NP_DTYPE = {"float16": np.float16, "float32": np.float32}


# --- paths ---------------------------------------------------------------------------------------

def model_dir(model_id: str) -> Path:
    return P.EMBEDDINGS_DIR / model_id


def part_path(model_id: str, shard: str) -> Path:
    return model_dir(model_id) / f"part_{shard}.parquet"


def meta_path(model_id: str) -> Path:
    return model_dir(model_id) / "_meta.json"


def shard_end(shard: str) -> "dt.date":
    """The last day of a quarterly shard key such as "2015Q3"."""
    import datetime as dt
    year, q = int(shard[:4]), int(shard[-1])
    month = q * 3
    last = {3: 31, 6: 30, 9: 30, 12: 31}[month]
    return dt.date(year, month, last)


def shards_up_to(cutoff, shards: list[str] | None = None, extra_years: int = 0) -> list[str]:
    """The shards a vintage should embed, which is **not** simply those up to its training cutoff.

    The roadmap said "the whole history up to that cutoff" and that reading is too narrow. A vintage is
    the model you hold *until the next one exists*, so it has to serve decisions made after its cutoff,
    and a decision at date t needs embeddings for every call up to t. With `extra_years=0` the 2015
    vintage cannot see the first half of 2016, so a mid-2016 decision is blind to **the most recent six
    months of calls**, which is the period a predictive model weights most. **Measured:** 269,182 chunks
    or 5.7% at 2016-06-30, 378,787 or 5.2% at 2020-06-30.

    `extra_years=1` extends each vintage to the end of the year after its cutoff, so any decision date
    up to the next vintage's arrival sees every call before it, with at most 12 months of model
    staleness. Embedding post-cutoff text is not leakage: the model cannot have learned from text it
    never saw, and `index.check_point_in_time` still refuses a query whose *model* is too new.
    """
    import datetime as dt
    cutoff = dt.date.fromisoformat(cutoff) if isinstance(cutoff, str) else cutoff
    horizon = dt.date(cutoff.year + extra_years, 12, 31) if extra_years else cutoff
    return [s for s in (shards or shard_keys()) if shard_end(s) <= horizon]


def shard_keys() -> list[str]:
    """Every quarter the chunk layer holds, oldest first."""
    return sorted(p.stem.removeprefix("chunks_") for p in P.CHUNKS_DIR.glob("chunks_*.parquet"))


# --- schema and IO -------------------------------------------------------------------------------

def embeddings_schema(dim: int, dtype: str = STORE_DTYPE, meta: dict | None = None) -> pa.Schema:
    """chunk_id plus a fixed-width vector. Provenance rides in the schema metadata rather than in a
    per-row column, so the file is self-describing and costs nothing per row."""
    schema = pa.schema([
        ("chunk_id", pa.string()),
        ("embedding", pa.list_(_PA_DTYPE[dtype], dim)),
    ])
    if meta:
        schema = schema.with_metadata({str(k): str(v) for k, v in meta.items()})
    return schema


def to_table(chunk_ids: list[str], vectors: np.ndarray, dtype: str = STORE_DTYPE,
             meta: dict | None = None) -> pa.Table:
    if len(chunk_ids) != len(vectors):
        raise ValueError(f"{len(chunk_ids)} ids against {len(vectors)} vectors")
    dim = int(vectors.shape[1])
    flat = np.ascontiguousarray(vectors, dtype=_NP_DTYPE[dtype]).reshape(-1)
    arr = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=_PA_DTYPE[dtype]), dim)
    schema = embeddings_schema(dim, dtype, meta)
    return pa.Table.from_arrays([pa.array(chunk_ids, type=pa.string()), arr], schema=schema)


def read_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a part file back as ids plus a float32 matrix, for tests and for downstream use."""
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    ids = t["chunk_id"].to_pylist()
    v = np.stack(t["embedding"].to_numpy(zero_copy_only=False)).astype(np.float32)
    return ids, v


def shard_is_complete(path: Path, expected_ids) -> bool:
    """Coverage, not existence. A part file interrupted mid-shard is present and short, and treating
    it as done loses those chunks silently for the life of the embedding set."""
    if not path.exists():
        return False
    try:
        have = set(pl.read_parquet(path, columns=["chunk_id"])["chunk_id"].to_list())
    except Exception:      # noqa: BLE001 -- a corrupt part file is an incomplete part file
        return False
    return set(expected_ids).issubset(have)


# --- the encoder ---------------------------------------------------------------------------------

def resolve_device(prefer: str | None = None, require_gpu: bool = False) -> str:
    """The device to run on, refusing to fall back to CPU when a GPU was demanded.

    `require_gpu` exists because of a measured failure rather than a hypothetical one. One array task
    landed on a node whose GPU was unusable, `nvidia-smi` reporting "Failed to get device handle for
    GPU 0", so `torch.cuda.is_available()` returned False and this function quietly returned "cpu". The
    task then ran at **4 chunk/s against 265 on a working GPU**, a 66x slowdown, and reported an eta of
    64 hours for its six shards. It had consumed 1h44 of an 8 hour allocation before anyone looked.

    A job that asked SLURM for `--gres=gpu:1` and got a broken one should die immediately and be
    resubmitted onto a working node. Silently succeeding two orders of magnitude slower is the worst
    available outcome, because nothing fails and nothing finishes.
    """
    import torch
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if require_gpu:
        raise RuntimeError(
            "no GPU is visible but --require-gpu was set. On a SLURM node that was allocated "
            "--gres=gpu:1 this means the device is unusable, not that the job should run on CPU: "
            "CPU throughput here is about 4 chunk/s against 265 on a working GPU. Resubmit so the "
            "scheduler picks a different node.")
    return "cpu"


def effective_max_length(entry: R.Entry, requested: int = MAX_LENGTH) -> int:
    """`requested`, capped by what the model can actually accept.

    The default 768 is measured against the chunk layer, where the longest chunk is 710 ChronoBERT
    tokens, and ChronoBERT accepts 8192. It is wrong for a shorter-context model: `bge-small-en-v1.5`
    accepts 512 and `all-MiniLM-L6-v2` accepts 256, and a BERT-family model asked for more positions
    than it has raises on its position embeddings. Capping here rather than at each call site means the
    pipeline and the diagnostics cannot disagree about it.

    **Capping is not free and the cost is recorded in docs/decisions.md, L10.** 18.2% of chunks exceed 512
    tokens, so a 512-context retrieval model sees a truncated version of roughly one chunk in five and
    loses 2.9% of the corpus characters.
    """
    limit = (entry.extra or {}).get("max_seq_len")
    return min(requested, int(limit)) if limit else requested


def gpu_memory_gb(device: str) -> float | None:
    """Total memory of the device in GB, or None when it is not a CUDA device."""
    import torch
    if not str(device).startswith("cuda"):
        return None
    try:
        return torch.cuda.get_device_properties(torch.device(device)).total_memory / 1e9
    except Exception:      # noqa: BLE001 -- a memory probe must never be the thing that fails a run
        return None


def auto_char_budget(device: str, per_gb: int = CHARS_PER_GB, cap: int = MAX_CHAR_BUDGET) -> int | None:
    """A padded-batch character budget scaled to the device actually allocated.

    `scavenger-gpu` hands out whatever is free, and the same job landed on 24 GB A5000s and on 10.58 GB
    2080s in one array. A fixed batch of 256 fit the first and OOMed the second: three of sixteen tasks
    and the benchmark died in `apply_rotary_pos_emb` at the long end of their shards, where a batch of
    256 sequences pads out to roughly 770,000 characters.

    Returns None off CUDA, where the fixed batch size is used instead, because host memory is ample and
    MPS reports no comparable budget.
    """
    gb = gpu_memory_gb(device)
    return min(int(gb * per_gb), cap) if gb else None


# Classes a registry entry may name when its config cannot be read by the Auto classes. FinBERT's
# config.json has no `model_type`, so AutoTokenizer cannot tell it is BERT and fails on its vocab.txt.
# Named here rather than imported by any string, so the registry cannot load arbitrary code.
NAMED_CLASSES = {"bert": ("BertTokenizerFast", "BertModel")}


def auto_classes(entry: R.Entry):
    """The tokenizer and model classes for an entry: the Auto classes unless `extra.hf_classes` names
    an architecture in NAMED_CLASSES."""
    import transformers

    name = (entry.extra or {}).get("hf_classes")
    if name is None:
        return transformers.AutoTokenizer, transformers.AutoModel
    if name not in NAMED_CLASSES:
        raise ValueError(f"{entry.model_id}: hf_classes {name!r} is not one of {sorted(NAMED_CLASSES)}")
    tok_name, model_name = NAMED_CLASSES[name]
    return getattr(transformers, tok_name), getattr(transformers, model_name)


def load_encoder(entry: R.Entry, device: str | None = None, compute_dtype: str = COMPUTE_DTYPE,
                 require_gpu: bool = False):
    """The encoder body for a registered entry, with the MLM head dropped.

    `AutoModel` rather than `AutoModelForMaskedLM`: we want hidden states, and keeping the 50,368-way
    output projection in memory buys nothing. The dimension is checked against the registry, because a
    silent mismatch would write vectors that no longer match what the registry promises.
    """
    import torch

    # Two source schemes. `huggingface:` is a repo to download, `local:` is a model this project
    # trained, exported in the same format so nothing downstream needs a special case. The local
    # path is derived from the registered name and never taken from a caller, which is the same rule
    # the registry itself enforces: a model resolved by path can be the wrong vintage silently.
    if entry.source.startswith("local:"):
        name = entry.source.removeprefix("local:")
        repo = P.PROCESSED_DIR / "models" / name / "hf"
        if not (repo / "config.json").exists():
            raise FileNotFoundError(
                f"{entry.model_id}: the registry points at {repo}, which holds no exported model. "
                f"The contrastive stage writes it; run that first or fix the registry.")
        repo = str(repo)
    elif entry.source.startswith("huggingface:"):
        repo = entry.source.removeprefix("huggingface:")
    else:
        raise ValueError(
            f"{entry.model_id}: source must start with 'huggingface:' or 'local:', got "
            f"{entry.source!r}")
    device = resolve_device(device, require_gpu=require_gpu)

    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[compute_dtype]
    if device == "cpu" and torch_dtype is not torch.float32:
        torch_dtype = torch.float32     # half precision on CPU is slower, not faster

    tok_cls, model_cls = auto_classes(entry)
    tok = tok_cls.from_pretrained(repo, **(entry.extra or {}).get("tokenizer_kwargs", {}))
    model = model_cls.from_pretrained(repo, dtype=torch_dtype).to(device).eval()
    dim = int(model.config.hidden_size)
    if dim != entry.dim:
        raise ValueError(
            f"{entry.model_id}: the registry records dim {entry.dim} but {repo} has hidden_size "
            f"{dim}. Fix the registry before embedding, or every consumer reading the registry is "
            "reading a lie about these vectors.")
    return tok, model, device


def mean_pool(hidden, mask):
    """Mean over real tokens only. Padding is excluded through the attention mask rather than by
    dividing by the padded length, which would shrink every short chunk's vector toward zero."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def cls_pool(hidden, mask):
    """The first token's state. Correct for models trained with a CLS objective, wrong for an
    MLM-only encoder whose CLS was never given a sentence-level job.

    This is not interchangeable with mean pooling: the BGE and GTE retriever families are trained so
    that CLS carries the sentence, and mean-pooling them degrades retrieval, while ChronoBERT has no
    trained pooler at all and CLS carries nothing in particular. Which one a model wants is a property
    of the model, so it lives in its registry entry rather than in a caller's flag.
    """
    return hidden[:, 0]


POOLERS = {"mean": mean_pool, "cls": cls_pool}


def plan_batches(lengths, batch_size: int = DEFAULT_BATCH, char_budget: int | None = None):
    """Indices grouped into batches, shortest texts first, as a list of index lists.

    With `char_budget` set, a batch grows until its **padded** size, the member count times the
    longest member, would exceed the budget. That is the quantity the GPU actually pays for, and a
    fixed sequence count gets it wrong at both ends of this corpus: 256 chunks of 60 tokens leave the
    device idle, while 256 chunks of 700 tokens is 12x the work of the first batch. On 2019Q3 the
    measured marginal rate fell from 979 to 194 chunk/s across one shard for exactly that reason.

    A text longer than the whole budget still forms a batch of one, so the budget bounds the work per
    batch without ever dropping an input or looping forever.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    if char_budget is None:
        return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    batches, cur, widest = [], [], 0
    for i in order:
        w = max(widest, lengths[i])
        if cur and (len(cur) + 1) * w > char_budget:
            batches.append(cur)
            cur, widest = [i], lengths[i]
        else:
            cur.append(i)
            widest = w
    if cur:
        batches.append(cur)
    return batches


def encode_texts(texts, tok, model, device: str, batch_size: int = DEFAULT_BATCH,
                 max_length: int = MAX_LENGTH, normalize: bool = True,
                 char_budget: int | None = None, pooling: str = "mean",
                 quiet_oom: bool = False, progress=None) -> np.ndarray:
    """Mean-pooled, optionally L2-normalized vectors, in the order the texts came in.

    Batches are formed over length-sorted texts so a 60-token chunk is not padded out to sit beside a
    700-token one, then the original order is restored. On a corpus whose chunk lengths span 10x this
    is most of the throughput.
    """
    import torch

    if pooling not in POOLERS:
        raise ValueError(f"pooling must be one of {sorted(POOLERS)}, got {pooling!r}")
    pool = POOLERS[pooling]
    n = len(texts)
    if n == 0:
        return np.zeros((0, int(model.config.hidden_size)), dtype=np.float32)

    batches = plan_batches([len(t) for t in texts], batch_size, char_budget)
    out = np.zeros((n, int(model.config.hidden_size)), dtype=np.float32)
    done = 0

    def encode_one(idx):
        batch = [texts[i] for i in idx]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state
        vec = pool(hidden, enc["attention_mask"])
        if normalize:
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        out[idx] = vec.float().cpu().numpy()

    with torch.inference_mode():
        for idx in batches:
            # Halve and retry on OOM rather than losing the shard. The budget above is calibrated
            # from one measurement on two card sizes, so it will be wrong somewhere; this makes being
            # wrong cost a few seconds instead of a task. Order is unaffected because results are
            # scattered by index.
            pending = [list(idx)]
            while pending:
                cur = pending.pop()
                try:
                    encode_one(cur)
                except torch.OutOfMemoryError:
                    if len(cur) == 1:
                        raise
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
                    mid = len(cur) // 2
                    pending.extend([cur[:mid], cur[mid:]])
                    if not quiet_oom:
                        print(f"    OOM on {len(cur)} sequences, splitting to "
                              f"{mid} and {len(cur) - mid}", flush=True)
                    continue
                done += len(cur)
                if progress:
                    progress(done, n)
    return out


# --- the run -------------------------------------------------------------------------------------

def write_meta(entry: R.Entry, device: str, max_length: int, dtype: str, batch_size: int,
               compute_dtype: str = COMPUTE_DTYPE, up_to_cutoff: bool = False,
               extra_years: int = 0, shards: list[str] | None = None) -> Path:
    import torch
    path = meta_path(entry.model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model_id": entry.model_id,
        "source": entry.source,
        "kind": entry.kind,
        "training_cutoff": entry.training_cutoff,
        "dim": entry.dim,
        "pooling": entry.pooling,
        "normalized": entry.normalized,
        "store_dtype": dtype,
        "compute_dtype": compute_dtype,
        "max_length": max_length,
        "batch_size": batch_size,
        "device": device,
        "torch_version": torch.__version__,
        "chunker_version": _chunker_version(),
        # The window this set is meant to cover. Without it the audit has to guess, and it would report
        # a vintage's intentionally absent later shards as missing, which turns the gate into noise.
        "up_to_cutoff": up_to_cutoff,
        "extra_years": extra_years if up_to_cutoff else None,
        "expected_shards": sorted(shards) if shards else None,
        "git_sha": M.git_sha(),
        "written_at": M.utc_now(),
    }, indent=2) + "\n")
    return path


def _chunker_version() -> str | None:
    shards = shard_keys()
    if not shards:
        return None
    v = pl.read_parquet(P.chunks_shard(shards[0]), columns=["chunker_version"])
    return v["chunker_version"][0] if v.height else None


def embed_shard(shard: str, entry: R.Entry, tok, model, device: str,
                batch_size: int = DEFAULT_BATCH, max_length: int = MAX_LENGTH,
                dtype: str = STORE_DTYPE, char_budget: int | None = None,
                force: bool = False, quiet: bool = False) -> dict:
    """One quarterly shard to one part file. Idempotent on coverage, atomic on write."""
    src = P.chunks_shard(shard)
    dst = part_path(entry.model_id, shard)
    df = pl.read_parquet(src, columns=["chunk_id", "text"])
    ids = df["chunk_id"].to_list()

    if not force and shard_is_complete(dst, ids):
        return {"shard": shard, "status": "skipped", "n_chunks": len(ids), "seconds": 0.0}

    texts = df["text"].to_list()
    max_length = effective_max_length(entry, max_length)
    t0 = time.time()
    last = [t0, 0]

    def tick(done, total):
        """Report the marginal rate beside the cumulative one.

        The cumulative rate is misleading by construction here: batches are formed over
        length-sorted texts, so the first batches hold the shortest chunks and the running average
        decays through the shard. On 2019Q3 it read 979, then 843, then 654 chunk/s while nothing
        had changed. Only the whole-shard figure is comparable across runs, and the marginal figure
        is what tells you the GPU is still healthy.
        """
        if quiet or done % (batch_size * 100) >= batch_size:
            return
        now = time.time()
        cum = done / max(now - t0, 1e-9)
        marg = (done - last[1]) / max(now - last[0], 1e-9)
        last[0], last[1] = now, done
        print(f"    {shard} {done:,}/{total:,}  {marg:.0f} chunk/s now, {cum:.0f} cumulative",
              flush=True)

    vectors = encode_texts(texts, tok, model, device, batch_size=batch_size,
                           max_length=max_length, normalize=entry.normalized,
                           char_budget=char_budget, pooling=entry.pooling, progress=tick)
    elapsed = time.time() - t0

    meta = {"model_id": entry.model_id, "training_cutoff": entry.training_cutoff or "",
            "pooling": entry.pooling, "normalized": entry.normalized, "max_length": max_length,
            "chunker_version": _chunker_version() or "", "git_sha": M.git_sha()}
    M.atomic_write_table(to_table(ids, vectors, dtype=dtype, meta=meta), dst)

    row = {"stage": "embed", "shard": shard, "model_id": entry.model_id,
           "n_chunks": len(ids), "dim": entry.dim, "store_dtype": dtype,
           "seconds": round(elapsed, 1), "rate_per_s": round(len(ids) / max(elapsed, 1e-9), 1),
           "char_budget": char_budget,
           "device": device, "bytes": dst.stat().st_size,
           "git_sha": M.git_sha(), "ts": M.utc_now()}
    M.append_manifest(row)
    return {"shard": shard, "status": "written", "n_chunks": len(ids),
            "seconds": elapsed, "bytes": dst.stat().st_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="chrono-bert-v1-20241231",
                    help="a registered model_id, never a path")
    ap.add_argument("--shards", nargs="*", default=None,
                    help="quarter keys such as 2015Q1; default is every shard")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                    help="sequences per batch; ignored when --char-budget is set")
    ap.add_argument("--char-budget", type=int, default=None,
                    help="cap a batch at members times longest-member characters, which is what the "
                         "GPU pays for. Defaults to a budget scaled from the device's memory")
    ap.add_argument("--fixed-batch", action="store_true",
                    help="use --batch-size instead of a memory-scaled char budget")
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    ap.add_argument("--dtype", default=STORE_DTYPE, choices=sorted(_PA_DTYPE))
    ap.add_argument("--device", default=None, help="cuda, mps or cpu; autodetected when omitted")
    ap.add_argument("--require-gpu", action="store_true",
                    help="fail rather than running on CPU when no GPU is visible. Set this on any "
                         "SLURM job that requested a GPU, so a broken device fails fast")
    ap.add_argument("--compute-dtype", default=COMPUTE_DTYPE,
                    choices=["float32", "float16", "bfloat16"],
                    help="the forward pass precision, distinct from --dtype which is storage")
    ap.add_argument("--up-to-cutoff", action="store_true",
                    help="restrict to shards the model should serve: up to its training cutoff plus "
                         "--extra-years, which is the expanding-window design for a vintage set")
    ap.add_argument("--extra-years", type=int, default=1,
                    help="years past the cutoff a vintage should cover, because it is the model held "
                         "until the next one exists. 0 restricts it to its cutoff, which leaves any "
                         "later decision blind to the calls in between")
    ap.add_argument("--force", action="store_true", help="re-embed shards that are already complete")
    ap.add_argument("--dry-run", action="store_true", help="report what would run and exit")
    args = ap.parse_args(argv)

    entry = R.load().by_name(args.model)
    # `--shards` with no values is a caller bug, not a request for everything. Treating [] as "all"
    # let three retried SLURM tasks each embed the whole corpus concurrently. Omitting the flag
    # entirely still means every shard, which is the intended way to ask for that.
    if args.shards is not None and not args.shards:
        raise SystemExit("--shards was given with no values. Omit the flag to embed every shard.")
    shards = args.shards or shard_keys()
    if args.up_to_cutoff:
        if not entry.cutoff:
            raise SystemExit(
                f"--up-to-cutoff needs a training cutoff and {entry.model_id} is {entry.kind} without "
                "one. Drop the flag to embed every shard.")
        before = len(shards)
        shards = shards_up_to(entry.cutoff, shards, args.extra_years)
        horizon = (f"{entry.cutoff.year + args.extra_years}-12-31" if args.extra_years
                   else str(entry.cutoff))
        print(f"expanding window: {len(shards)} of {before} shards end on or before {horizon} "
              f"(cutoff {entry.cutoff} plus {args.extra_years} year(s))")
        if not shards:
            raise SystemExit(f"no shard ends on or before {entry.cutoff}; nothing to embed")
    unknown = sorted(set(shards) - set(shard_keys()))
    if unknown:
        raise SystemExit(f"no chunk shard for {unknown}. Available: {shard_keys()[:3]} ... "
                         f"{shard_keys()[-3:]}")

    todo = []
    for s in shards:
        ids = pl.read_parquet(P.chunks_shard(s), columns=["chunk_id"])["chunk_id"].to_list()
        if args.force or not shard_is_complete(part_path(entry.model_id, s), ids):
            todo.append((s, len(ids)))
    total = sum(n for _, n in todo)
    print(f"{entry.model_id}: {len(shards)} shards requested, {len(todo)} to embed, "
          f"{total:,} chunks, dim {entry.dim}, store {args.dtype}")
    if args.dry_run or not todo:
        return 0

    tok, model, device = load_encoder(entry, args.device, args.compute_dtype, args.require_gpu)
    budget = args.char_budget
    if budget is None and not args.fixed_batch:
        budget = auto_char_budget(device)
    gb = gpu_memory_gb(device)
    eff = effective_max_length(entry, args.max_length)
    capped = "" if eff == args.max_length else f" (capped from {args.max_length} by the model)"
    print(f"loaded on {device} in {args.compute_dtype}"
          + (f" ({gb:.1f} GB)" if gb else "")
          + f", max_length {eff}{capped}, "
          + (f"char budget {budget:,}" if budget else f"batch {args.batch_size}"))
    write_meta(entry, device, args.max_length, args.dtype, args.batch_size, args.compute_dtype,
               up_to_cutoff=args.up_to_cutoff, extra_years=args.extra_years, shards=shards)

    t0, done = time.time(), 0
    for s, n in todo:
        res = embed_shard(s, entry, tok, model, device, batch_size=args.batch_size,
                          max_length=args.max_length, dtype=args.dtype,
                          char_budget=budget, force=args.force)
        done += res["n_chunks"]
        el = time.time() - t0
        eta = (total - done) / (done / el) if done else float("nan")
        print(f"  {s}: {res['status']} {res['n_chunks']:,} in {res['seconds']:.0f}s "
              f"({res['n_chunks'] / max(res['seconds'], 1e-9):.0f} chunk/s), "
              f"{done:,}/{total:,} done, eta {eta / 3600:.2f}h", flush=True)
    print(f"done: {done:,} chunks in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
