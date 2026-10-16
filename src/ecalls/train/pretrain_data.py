"""Pack a vintage's text into fixed-length token sequences for masked language modeling.

Three properties matter, and each one corresponds to a way this stage can quietly ruin the vintage set.

**The window is enforced here.** Only shards ending on or before the cutoff are read, so a vintage
cannot be pretrained on text from after its own cutoff. Section 5 of the methodology notes flags this as
easy to get wrong because the packing code is separate from the corpus filter.

**Sequences never cross a call boundary.** Packing runs within one `keydevid` and starts a new sequence
at the next call. Crossing the boundary would teach the model associations between unrelated companies
that happen to be adjacent in a shard, and adjacency in a shard is an artifact of how the corpus was
written rather than anything about the text.

**The packing is deterministic.** Shards are read in calendar order and calls within a shard in
`keydevid` order, so a rebuild produces the same array. A vintage set whose training data cannot be
reproduced cannot support a claim about what any vintage saw.

Tokens are stored as `uint16`, which is exact because the vocabulary is 32,768 and fits in 16 bits, and
which halves the file against `int32`. The file is a flat binary of token ids that a trainer memory-maps,
the usual arrangement for pretraining data.

Run:
    python -m ecalls.train.pretrain_data --cutoff 2010-12-31
"""
import argparse
import json

import numpy as np
import polars as pl

from ecalls.config import paths as P
from ecalls.embed.encode import shards_up_to
from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.train import holdout as HO
from ecalls.train import tokenizer as T

CONTEXT = 768        # measured: covers 99.95% of chunks whole; 512 would truncate 21.9% of them
DTYPE = np.uint16    # exact for a 32,768 vocabulary, and half the size of int32
PRETRAIN_DIR = P.PROCESSED_DIR / "pretrain"


def window_dir(name: str):
    return PRETRAIN_DIR / name


def window_name(cutoff: str, context: int, tokenizer_name: str,
                holdout_only: bool = False) -> str:
    suffix = "-holdout" if holdout_only else ""
    # A borrowed vocabulary is named `registry:<model_id>`, and a colon in a directory name is legal
    # but miserable to type and ambiguous on some filesystems.
    tag = tokenizer_name.replace("registry:", "borrowed-")
    return f"{tag}-ctx{context}-{cutoff.replace('-', '')}{suffix}"


def pack_call(ids_per_chunk: list[list[int]], context: int, cls_id: int, sep_id: int):
    """Sequences of exactly `context` tokens from one call's chunks, dropping the ragged tail.

    Each chunk is wrapped in `[CLS] ... [SEP]` and chunks are concatenated until the context is full.
    A chunk longer than the context on its own is split across sequences rather than truncated, because
    dropping the tail of a long answer would silently remove the part of an answer where a quantified
    statement usually appears.

    The final partial sequence of each call is dropped rather than padded. Padding would spend compute
    on positions that teach nothing, and at a mean of 300 tokens per chunk the loss is small.
    """
    out = []
    buf: list[int] = []
    for ids in ids_per_chunk:
        buf.extend([cls_id, *ids, sep_id])
        while len(buf) >= context:
            out.append(buf[:context])
            buf = buf[context:]
    return out


def pack_window(cutoff: str, tokenizer_name: str, context: int = CONTEXT,
                extra_years: int = 0, limit_shards: int | None = None,
                quiet: bool = False, holdout_fraction: float = HO.DEFAULT_FRACTION,
                holdout_salt: str = HO.SALT, holdout_only: bool = False) -> dict:
    """Write one vintage's packed pretraining corpus and its sidecar.

    `holdout_only` inverts the split and packs the calls training never saw, which is how a vintage is
    evaluated. Its purpose is to make losses comparable: each vintage's training loss is measured on its
    own corpus, and a later vintage's corpus is larger and more varied, so a higher loss there is not
    evidence of a less converged model. One held-out set inside every vintage's window gives all fifteen
    a common yardstick.
    """
    # An empty window is checked first, because "no data before this date" is the useful message when
    # both problems are present and a tokenizer mismatch would otherwise mask it.
    if holdout_only and not holdout_fraction:
        raise ValueError(
            "holdout_only with holdout_fraction=0 would pack the whole corpus and call it the "
            "held-out set, which is the quietest possible way to evaluate on training data.")
    shards = shards_up_to(cutoff, extra_years=extra_years)
    if limit_shards:
        shards = shards[:limit_shards]
    if not shards:
        raise SystemExit(f"no chunk shard ends on or before {cutoff}")

    tok = T.load_any(tokenizer_name)
    if tokenizer_name.startswith(T.REGISTRY_PREFIX):
        # A borrowed vocabulary, for domain-adaptive pretraining. Its cutoff is the registered
        # model's, which `registry` already validated, and it was not fit on our corpus at all, so
        # the fit-cutoff check below does not apply. Its own cutoff still must not post-date the
        # vintage, or the vocabulary itself carries later information.
        from ecalls.chunk import registry as R
        entry = R.load().by_name(tokenizer_name.removeprefix(T.REGISTRY_PREFIX))
        if entry.training_cutoff and entry.training_cutoff > cutoff:
            raise ValueError(
                f"{entry.model_id} has training cutoff {entry.training_cutoff}, after this vintage's "
                f"{cutoff}. Borrowing its vocabulary would carry later information into an earlier "
                "vintage, which is the same leak D20 split digits to prevent.")
        tmeta = {"fit_cutoff": entry.training_cutoff or cutoff,
                 "vocab_size": tok.get_vocab_size(), "borrowed_from": entry.model_id}
    else:
        tmeta = json.loads((T.tokenizer_path(tokenizer_name) / "_meta.json").read_text())
        if tmeta["fit_cutoff"] > cutoff:
            raise ValueError(
                f"the tokenizer {tokenizer_name} was fit on text up to {tmeta['fit_cutoff']}, which "
                f"is after this vintage's cutoff {cutoff}. Every vintage must reuse the tokenizer fit "
                "on the earliest window, or later vocabulary leaks into an earlier model.")
    cls_id, sep_id = T.special_ids(tok, tokenizer_name)

    name = window_name(cutoff, context, tokenizer_name, holdout_only)
    d = window_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    bin_path = d / "tokens.bin"
    tmp = bin_path.with_suffix(".bin.tmp")

    if tok.get_vocab_size() > np.iinfo(DTYPE).max + 1:
        raise ValueError(
            f"{tokenizer_name} has {tok.get_vocab_size():,} tokens, which does not fit in "
            f"{np.dtype(DTYPE).name}. The packed array would wrap silently and every id above the "
            "limit would decode to the wrong token.")
    n_seq = 0
    n_chunks = 0
    n_held = 0
    offsets = []
    with open(tmp, "wb") as fh:
        for s in shards:
            df = (pl.read_parquet(P.chunks_shard(s), columns=["keydevid", "chunk_id", "text"])
                  .sort(["keydevid", "chunk_id"]))          # deterministic order within the shard
            if holdout_fraction:
                # Held-out calls are excluded from masked language modeling as well as from the
                # contrastive stage. A model that has seen a call's exact text can memorize it, and the
                # retrieval benchmark draws its queries from those same calls, so training on them would
                # inflate the one number Phase 4 steers by.
                keep = [k for k in df["keydevid"].unique().to_list()
                        if HO.is_holdout(k, holdout_fraction, holdout_salt) == holdout_only]
                n_held += df["keydevid"].n_unique() - len(keep)
                df = df.filter(pl.col("keydevid").is_in(set(keep)))
            for _, sub in df.group_by("keydevid", maintain_order=True):
                texts = sub["text"].to_list()
                n_chunks += len(texts)
                enc = tok.encode_batch(texts, add_special_tokens=False)
                seqs = pack_call([e.ids for e in enc], context, cls_id, sep_id)
                if seqs:
                    np.asarray(seqs, dtype=DTYPE).tofile(fh)
                    n_seq += len(seqs)
            offsets.append({"shard": s, "end": n_seq})
            if not quiet:
                print(f"  {s}: {n_seq:,} sequences so far", flush=True)
    tmp.replace(bin_path)

    meta = {
        "name": name,
        "cutoff": cutoff,
        "extra_years": extra_years,
        "context": context,
        "dtype": str(np.dtype(DTYPE)),
        "n_sequences": n_seq,
        "n_tokens": n_seq * context,
        "n_chunks_read": n_chunks,
        "n_calls_held_out": n_held,
        "holdout_only": holdout_only,
        **HO.describe(holdout_fraction, holdout_salt),
        "shards": shards,
        "n_shards": len(shards),
        # The cumulative sequence count after each shard. Because shards are written in calendar order
        # and no sequence crosses a call, a vintage's window is exactly a prefix of this array, so one
        # packed corpus serves every vintage. Packing each separately would store the same text 7.7
        # times over, 46 GB against 6 GB.
        "shard_offsets": offsets,
        "tokenizer": tokenizer_name,
        "tokenizer_fit_cutoff": tmeta["fit_cutoff"],
        "vocab_size": tmeta["vocab_size"],
        "bytes": bin_path.stat().st_size,
        "git_sha": git_sha(),
        "built_at": utc_now(),
    }
    (d / "_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def sequences_up_to(meta: dict, cutoff: str) -> int:
    """How many leading sequences belong to the window ending at `cutoff`.

    Raises rather than returning zero when the cutoff precedes every shard, because a vintage silently
    training on nothing is the failure this whole module guards against.
    """
    offs = meta.get("shard_offsets")
    if not offs:
        raise ValueError(
            f"{meta['name']} was packed before shard offsets were recorded, so a prefix cannot be "
            "taken. Rebuild it.")
    want = set(shards_up_to(cutoff, [o["shard"] for o in offs]))
    end = 0
    for o in offs:
        if o["shard"] in want:
            end = max(end, o["end"])
    if end == 0:
        raise ValueError(
            f"no shard of {meta['name']} ends on or before {cutoff}; the earliest is "
            f"{offs[0]['shard']}")
    return end


def increment_bounds(meta: dict, start_cutoff: str, cutoff: str) -> tuple[int, int]:
    """`(first, end)` of the sequences whose calls fall after `start_cutoff` and on or before `cutoff`.

    A chained segment (D29) trains on one year's new text and nothing older. The packed corpus is in
    calendar order and a window is a leading slice of it, so the increment is the difference of two
    prefixes. Both ends come from `sequences_up_to`, the same function that sizes every from-scratch
    vintage, so consecutive increments meet exactly and a segment cannot see a sequence the
    from-scratch vintage of its year could not.

    Raises on an empty increment, because a segment that trains for zero steps still writes a record
    and would register as a vintage identical to its parent.
    """
    if start_cutoff >= cutoff:
        raise ValueError(f"the increment must start before it ends, got {start_cutoff} to {cutoff}")
    if cutoff > meta["cutoff"]:
        raise ValueError(
            f"{meta['name']} was packed to {meta['cutoff']}, so it cannot supply the text through "
            f"{cutoff}. The slice would end early and the record would claim the later cutoff.")
    first = sequences_up_to(meta, start_cutoff)
    end = sequences_up_to(meta, cutoff)
    if end <= first:
        raise ValueError(f"{meta['name']} holds no sequence after {start_cutoff} and on or before {cutoff}")
    return first, end


def load_window(name: str, cutoff: str | None = None):
    """Memory-map a packed corpus, optionally only the prefix belonging to one vintage.

    `cutoff` is how a single packed corpus serves every vintage: the array is in calendar order, so the
    window is a prefix. The returned meta reports the sliced counts rather than the file's, so a caller
    reading `n_sequences` gets what it was handed.
    """
    d = window_dir(name)
    f = d / "_meta.json"
    if not f.exists():
        raise FileNotFoundError(
            f"no packed corpus at {d}. Build it with "
            f"`python -m ecalls.train.pretrain_data --cutoff <date>`.")
    meta = json.loads(f.read_text())
    arr = np.memmap(d / "tokens.bin", dtype=np.dtype(meta["dtype"]), mode="r",
                    shape=(meta["n_sequences"], meta["context"]))
    if cutoff is None:
        return arr, meta
    end = sequences_up_to(meta, cutoff)
    sliced = {**meta, "n_sequences": end, "n_tokens": end * meta["context"],
              "sliced_to_cutoff": cutoff, "full_n_sequences": meta["n_sequences"]}
    return arr[:end], sliced


def load_increment(name: str, start_cutoff: str, cutoff: str):
    """Memory-map the slice of a packed corpus between two cutoffs, for one chained segment.

    The returned meta reports the slice's own counts, as `load_window` does for a prefix, and adds the
    absolute indices of its first and last sequence so the run record can state exactly what it saw.
    """
    arr, meta = load_window(name)
    first, end = increment_bounds(meta, start_cutoff, cutoff)
    sliced = {**meta, "n_sequences": end - first, "n_tokens": (end - first) * meta["context"],
              "sliced_from_cutoff": start_cutoff, "sliced_to_cutoff": cutoff,
              "first_sequence": first, "last_sequence": end - 1,
              "full_n_sequences": meta["n_sequences"]}
    return arr[first:end], sliced


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--tokenizer", default="ecalls-bpe-32768-20101231")
    ap.add_argument("--context", type=int, default=CONTEXT)
    ap.add_argument("--extra-years", type=int, default=0,
                    help="0 for pretraining: a vintage must not see text past its own cutoff. The "
                         "extra year used for the embedding sets is about serving decisions, not "
                         "training")
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--holdout-only", action="store_true",
                    help="pack the held-out calls instead of the trainable ones, for evaluation")
    args = ap.parse_args(argv)

    m = pack_window(args.cutoff, args.tokenizer, args.context, args.extra_years,
                    args.limit_shards, holdout_only=args.holdout_only)
    print(f"\n{m['name']}")
    print(f"  {m['n_sequences']:,} sequences of {m['context']} = {m['n_tokens']/1e9:.3f} B tokens")
    print(f"  from {m['n_chunks_read']:,} chunks over {m['n_shards']} shards up to {m['cutoff']}")
    print(f"  {m['bytes']/1e9:.2f} GB, tokenizer {m['tokenizer']} fit to {m['tokenizer_fit_cutoff']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
