"""Fit the tokenizer for the vintage encoder set, once, on the earliest window.

One tokenizer is reused across every vintage, for two reasons. Fitting it on the whole corpus would leak later vocabulary into every
vintage, new tickers and product and event names among them. And letting the tokenizer vary with the
data means a measured difference across vintages cannot be attributed to either one. Fitting once on
the earliest window satisfies both, and it stays leakage-free because that window is a subset of
every later one.

**The date restriction is enforced here rather than left to the caller.** `fit` takes a cutoff and
reads only shards ending on or before it, so a tokenizer cannot accidentally be fit on the full
corpus. That is the failure this module exists to prevent, and it would be invisible afterwards: a
tokenizer carries no record of what it saw.

A tokenizer cannot be refit while keeping existing weights, because the input embedding matrix and the
output head are indexed by token id, so the vocabulary is frozen for the life of the model set.

Run:
    python -m ecalls.train.tokenizer measure --cutoff 2010-12-31
    python -m ecalls.train.tokenizer fit --cutoff 2010-12-31 --vocab-size 32768
"""
import argparse
import json

import polars as pl

from ecalls.config import paths as P
from ecalls.embed.encode import shards_up_to
from ecalls.ingest.manifest import git_sha, utc_now

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
DEFAULT_VOCAB = 32_768
TOKENIZER_DIR = P.PROCESSED_DIR / "tokenizers"


def tokenizer_path(name: str):
    return TOKENIZER_DIR / name


def text_iter(cutoff: str, extra_years: int = 0, limit_chunks: int | None = None,
              batch: int = 20_000):
    """Chunk text from shards ending on or before `cutoff`, in batches.

    Yields lists of strings rather than one string at a time, because the Rust trainer is much faster
    fed in batches and the corpus does not fit in memory as one list.
    """
    shards = shards_up_to(cutoff, extra_years=extra_years)
    if not shards:
        raise SystemExit(f"no chunk shard ends on or before {cutoff}")
    seen = 0
    for s in shards:
        df = pl.read_parquet(P.chunks_shard(s), columns=["text"])
        texts = df["text"].to_list()
        for i in range(0, len(texts), batch):
            part = texts[i:i + batch]
            if limit_chunks is not None and seen + len(part) > limit_chunks:
                part = part[: limit_chunks - seen]
            if not part:
                return
            seen += len(part)
            yield part
            if limit_chunks is not None and seen >= limit_chunks:
                return


def build(vocab_size: int = DEFAULT_VOCAB, split_digits: bool = True):
    """An untrained byte-level BPE tokenizer with BERT's special tokens.

    Byte-level BPE rather than WordPiece, because it has no out-of-vocabulary token at all. On this
    corpus that matters more than usual: tickers, CUSIPs, currency amounts and one-off product names
    appear constantly, and WordPiece would map the unseen ones to `[UNK]` and destroy them. ModernBERT
    makes the same choice.

    `split_digits` puts every digit in its own token. Without it the tokenizer memorizes frequent
    numbers as single tokens, so "2010" is one token and "2011" is three, which makes numerically
    similar text tokenize inconsistently. Earnings calls are dense with figures, so the option is
    measured rather than assumed, see `measure`.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    pre = [pre_tokenizers.ByteLevel(add_prefix_space=True)]
    if split_digits:
        pre.insert(0, pre_tokenizers.Digits(individual_digits=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence(pre)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                  show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    return tok, trainer


def fit(cutoff: str, vocab_size: int = DEFAULT_VOCAB, split_digits: bool = True,
        name: str | None = None, limit_chunks: int | None = None):
    """Fit and save a tokenizer, with the window it saw recorded beside it."""
    from tokenizers.processors import TemplateProcessing

    name = name or f"ecalls-bpe-{vocab_size}-{cutoff.replace('-', '')}"
    tok, trainer = build(vocab_size, split_digits)
    shards = shards_up_to(cutoff)
    tok.train_from_iterator(text_iter(cutoff, limit_chunks=limit_chunks), trainer=trainer)

    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls), ("[SEP]", sep)])

    out = tokenizer_path(name)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    (out / "_meta.json").write_text(json.dumps({
        "name": name,
        "algorithm": "byte-level BPE",
        "vocab_size": tok.get_vocab_size(),
        "requested_vocab_size": vocab_size,
        "split_digits": split_digits,
        "specials": SPECIALS,
        # The window is the whole point: a tokenizer carries no record of what it saw, so without this
        # nobody can state the training cutoff of any model built on it.
        "fit_cutoff": cutoff,
        "fit_shards": shards,
        "n_shards": len(shards),
        "limit_chunks": limit_chunks,
        "git_sha": git_sha(),
        "fitted_at": utc_now(),
    }, indent=2) + "\n")
    return out


REGISTRY_PREFIX = "registry:"


def load_any(name: str):
    """A tokenizer by name, from this project's fits or from a registered model.

    `registry:<model_id>` returns the backend tokenizer of a registered encoder, which is what
    domain-adaptive pretraining needs: continuing masked language modeling from ChronoBERT means
    packing the corpus with **ChronoBERT's** vocabulary, not ours, or every token id means something
    different to the model than it did during its pretraining.

    The backend object is returned rather than the transformers wrapper because the packer calls
    `encode_batch` and `token_to_id`, which belong to the fast tokenizer underneath.
    """
    if not name.startswith(REGISTRY_PREFIX):
        return load(name)
    from transformers import AutoTokenizer

    from ecalls.chunk import registry as R
    entry = R.load().by_name(name.removeprefix(REGISTRY_PREFIX))
    if not entry.source.startswith("huggingface:"):
        raise ValueError(
            f"{entry.model_id}: only a huggingface source has a tokenizer to borrow, got "
            f"{entry.source!r}")
    return AutoTokenizer.from_pretrained(entry.source.removeprefix("huggingface:")).backend_tokenizer


def fingerprint(tok) -> str:
    """sha256 of a tokenizer's full serialization: vocabulary, merges, normalizer and specials.

    Two tokenizers with the same fingerprint give every text the same ids, which is what lets a
    vintage use a vocabulary registered under a later model when the files are the same.
    """
    import hashlib
    return hashlib.sha256(tok.to_str().encode("utf-8")).hexdigest()


def special_ids(tok, name: str) -> tuple[int, int]:
    """The sequence-start and sequence-end ids, under whichever naming the vocabulary uses.

    Ours are `[CLS]` and `[SEP]`. ChronoBERT is ModernBERT and uses the same two, but a borrowed
    vocabulary that used `<s>` and `</s>` would otherwise silently pack `None` into a uint16 array.
    """
    for start, end in (("[CLS]", "[SEP]"), ("<s>", "</s>"), ("<|endoftext|>", "<|endoftext|>")):
        a, b = tok.token_to_id(start), tok.token_to_id(end)
        if a is not None and b is not None:
            return a, b
    raise ValueError(
        f"{name}: found no sequence-start and sequence-end pair in this vocabulary. Packing would "
        "write None into the token array.")


def load(name: str):
    from tokenizers import Tokenizer
    d = tokenizer_path(name)
    f = d / "tokenizer.json"
    if not f.exists():
        raise FileNotFoundError(
            f"no tokenizer at {f}. Fit one with "
            f"`python -m ecalls.train.tokenizer fit --cutoff 2010-12-31`.")
    return Tokenizer.from_file(str(f))


def compression(tok, texts: list[str]) -> dict:
    """Characters per token and the share of words left whole.

    Characters per token is the headline number, since it sets how much text fits in a context window
    and therefore the compute per unit of corpus. The whole-word share is reported beside it because a
    tokenizer can win on compression by fragmenting rare words, which hurts a domain model on exactly
    the terms that carry the domain.
    """
    import numpy as np

    enc = tok.encode_batch(texts, add_special_tokens=False)
    n_tok = sum(len(e.ids) for e in enc)
    n_chars = sum(len(t) for t in texts)
    words = [w for t in texts for w in t.split()]
    single = sum(1 for w in words
                 if len(tok.encode(w, add_special_tokens=False).ids) == 1)
    lens = np.array([len(e.ids) for e in enc])
    return {"chars_per_token": n_chars / max(n_tok, 1),
            "tokens_total": n_tok,
            "whole_word_share": single / max(len(words), 1),
            "median_tokens_per_chunk": float(np.median(lens)),
            "p99_tokens_per_chunk": float(np.percentile(lens, 99))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="compare vocab sizes and digit handling on a sample")
    m.add_argument("--cutoff", default="2010-12-31")
    m.add_argument("--fit-chunks", type=int, default=250_000,
                   help="chunks used to fit each candidate; a sample, since this is a comparison")
    m.add_argument("--eval-chunks", type=int, default=20_000)
    m.add_argument("--vocab-sizes", type=int, nargs="*", default=[16_384, 32_768, 50_368])

    f = sub.add_parser("fit", help="fit the real tokenizer on the whole window")
    f.add_argument("--cutoff", default="2010-12-31")
    f.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    f.add_argument("--no-split-digits", action="store_true")
    f.add_argument("--name", default=None)
    f.add_argument("--limit-chunks", type=int, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "fit":
        out = fit(args.cutoff, args.vocab_size, not args.no_split_digits, args.name,
                  args.limit_chunks)
        meta = json.loads((out / "_meta.json").read_text())
        print(f"fitted {meta['name']}: vocab {meta['vocab_size']:,} over {meta['n_shards']} shards "
              f"up to {meta['fit_cutoff']}")
        print(f"  {out}")
        return 0

    # measure
    import itertools
    ev = list(itertools.chain.from_iterable(
        text_iter(args.cutoff, limit_chunks=args.eval_chunks)))
    print(f"evaluating on {len(ev):,} chunks from the window up to {args.cutoff}\n")
    print(f"{'vocab':>7} {'digits':>7} {'chars/tok':>10} {'whole words':>12} "
          f"{'median tok':>11} {'p99 tok':>8}")
    for vs in args.vocab_sizes:
        for sd in (True, False):
            tok, trainer = build(vs, sd)
            tok.train_from_iterator(
                text_iter(args.cutoff, limit_chunks=args.fit_chunks), trainer=trainer)
            c = compression(tok, ev)
            print(f"{vs:7d} {'split' if sd else 'keep':>7} {c['chars_per_token']:10.3f} "
                  f"{c['whole_word_share']:11.1%} {c['median_tokens_per_chunk']:11.0f} "
                  f"{c['p99_tokens_per_chunk']:8.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
