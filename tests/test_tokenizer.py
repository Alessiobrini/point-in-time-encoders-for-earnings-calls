"""Edge cases for the vintage tokenizer.

The load-bearing property is the date restriction. A tokenizer carries no record of what text it saw,
so fitting one on the full corpus would leak later vocabulary into every vintage built on it and
nothing afterwards could detect it. That is checked here, along with the byte-level guarantee that
gives this corpus no out-of-vocabulary tokens at all.
"""
import datetime as dt
import json

import polars as pl
import pytest

from ecalls.train import tokenizer as T

D = dt.date


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Four quarterly shards whose text differs by year, so leakage is visible in the vocabulary."""
    monkeypatch.setattr(T.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(T.P, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(T, "TOKENIZER_DIR", tmp_path / "processed" / "tokenizers")
    monkeypatch.setattr(T.P, "chunks_shard", lambda s: T.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    from ecalls.embed import encode as E
    monkeypatch.setattr(E.P, "CHUNKS_DIR", T.P.CHUNKS_DIR)
    T.P.CHUNKS_DIR.mkdir(parents=True)

    words = {"2010Q1": "margin freight inventory", "2010Q4": "guidance backlog pricing",
             "2015Q2": "zzzfuturewordzzz blockchain", "2020Q1": "qqqpandemicqqq lockdown"}
    for shard, w in words.items():
        texts = [f"{w} discussion number {i} of the quarter" for i in range(400)]
        pl.DataFrame({"chunk_id": [f"{shard}-{i}" for i in range(400)], "text": texts,
                      "n_chars": [len(t) for t in texts]}).write_parquet(T.P.chunks_shard(shard))
    return words


# --- the date restriction, which is the reason this module exists --------------------------------

def test_text_iter_reads_only_shards_inside_the_window(corpus):
    got = [t for batch in T.text_iter("2010-12-31") for t in batch]
    assert len(got) == 800                                  # the two 2010 shards only
    assert not any("zzzfuturewordzzz" in t for t in got)
    assert not any("qqqpandemicqqq" in t for t in got)


def test_a_fitted_tokenizer_has_not_seen_later_vocabulary(corpus):
    """The failure this prevents. A word that appears only after the cutoff must not become a token,
    because a tokenizer fit on the full corpus leaks future vocabulary into every vintage."""
    out = T.fit("2010-12-31", vocab_size=500)
    tok = T.load(json.loads((out / "_meta.json").read_text())["name"])
    vocab = set(tok.get_vocab())
    assert not any("zzzfuturewordzzz" in v for v in vocab)
    assert not any("qqqpandemicqqq" in v for v in vocab)
    # and it did learn the in-window vocabulary
    assert any("margin" in v or "freight" in v for v in vocab)


def test_the_window_is_recorded_beside_the_tokenizer(corpus):
    """Without this nobody can state the training cutoff of a model built on it."""
    out = T.fit("2010-12-31", vocab_size=500)
    meta = json.loads((out / "_meta.json").read_text())
    assert meta["fit_cutoff"] == "2010-12-31"
    assert meta["fit_shards"] == ["2010Q1", "2010Q4"]
    assert meta["n_shards"] == 2
    assert meta["algorithm"] == "byte-level BPE"


def test_a_cutoff_before_the_corpus_raises_rather_than_fitting_on_nothing(corpus):
    with pytest.raises(SystemExit, match="no chunk shard"):
        list(T.text_iter("1990-01-01"))


def test_limit_chunks_caps_the_sample_exactly(corpus):
    got = [t for batch in T.text_iter("2020-12-31", limit_chunks=37) for t in batch]
    assert len(got) == 37


# --- the byte-level guarantee --------------------------------------------------------------------

def test_no_text_is_ever_out_of_vocabulary(corpus):
    """Byte-level BPE has no UNK at all. On this corpus that matters more than usual, since tickers,
    currency amounts and one-off product names appear constantly and WordPiece would destroy the
    unseen ones."""
    out = T.fit("2010-12-31", vocab_size=500)
    tok = T.load(json.loads((out / "_meta.json").read_text())["name"])
    unk = tok.token_to_id("[UNK]")
    for weird in ["AAPL", "$1.25", "CUSIP 037833100", "Ürün", "zzzneverseenzzz", "→", "🙂"]:
        ids = tok.encode(weird, add_special_tokens=False).ids
        assert ids, weird
        assert unk not in ids, f"{weird} produced [UNK]"
        # Digit splitting and the prefix space mean the decode is not character-identical, so the
        # invariant is that no information is lost rather than that the string is reproduced exactly.
        assert tok.decode(ids).replace(" ", "") == weird.replace(" ", ""), weird


def test_special_tokens_are_added_and_come_first(corpus):
    out = T.fit("2010-12-31", vocab_size=500)
    tok = T.load(json.loads((out / "_meta.json").read_text())["name"])
    for s in T.SPECIALS:
        assert tok.token_to_id(s) is not None
    assert tok.token_to_id("[PAD]") == 0            # pad at 0 is the convention the collator assumes


def test_the_post_processor_wraps_a_sequence_in_cls_and_sep(corpus):
    out = T.fit("2010-12-31", vocab_size=500)
    tok = T.load(json.loads((out / "_meta.json").read_text())["name"])
    enc = tok.encode("margin discussion")
    assert enc.tokens[0] == "[CLS]" and enc.tokens[-1] == "[SEP]"


def test_digit_splitting_makes_numerically_similar_text_tokenize_consistently(corpus):
    """Without it the tokenizer memorizes frequent numbers whole, so 2010 is one token and 2011 is
    three, and figures that mean similar things fragment differently."""
    tok_split, tr = T.build(500, split_digits=True)
    tok_split.train_from_iterator(T.text_iter("2020-12-31"), trainer=tr)
    n = [len(tok_split.encode(str(y), add_special_tokens=False).ids) for y in (2010, 2011, 1997)]
    assert len(set(n)) == 1                          # every 4-digit number costs the same
    assert n[0] == 4


def test_loading_a_missing_tokenizer_says_how_to_make_one(corpus):
    with pytest.raises(FileNotFoundError, match="fit"):
        T.load("does-not-exist")


# --- the compression measurement -----------------------------------------------------------------

def test_compression_reports_chars_per_token_and_whole_word_share(corpus):
    out = T.fit("2010-12-31", vocab_size=500)
    tok = T.load(json.loads((out / "_meta.json").read_text())["name"])
    c = T.compression(tok, ["margin freight inventory discussion"] * 20)
    assert c["chars_per_token"] > 1.0
    assert 0.0 <= c["whole_word_share"] <= 1.0
    assert c["tokens_total"] > 0
    assert c["median_tokens_per_chunk"] > 0


def test_compression_improves_with_a_larger_vocabulary(corpus):
    """The headline tradeoff. A bigger vocabulary fits more text per token, which is why the choice is
    measured rather than inherited from whichever model was read last."""
    ev = ["margin freight inventory guidance backlog pricing discussion"] * 40
    small, tr_s = T.build(300, split_digits=True)
    small.train_from_iterator(T.text_iter("2020-12-31"), trainer=tr_s)
    big, tr_b = T.build(2000, split_digits=True)
    big.train_from_iterator(T.text_iter("2020-12-31"), trainer=tr_b)
    assert T.compression(big, ev)["chars_per_token"] > T.compression(small, ev)["chars_per_token"]


def test_special_ids_refuses_a_vocabulary_with_no_boundary_pair():
    """Otherwise `token_to_id` returns None and None is packed into a uint16 array."""
    class NoSpecials:
        def token_to_id(self, t):
            return None

    with pytest.raises(ValueError, match="no sequence-start"):
        T.special_ids(NoSpecials(), "pretend")


def test_special_ids_accepts_either_naming():
    """Ours use [CLS]/[SEP]; a borrowed vocabulary might use <s>/</s>."""
    class Roberta:
        def token_to_id(self, t):
            return {"<s>": 0, "</s>": 2}.get(t)

    assert T.special_ids(Roberta(), "pretend") == (0, 2)
