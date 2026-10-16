"""Edge cases for the packed pretraining corpus.

Three properties decide whether the vintage set measures anything, and each is a way this stage can
quietly ruin it: the window restriction, sequences never crossing a call boundary, and determinism.
"""
import json

import numpy as np
import polars as pl
import pytest

from ecalls.train import pretrain_data as PD
from ecalls.train import tokenizer as T


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(PD.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(PD.P, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(PD, "PRETRAIN_DIR", tmp_path / "processed" / "pretrain")
    monkeypatch.setattr(PD.P, "chunks_shard", lambda s: PD.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    monkeypatch.setattr(T.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    monkeypatch.setattr(T, "TOKENIZER_DIR", tmp_path / "processed" / "tokenizers")
    monkeypatch.setattr(T.P, "chunks_shard", PD.P.chunks_shard)
    from ecalls.embed import encode as E
    monkeypatch.setattr(E.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    PD.P.CHUNKS_DIR.mkdir(parents=True)

    spec = {"2010Q1": (1, "margin freight"), "2010Q4": (2, "guidance backlog"),
            "2015Q2": (3, "zzzlaterwordzzz")}
    for shard, (kd, w) in spec.items():
        texts = [f"{w} discussion of the quarter item {i} with more words to pad the chunk out"
                 for i in range(40)]
        pl.DataFrame({"keydevid": [kd] * 40, "chunk_id": [f"{shard}-{i:03d}" for i in range(40)],
                      "text": texts, "n_chars": [len(t) for t in texts]}
                     ).write_parquet(PD.P.chunks_shard(shard))
    T.fit("2010-12-31", vocab_size=400)
    return "ecalls-bpe-400-20101231"


# --- the window restriction ----------------------------------------------------------------------

def test_packing_reads_only_shards_inside_the_window(corpus):
    """`holdout_fraction=0` isolates the window from the holdout, so a failure here names one cause."""
    m = PD.pack_window("2010-12-31", corpus, context=64, quiet=True, holdout_fraction=0)
    assert m["shards"] == ["2010Q1", "2010Q4"]
    assert m["n_chunks_read"] == 80


def test_held_out_calls_are_excluded_from_the_pretraining_text(tmp_path, monkeypatch):
    """Held out means held out of both stages. A model that has seen a call's exact text can memorize
    it, and the retrieval benchmark draws its queries from those same calls."""
    monkeypatch.setattr(PD.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(PD.P, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(PD, "PRETRAIN_DIR", tmp_path / "processed" / "pretrain")
    monkeypatch.setattr(PD.P, "chunks_shard", lambda s: PD.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    monkeypatch.setattr(T.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    monkeypatch.setattr(T, "TOKENIZER_DIR", tmp_path / "processed" / "tokenizers")
    monkeypatch.setattr(T.P, "chunks_shard", PD.P.chunks_shard)
    from ecalls.embed import encode as E
    monkeypatch.setattr(E.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    PD.P.CHUNKS_DIR.mkdir(parents=True)

    from ecalls.train import holdout as HO
    ids = list(range(1, 400))
    held = [k for k in ids if HO.is_holdout(k)]
    trained = [k for k in ids if not HO.is_holdout(k)]
    assert held and trained, "the fixture needs calls on both sides of the split"

    rows = [{"keydevid": k, "chunk_id": f"{k}-{i:02d}",
             "text": " ".join([f"w{k}x"] * 60), "n_chars": 240}
            for k in ids for i in range(3)]
    pl.DataFrame(rows).write_parquet(PD.P.chunks_shard("2010Q1"))
    T.fit("2010-12-31", vocab_size=800, name="ho-tok")

    m = PD.pack_window("2010-12-31", "ho-tok", context=64, quiet=True)
    assert m["n_calls_held_out"] == len(held)
    assert m["holdout_fraction"] == HO.DEFAULT_FRACTION
    assert m["holdout_salt"] == HO.SALT

    tok = T.load("ho-tok")
    arr, _ = PD.load_window(m["name"])
    # Decoded text rather than token ids. With digits split, "w27x" tokenizes to w, 2, 7, x and every
    # one of those appears in other calls, so a token-set test can never isolate a marker. An earlier
    # version of this test did exactly that and reported twelve false leaks.
    decoded = " ".join(tok.decode([int(x) for x in row]) for row in np.asarray(arr))
    # Spaces are stripped on both sides because digit splitting renders "w27x" as "w 2 7 x" on decode,
    # which is the same artifact that made the first two versions of this assertion wrong.
    flat = decoded.replace(" ", "")
    leaked = [k for k in held if f"w{k}x" in flat]
    assert not leaked, f"text from held-out calls {leaked[:10]} reached the pretraining corpus"
    kept = [k for k in trained if f"w{k}x" in flat]
    assert kept, "no trainable call reached the corpus either, so the test proves nothing"


def test_holdout_zero_keeps_every_call(corpus):
    with_all = PD.pack_window("2010-12-31", corpus, context=64, quiet=True, holdout_fraction=0)
    assert with_all["n_calls_held_out"] == 0
    assert with_all["n_chunks_read"] == 80


def test_a_tokenizer_fit_after_the_vintage_cutoff_is_refused(corpus):
    """Every vintage must reuse the tokenizer fit on the earliest window. A later one would put
    vocabulary the vintage could not have seen into its own inputs."""
    T.fit("2015-12-31", vocab_size=400, name="late-tok")
    with pytest.raises(ValueError, match="after this vintage's cutoff"):
        PD.pack_window("2010-12-31", "late-tok", context=64, quiet=True)


def test_a_cutoff_before_the_corpus_raises_about_the_data_not_the_tokenizer(corpus):
    """Both problems are present at a 1990 cutoff, and the empty window is the one worth reporting, so
    the order of the two checks is itself the thing under test."""
    with pytest.raises(SystemExit, match="no chunk shard"):
        PD.pack_window("1990-01-01", corpus, context=64, quiet=True)


# --- call boundaries -----------------------------------------------------------------------------

def test_sequences_never_mix_two_calls(tmp_path, monkeypatch):
    """Adjacency inside a shard is an artifact of how the corpus was written, so packing across a call
    boundary would teach associations between unrelated companies.

    Written against `pack_window` rather than `pack_call`, because an earlier version of this test
    exercised `pack_call` in isolation and asserted `arr.shape[1] == 64`, which is trivially true. A
    mutant that flattened every call in a shard into one group passed all twelve tests.
    """
    monkeypatch.setattr(PD.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(PD.P, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(PD, "PRETRAIN_DIR", tmp_path / "processed" / "pretrain")
    monkeypatch.setattr(PD.P, "chunks_shard", lambda s: PD.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    monkeypatch.setattr(T.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    monkeypatch.setattr(T, "TOKENIZER_DIR", tmp_path / "processed" / "tokenizers")
    monkeypatch.setattr(T.P, "chunks_shard", PD.P.chunks_shard)
    from ecalls.embed import encode as E
    monkeypatch.setattr(E.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    PD.P.CHUNKS_DIR.mkdir(parents=True)

    # Two calls in ONE shard with disjoint alphabets, so a mixed sequence is detectable by token id.
    rows = []
    for kd, letter in ((11, "a"), (22, "b")):
        word = letter * 6
        for i in range(30):
            rows.append({"keydevid": kd, "chunk_id": f"{kd}-{i:03d}",
                         "text": " ".join([word] * 40), "n_chars": 7 * 40})
    pl.DataFrame(rows).write_parquet(PD.P.chunks_shard("2010Q1"))
    T.fit("2010-12-31", vocab_size=300, name="ab-tok")

    tok = T.load("ab-tok")
    ids_a = {i for i in tok.encode("a" * 6, add_special_tokens=False).ids}
    ids_b = {i for i in tok.encode("b" * 6, add_special_tokens=False).ids}
    assert ids_a and ids_b and not (ids_a & ids_b), "the fixture must give the two calls disjoint ids"

    m = PD.pack_window("2010-12-31", "ab-tok", context=48, quiet=True)
    arr, _ = PD.load_window(m["name"])
    assert m["n_sequences"] > 1
    mixed = 0
    for row in np.asarray(arr):
        present = set(int(x) for x in row)
        if (present & ids_a) and (present & ids_b):
            mixed += 1
    assert mixed == 0, f"{mixed} of {m['n_sequences']} sequences span both calls"


def test_a_chunk_longer_than_the_context_is_split_not_truncated():
    """Truncating would drop the end of a long answer, which is where a quantified statement sits, so a
    long chunk spills into the next sequence instead. The invariant is that what comes out is a prefix
    of what went in, in order, with only the ragged tail missing."""
    payload = [1] + [7] * 200 + [2]            # [CLS] + 200 tokens + [SEP]
    seqs = PD.pack_call([[7] * 200], context=32, cls_id=1, sep_id=2)
    assert len(seqs) == len(payload) // 32     # 202 // 32 = 6 full sequences
    flat = [t for s in seqs for t in s]
    assert flat == payload[:len(flat)]         # a prefix, in order, nothing reordered or dropped inside
    assert len(payload) - len(flat) < 32       # only the ragged tail is missing


def test_the_ragged_tail_of_a_call_is_dropped_rather_than_padded():
    seqs = PD.pack_call([[5] * 10], context=64, cls_id=1, sep_id=2)
    assert seqs == []                          # 12 tokens cannot fill a 64-token sequence
    for s in PD.pack_call([[5] * 200], context=64, cls_id=1, sep_id=2):
        assert len(s) == 64                    # every emitted sequence is exactly full


# --- determinism and the stored form -------------------------------------------------------------

def test_rebuilding_gives_a_byte_identical_file(corpus):
    """A vintage set whose training data cannot be reproduced cannot support a claim about what it saw."""
    m1 = PD.pack_window("2010-12-31", corpus, context=64, quiet=True)
    b1 = (PD.window_dir(m1["name"]) / "tokens.bin").read_bytes()
    m2 = PD.pack_window("2010-12-31", corpus, context=64, quiet=True)
    b2 = (PD.window_dir(m2["name"]) / "tokens.bin").read_bytes()
    assert b1 == b2
    assert m1["n_sequences"] == m2["n_sequences"]


def test_tokens_round_trip_through_the_memmap(corpus):
    m = PD.pack_window("2010-12-31", corpus, context=64, quiet=True)
    arr, meta = PD.load_window(m["name"])
    assert arr.shape == (meta["n_sequences"], 64)
    assert arr.dtype == np.uint16
    assert int(arr.max()) < meta["vocab_size"]
    assert meta["n_tokens"] == meta["n_sequences"] * 64


def test_uint16_is_exact_for_this_vocabulary(corpus):
    """32,768 fits in 16 bits. If the vocabulary ever grew past 65,536 this storage would wrap
    silently, so the assertion belongs in a test rather than a comment."""
    m = PD.pack_window("2010-12-31", corpus, context=64, quiet=True)
    assert m["vocab_size"] <= np.iinfo(np.uint16).max + 1


def test_the_sidecar_records_the_window_and_the_tokenizer(corpus):
    m = PD.pack_window("2010-12-31", corpus, context=64, quiet=True)
    meta = json.loads((PD.window_dir(m["name"]) / "_meta.json").read_text())
    assert meta["cutoff"] == "2010-12-31"
    assert meta["tokenizer"] == corpus
    assert meta["tokenizer_fit_cutoff"] == "2010-12-31"
    assert meta["extra_years"] == 0
    assert meta["n_shards"] == 2


def test_loading_a_missing_window_says_how_to_build_it(corpus):
    with pytest.raises(FileNotFoundError, match="Build it"):
        PD.load_window("not-built")


def test_no_partial_file_is_left_when_packing_fails(corpus, monkeypatch):
    """The write goes to a temp file and is renamed, so an interrupted build cannot leave a short
    corpus that looks complete."""
    orig = PD.pl.read_parquet
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("disk died")
        return orig(*a, **k)

    monkeypatch.setattr(PD.pl, "read_parquet", boom)
    with pytest.raises(RuntimeError):
        PD.pack_window("2010-12-31", corpus, context=64, quiet=True)
    name = PD.window_name("2010-12-31", 64, corpus)
    assert not (PD.window_dir(name) / "tokens.bin").exists()


# --- one packed corpus serving every vintage ------------------------------------------------------

def test_a_vintage_reads_the_prefix_belonging_to_its_window(corpus):
    """Each vintage's window contains every earlier one, so packing them separately would store the same
    text 7.7 times over, 46 GB against 6 GB. The array is in calendar order and no sequence crosses a
    call, so a window is exactly a prefix."""
    m = PD.pack_window("2020-12-31", corpus, context=64, quiet=True, holdout_fraction=0)
    assert [o["shard"] for o in m["shard_offsets"]] == ["2010Q1", "2010Q4", "2015Q2"]

    full, fmeta = PD.load_window(m["name"])
    early, emeta = PD.load_window(m["name"], cutoff="2010-12-31")
    assert emeta["n_sequences"] < fmeta["n_sequences"]
    assert emeta["sliced_to_cutoff"] == "2010-12-31"
    assert emeta["n_tokens"] == emeta["n_sequences"] * 64
    # the prefix is the same bytes as the front of the full array, not a re-derivation
    assert np.array_equal(np.asarray(early), np.asarray(full)[:emeta["n_sequences"]])


def test_the_prefix_ends_exactly_at_a_shard_boundary(corpus):
    m = PD.pack_window("2020-12-31", corpus, context=64, quiet=True, holdout_fraction=0)
    ends = {o["shard"]: o["end"] for o in m["shard_offsets"]}
    _, emeta = PD.load_window(m["name"], cutoff="2010-12-31")
    assert emeta["n_sequences"] == ends["2010Q4"]


def test_a_prefix_containing_no_shard_raises_rather_than_training_on_nothing(corpus):
    """A vintage silently training on an empty array is the failure this module exists to prevent."""
    m = PD.pack_window("2020-12-31", corpus, context=64, quiet=True, holdout_fraction=0)
    with pytest.raises(ValueError, match="no shard of"):
        PD.load_window(m["name"], cutoff="1995-01-01")


def test_a_corpus_packed_without_offsets_refuses_a_prefix(corpus):
    """Rather than guessing, since a wrong prefix trains a vintage on text from after its cutoff."""
    m = PD.pack_window("2010-12-31", corpus, context=64, quiet=True, holdout_fraction=0)
    meta = json.loads((PD.window_dir(m["name"]) / "_meta.json").read_text())
    del meta["shard_offsets"]
    with pytest.raises(ValueError, match="rebuild it|Rebuild it"):
        PD.sequences_up_to(meta, "2010-12-31")


# --- the inverted pack, which is how a vintage is evaluated --------------------------------------

def _split_fixture(tmp_path, monkeypatch):
    """A corpus whose calls straddle the holdout split, with a per-call text marker."""
    monkeypatch.setattr(PD.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(PD.P, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(PD, "PRETRAIN_DIR", tmp_path / "processed" / "pretrain")
    monkeypatch.setattr(PD.P, "chunks_shard", lambda s: PD.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    monkeypatch.setattr(T.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    monkeypatch.setattr(T, "TOKENIZER_DIR", tmp_path / "processed" / "tokenizers")
    monkeypatch.setattr(T.P, "chunks_shard", PD.P.chunks_shard)
    from ecalls.embed import encode as E
    monkeypatch.setattr(E.P, "CHUNKS_DIR", PD.P.CHUNKS_DIR)
    PD.P.CHUNKS_DIR.mkdir(parents=True)

    from ecalls.train import holdout as HO
    ids = list(range(1, 400))
    rows = [{"keydevid": k, "chunk_id": f"{k}-{i:02d}",
             "text": " ".join([f"w{k}x"] * 60), "n_chars": 240}
            for k in ids for i in range(3)]
    pl.DataFrame(rows).write_parquet(PD.P.chunks_shard("2010Q1"))
    T.fit("2010-12-31", vocab_size=800, name="inv-tok")
    return ids, [k for k in ids if HO.is_holdout(k)], "inv-tok"


def test_the_inverted_pack_is_the_exact_complement_of_the_training_pack(tmp_path, monkeypatch):
    """Not merely "some calls": the two packs must partition the corpus with nothing lost or shared.

    An evaluation set that overlaps training reports memorization, and one that silently drops calls
    makes a loss comparison rest on a different sample per vintage.
    """
    ids, held, tokname = _split_fixture(tmp_path, monkeypatch)
    train = PD.pack_window("2010-12-31", tokname, context=64, quiet=True)
    evalm = PD.pack_window("2010-12-31", tokname, context=64, quiet=True, holdout_only=True)

    assert train["name"] != evalm["name"], "both packs would otherwise overwrite one file"
    assert evalm["name"].endswith("-holdout")
    # n_calls_held_out counts the calls each pack discarded, so the two must swap.
    assert train["n_calls_held_out"] == len(held)
    assert evalm["n_calls_held_out"] == len(ids) - len(held)
    assert train["n_chunks_read"] + evalm["n_chunks_read"] == 3 * len(ids)
    assert evalm["holdout_only"] is True and train["holdout_only"] is False


def test_no_call_appears_in_both_packs(tmp_path, monkeypatch):
    """Checked against the decoded text, not the manifest counts, because the counts agreeing is what
    a wrongly inverted filter would also produce."""
    ids, held, tokname = _split_fixture(tmp_path, monkeypatch)
    train = PD.pack_window("2010-12-31", tokname, context=64, quiet=True)
    evalm = PD.pack_window("2010-12-31", tokname, context=64, quiet=True, holdout_only=True)
    tok = T.load(tokname)

    def markers(meta):
        arr, _ = PD.load_window(meta["name"])
        flat = " ".join(tok.decode([int(x) for x in row]) for row in np.asarray(arr))
        flat = flat.replace(" ", "")      # digits are split, so "w27x" decodes as "w 2 7 x"
        return {k for k in ids if f"w{k}x" in flat}

    in_train, in_eval = markers(train), markers(evalm)
    assert in_train and in_eval, "one pack is empty, so the test proves nothing"
    assert not (in_train & in_eval), f"calls in both packs: {sorted(in_train & in_eval)[:10]}"
    assert in_eval <= set(held), "the evaluation pack holds a call that training was allowed to see"


def test_the_inverted_pack_refuses_a_zero_holdout_fraction(tmp_path, monkeypatch):
    """Otherwise the filter is skipped entirely and the whole corpus is written as the held-out set."""
    _, _, tokname = _split_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="whole corpus"):
        PD.pack_window("2010-12-31", tokname, context=64, quiet=True,
                       holdout_fraction=0, holdout_only=True)


# --- a borrowed vocabulary, for domain-adaptive pretraining --------------------------------------

def test_a_borrowed_vocabulary_must_not_post_date_the_vintage(tmp_path, monkeypatch):
    """Continuing from a 2024 model into a 2015 vintage would carry nine years of later vocabulary
    into it, which is the same leak D20 split digits to prevent."""
    _, _, tokname = _split_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="after this vintage"):
        PD.pack_window("2015-12-31", "registry:chrono-bert-v1-20241231", context=64, quiet=True,
                       holdout_fraction=0)


def test_the_window_name_of_a_borrowed_vocabulary_has_no_colon(tmp_path, monkeypatch):
    """`registry:x` is a legal directory name and a miserable one, and ambiguous on some filesystems."""
    n = PD.window_name("2015-12-31", 768, "registry:chrono-bert-v1-20151231")
    assert ":" not in n
    assert n.startswith("borrowed-")


def test_a_vocabulary_too_large_for_uint16_is_refused(tmp_path, monkeypatch):
    """The packed array is uint16, exact only below 65,536. A larger vocabulary would wrap silently
    and every id above the limit would decode to a different token."""
    _, _, tokname = _split_fixture(tmp_path, monkeypatch)

    class Huge:
        def get_vocab_size(self):
            return 70_000

        def token_to_id(self, t):
            return {"[CLS]": 1, "[SEP]": 2}.get(t)

    monkeypatch.setattr(PD.T, "load_any", lambda name: Huge())
    monkeypatch.setattr(PD.T, "special_ids", lambda tok, name: (1, 2))
    with pytest.raises(ValueError, match="does not fit"):
        PD.pack_window("2010-12-31", tokname, context=64, quiet=True, holdout_fraction=0)
