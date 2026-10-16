"""Edge cases for the contrastive pairs.

The two filters are what the methodology notes flag as easy to get wrong, because the pair construction
code is separate from the corpus filter: a pair from after the vintage's cutoff, or from a held-out call,
both train a model on text it must not see, and both are invisible afterwards.
"""
import datetime as dt
import json

import polars as pl
import pytest

from ecalls.train import holdout as HO
from ecalls.train import pairs as PR

D = dt.date
LONG = "y" * 400


def _components(rows):
    return pl.DataFrame(rows, schema={
        "keydevid": pl.Int64, "companyid": pl.Int64, "component_order": pl.Int32,
        "component_id": pl.Int64, "section": pl.Utf8, "speaker_type": pl.Utf8,
        "speaker_name": pl.Utf8, "text": pl.Utf8, "n_chars": pl.Int32})


def _call_rows(kd, base_id, tag):
    """One call with six analyst-executive exchanges, tagged so leakage is detectable in the text."""
    rows, order = [], 0
    for i in range(6):
        for sp, who in (("Analysts", "q"), ("Executives", "a")):
            order += 1
            txt = f"{tag}{who}{i} {LONG}"
            rows.append({"keydevid": kd, "companyid": kd, "component_order": order,
                         "component_id": base_id + order, "section": "qa", "speaker_type": sp,
                         "speaker_name": sp, "text": txt, "n_chars": len(txt)})
    return rows


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Two shards, one inside a 2010 window and one after it, each with a held-out and a trained call."""
    monkeypatch.setattr(PR.P, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(PR, "PAIRS_DIR", tmp_path / "processed" / "pairs")
    monkeypatch.setattr(PR.P, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(PR.P, "chunks_shard", lambda s: PR.P.CHUNKS_DIR / f"chunks_{s}.parquet")
    from ecalls.diagnostics import bow_vs_dense as B
    from ecalls.embed import encode as E
    monkeypatch.setattr(E.P, "CHUNKS_DIR", PR.P.CHUNKS_DIR)
    monkeypatch.setattr(E.P, "chunks_shard", PR.P.chunks_shard)
    monkeypatch.setattr(B.P, "components_shard",
                        lambda s: PR.P.CHUNKS_DIR / f"components_{s}.parquet")
    PR.P.CHUNKS_DIR.mkdir(parents=True)

    # pick real ids on each side of the split so the fixture exercises both
    held = next(k for k in range(1, 5000) if HO.is_holdout(k))
    kept = next(k for k in range(1, 5000) if not HO.is_holdout(k))

    for shard, tag in (("2010Q2", "early"), ("2015Q2", "later")):
        rows = _call_rows(kept, 1000, f"{tag}kept") + _call_rows(held, 2000, f"{tag}held")
        _components(rows).write_parquet(PR.P.CHUNKS_DIR / f"components_{shard}.parquet")
        # the chunk shard only has to exist for shards_up_to to see the quarter
        pl.DataFrame({"chunk_id": ["x"], "text": ["x"], "n_chars": [1]}).write_parquet(
            PR.P.chunks_shard(shard))
    return {"held": held, "kept": kept}


# --- the window filter ---------------------------------------------------------------------------

def test_pairs_come_only_from_inside_the_window(corpus):
    m = PR.build_pairs("2010-12-31", quiet=True)
    assert m["shards"] == ["2010Q2"]
    df, _ = PR.load_pairs(m["name"])
    assert df.height > 0
    assert not any("later" in t for t in df["text"].to_list())
    assert not any("later" in t for t in df["answer_text"].to_list())


def test_a_wider_cutoff_reaches_the_later_shard(corpus):
    m = PR.build_pairs("2015-12-31", quiet=True)
    assert m["shards"] == ["2010Q2", "2015Q2"]
    df, _ = PR.load_pairs(m["name"])
    assert any("later" in t for t in df["text"].to_list())


def test_a_cutoff_before_the_corpus_raises(corpus):
    with pytest.raises(SystemExit, match="no chunk shard"):
        PR.build_pairs("1990-01-01", quiet=True)


# --- the holdout filter --------------------------------------------------------------------------

def test_held_out_calls_contribute_no_pairs(corpus):
    """A pair from a held-out call trains the model on the exact text the benchmark queries."""
    m = PR.build_pairs("2015-12-31", quiet=True)
    df, _ = PR.load_pairs(m["name"])
    assert corpus["held"] not in df["keydevid"].to_list()
    assert corpus["kept"] in df["keydevid"].to_list()
    assert not any("held" in t for t in df["text"].to_list())
    assert m["n_calls_held_out"] == 2          # one per shard


def test_disabling_the_holdout_admits_those_calls(corpus):
    m = PR.build_pairs("2015-12-31", holdout_fraction=0, quiet=True)
    df, _ = PR.load_pairs(m["name"])
    assert corpus["held"] in df["keydevid"].to_list()
    assert m["n_calls_held_out"] == 0


def test_the_sidecar_records_both_filters(corpus):
    m = PR.build_pairs("2010-12-31", quiet=True)
    meta = json.loads((PR.pairs_dir(m["name"]) / "_meta.json").read_text())
    assert meta["cutoff"] == "2010-12-31"
    assert meta["extra_years"] == 0
    assert meta["holdout_salt"] == HO.SALT
    assert meta["holdout_fraction"] == HO.DEFAULT_FRACTION
    assert "bow_vs_dense" in meta["extraction_rule"]


# --- the pairs themselves ------------------------------------------------------------------------

def test_every_pair_carries_its_answer_text(corpus):
    m = PR.build_pairs("2010-12-31", quiet=True)
    df, _ = PR.load_pairs(m["name"])
    assert df["answer_text"].null_count() == 0
    assert all(len(t) >= 120 for t in df["answer_text"].to_list())


def test_a_query_and_its_answer_come_from_the_same_call(corpus):
    """The pair is only a positive because the answer responds to the question. A pair crossing calls
    would be a false positive that actively teaches the wrong thing."""
    m = PR.build_pairs("2015-12-31", quiet=True)
    df, _ = PR.load_pairs(m["name"])
    for q, a in zip(df["text"].to_list(), df["answer_text"].to_list()):
        assert q.split()[0][:-2] == a.split()[0][:-2]    # same tag prefix, so same call


def test_the_build_is_deterministic(corpus):
    a = PR.build_pairs("2015-12-31", quiet=True)
    da, _ = PR.load_pairs(a["name"])
    b = PR.build_pairs("2015-12-31", quiet=True)
    db, _ = PR.load_pairs(b["name"])
    assert da.equals(db)


def test_pool_sizes_reports_in_call_negative_counts(corpus):
    m = PR.build_pairs("2015-12-31", quiet=True)
    df, _ = PR.load_pairs(m["name"])
    pools = PR.pool_sizes(df)
    assert pools.height == df["keydevid"].n_unique()
    assert pools["n_pairs"].min() >= 1


def test_loading_missing_pairs_says_how_to_build_them(corpus):
    with pytest.raises(FileNotFoundError, match="Build them"):
        PR.load_pairs("not-built")
