"""Edge cases for the bag-of-words against dense diagnostic.

The metric is where this can go wrong quietly. A scorer that returns a constant must score *worst*,
not accidentally well, and a rank computation that breaks ties optimistically hands exactly that free
win to a broken dense model. So the tie behaviour is pinned by a test rather than assumed.
"""
import numpy as np
import polars as pl
import pytest

from ecalls.diagnostics import bow_vs_dense as B


# --- the metric ----------------------------------------------------------------------------------

def test_a_perfect_scorer_scores_one():
    s = np.array([[9.0, 1, 1], [1, 9, 1]])
    m = B.rank_metrics(s, np.array([0, 1]))
    assert m["recall_at_1"] == 1.0 and m["mrr"] == 1.0 and m["median_rank"] == 1.0


def test_a_worst_case_scorer_ranks_gold_last():
    s = np.array([[1.0, 9, 8], [1, 9, 8]])
    m = B.rank_metrics(s, np.array([0, 0]))
    assert m["recall_at_1"] == 0.0
    assert m["median_rank"] == 3.0
    assert m["mrr"] == pytest.approx(1 / 3)


def test_a_constant_scorer_is_scored_pessimistically_not_generously():
    """The test that protects the whole comparison. Under optimistic tie-breaking a scorer that
    returns the same number everywhere would report recall@1 of 1.0 and look like the winner."""
    s = np.ones((3, 5))
    m = B.rank_metrics(s, np.array([0, 1, 2]))
    assert m["recall_at_1"] == 0.0
    assert m["median_rank"] == 5.0
    assert m["mrr"] == pytest.approx(0.2)


def test_recall_at_5_counts_rank_five_as_a_hit():
    s = np.zeros((1, 6))
    s[0] = [1, 6, 5, 4, 3, 2]          # gold at index 0 scores 1, so five candidates beat it
    m = B.rank_metrics(s, np.array([0]))
    assert m["median_rank"] == 6.0 and m["recall_at_5"] == 0.0
    s[0, 0] = 2.5                       # now 6, 5, 4 and 3 beat it, so rank 5
    m = B.rank_metrics(s, np.array([0]))
    assert m["median_rank"] == 5.0 and m["recall_at_5"] == 1.0


def test_the_mask_restricts_the_candidate_pool():
    """Gold is beaten by a candidate the query is not allowed to see, so masking must rescue it."""
    s = np.array([[5.0, 99.0]])
    gold = np.array([0])
    assert B.rank_metrics(s, gold)["recall_at_1"] == 0.0
    mask = np.array([[True, False]])
    assert B.rank_metrics(s, gold, mask)["recall_at_1"] == 1.0


def test_masking_out_the_gold_raises_rather_than_reporting_a_perfect_rank():
    """Found by this test. With the gold masked to -inf every visible candidate counted as better,
    the tie set came back empty, and `ties.sum() - 1` subtracted one from the rank, so an
    unanswerable query scored rank 1. A silent perfect hit is the worst possible failure for a
    retrieval metric, because it inflates whichever scorer the caller happened to be testing."""
    s = np.array([[5.0, 1.0]])
    mask = np.array([[False, True]])
    with pytest.raises(ValueError, match="unanswerable"):
        B.rank_metrics(s, np.array([0]), mask)


def test_within_call_mask_pairs_queries_to_their_own_call():
    q = np.array([10, 10, 20])
    a = np.array([10, 20, 20, 30])
    m = B.within_call_mask(q, a)
    assert m.tolist() == [[True, False, False, False],
                          [True, False, False, False],
                          [False, True, True, False]]


# --- BM25 ----------------------------------------------------------------------------------------

def test_bm25_prefers_the_document_sharing_rare_terms():
    docs = [tok.split() for tok in ["alpha beta gamma", "delta epsilon zeta", "alpha alpha beta"]]
    q = [["gamma"]]
    s = B.bm25_scores(q, docs)
    assert s[0].argmax() == 0


def test_bm25_gives_zero_to_a_query_with_no_shared_terms():
    docs = [["alpha"], ["beta"]]
    assert B.bm25_scores([["omega"]], docs).sum() == 0.0


def test_bm25_discounts_a_term_that_appears_in_every_document():
    """A term in every document carries no signal, and BM25's idf must reflect that."""
    docs = [["the", "alpha"], ["the", "beta"], ["the", "gamma"]]
    s = B.bm25_scores([["the"]], docs)
    rare = B.bm25_scores([["alpha"]], docs)
    assert s.max() < rare.max()


def test_bm25_normalizes_by_document_length():
    """Two documents with one match each, differing only in length. The longer is diluted."""
    docs = [["alpha"], ["alpha"] + ["pad"] * 50]
    s = B.bm25_scores([["alpha"]], docs)
    assert s[0, 0] > s[0, 1]


def test_tokenize_lowercases_and_drops_punctuation():
    assert B.tokenize("Q3 EPS was $1.25, up 12%!") == ["q3", "eps", "was", "1", "25", "up", "12"]


# --- pair construction --------------------------------------------------------------------------

def _components(rows):
    """Build a minimal component frame, matching the columns `qa_pairs` reads."""
    return pl.DataFrame(rows, schema={
        "keydevid": pl.Int64, "companyid": pl.Int64, "component_order": pl.Int32,
        "component_id": pl.Int64, "section": pl.Utf8, "speaker_type": pl.Utf8,
        "speaker_name": pl.Utf8, "text": pl.Utf8, "n_chars": pl.Int32})


@pytest.fixture
def one_call(tmp_path, monkeypatch):
    """One call with three analyst-executive exchanges plus noise that must be filtered out."""
    long = "x" * 400
    rows = []
    order = 0

    def add(cid, sp, text, section="qa", call=1):
        nonlocal order
        order += 1
        rows.append({"keydevid": call, "companyid": 7, "component_order": order,
                     "component_id": cid, "section": section, "speaker_type": sp,
                     "speaker_name": sp, "text": text, "n_chars": len(text)})

    add(100, "Operator", long, section="operator")
    add(101, "Executives", long, section="presentation")   # presentation is out of scope
    for i in range(6):
        add(200 + 2 * i, "Analysts", f"question {i} {long}")
        add(201 + 2 * i, "Executives", f"answer {i} {long}")
    add(900, "Analysts", "thanks")                          # too short, must be dropped
    add(901, "Executives", "sure")                          # too short, must be dropped

    monkeypatch.setattr(B.P, "components_shard", lambda s: tmp_path / f"components_{s}.parquet")
    _components(rows).write_parquet(B.P.components_shard("2019Q3"))
    return rows


def test_pairs_link_each_question_to_the_answer_that_follows_it(one_call):
    q, a = B.qa_pairs("2019Q3", n_calls=10, min_pool=5)
    assert q.height == 6
    assert sorted(zip(q["query_id"], q["answer_id"])) == [
        (200, 201), (202, 203), (204, 205), (206, 207), (208, 209), (210, 211)]


def test_short_turns_and_non_qa_sections_are_excluded(one_call):
    q, a = B.qa_pairs("2019Q3", n_calls=10, min_pool=5)
    assert 900 not in q["query_id"].to_list()
    assert 101 not in a["answer_id"].to_list()     # the presentation turn
    assert 901 not in a["answer_id"].to_list()     # too short


def test_a_call_with_too_small_a_pool_is_dropped_entirely(one_call):
    """With min_pool above the call's 6 answers, nothing survives. A 4-answer pool makes recall@1 a
    coin flip, which would be reported as a measurement."""
    q, a = B.qa_pairs("2019Q3", n_calls=10, min_pool=99)
    assert q.height == 0 and a.height == 0


def test_a_question_whose_own_answer_was_filtered_out_is_dropped(tmp_path, monkeypatch):
    """Otherwise the gold document is absent from the pool and the query is unanswerable, which would
    depress every scorer equally and silently."""
    long = "y" * 400
    rows = [{"keydevid": 1, "companyid": 7, "component_order": i + 1, "component_id": cid,
             "section": "qa", "speaker_type": sp, "speaker_name": sp, "text": t, "n_chars": len(t)}
            for i, (cid, sp, t) in enumerate([
                (1, "Analysts", "q " + long), (2, "Executives", "short"),      # answer too short
                (3, "Analysts", "q2 " + long), (4, "Executives", "a2 " + long),
                (5, "Analysts", "q3 " + long), (6, "Executives", "a3 " + long),
                (7, "Analysts", "q4 " + long), (8, "Executives", "a4 " + long),
                (9, "Analysts", "q5 " + long), (10, "Executives", "a5 " + long),
                (11, "Analysts", "q6 " + long), (12, "Executives", "a6 " + long)])]
    monkeypatch.setattr(B.P, "components_shard", lambda s: tmp_path / f"c_{s}.parquet")
    _components(rows).write_parquet(B.P.components_shard("2019Q3"))
    q, a = B.qa_pairs("2019Q3", n_calls=10, min_pool=5)
    assert 1 not in q["query_id"].to_list()
    assert set(q["answer_id"].to_list()).issubset(set(a["answer_id"].to_list()))


def test_analyst_followed_by_another_analyst_is_not_a_pair(tmp_path, monkeypatch):
    long = "z" * 400
    seq = [("Analysts", "q1"), ("Analysts", "q2"), ("Executives", "a1")]
    seq += [("Analysts", f"q{i}") if i % 2 else ("Executives", f"a{i}") for i in range(3, 15)]
    rows = [{"keydevid": 1, "companyid": 7, "component_order": i + 1, "component_id": i + 1,
             "section": "qa", "speaker_type": sp, "speaker_name": sp,
             "text": f"{t} {long}", "n_chars": len(t) + 401}
            for i, (sp, t) in enumerate(seq)]
    monkeypatch.setattr(B.P, "components_shard", lambda s: tmp_path / f"c2_{s}.parquet")
    _components(rows).write_parquet(B.P.components_shard("2019Q3"))
    q, _ = B.qa_pairs("2019Q3", n_calls=10, min_pool=5)
    assert 1 not in q["query_id"].to_list()        # followed by another analyst
    assert 2 in q["query_id"].to_list()            # followed by an executive


def test_pairs_never_cross_a_call_boundary(tmp_path, monkeypatch):
    """The shift is windowed by keydevid. Without the window the last turn of one call would be
    paired with the first turn of the next, which sorts adjacently and would look plausible."""
    long = "w" * 400
    rows = []
    for call in (1, 2):
        for i in range(6):
            rows.append({"keydevid": call, "companyid": call, "component_order": i * 2 + 1,
                         "component_id": call * 1000 + i * 2, "section": "qa",
                         "speaker_type": "Analysts", "speaker_name": "A",
                         "text": f"q{i} {long}", "n_chars": 405})
            rows.append({"keydevid": call, "companyid": call, "component_order": i * 2 + 2,
                         "component_id": call * 1000 + i * 2 + 1, "section": "qa",
                         "speaker_type": "Executives", "speaker_name": "E",
                         "text": f"a{i} {long}", "n_chars": 405})
    monkeypatch.setattr(B.P, "components_shard", lambda s: tmp_path / f"c3_{s}.parquet")
    _components(rows).write_parquet(B.P.components_shard("2019Q3"))
    q, a = B.qa_pairs("2019Q3", n_calls=10, min_pool=5)
    call_of = dict(zip(a["answer_id"].to_list(), a["keydevid"].to_list()))
    for qid, aid, kd in zip(q["query_id"], q["answer_id"], q["keydevid"]):
        assert call_of[aid] == kd, (qid, aid)


# --- the hybrid's row standardization -----------------------------------------------------------

def test_zscore_rows_standardizes_each_row_independently():
    S = np.array([[1.0, 2, 3], [100.0, 200, 300]])
    Z = B.zscore_rows(S)
    assert np.allclose(Z[0], Z[1])                    # scale-free, so both rows land identically
    assert np.allclose(Z.mean(axis=1), 0)
    assert np.allclose(Z.std(axis=1), 1)


def test_zscore_rows_returns_zeros_for_a_flat_row_rather_than_dividing_by_zero():
    """A query sharing no term with any candidate gives BM25 a row of zeros. Dividing by its zero
    standard deviation would put NaN into the hybrid and poison every ranking."""
    Z = B.zscore_rows(np.array([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]]))
    assert np.array_equal(Z[0], np.zeros(3))
    assert np.isfinite(Z).all()


def test_zscore_rows_preserves_the_ranking_within_a_row():
    S = np.array([[3.0, 1.0, 2.0]])
    assert np.array_equal(np.argsort(-B.zscore_rows(S)[0]), np.argsort(-S[0]))


def test_the_hybrid_is_not_just_the_stronger_scorer_when_scales_differ_wildly():
    """The reason to z-score. Raw BM25 sums dwarf cosines, so a raw sum would be BM25 with noise."""
    bm = np.array([[0.0, 50.0]])
    dn = np.array([[0.9, 0.1]])
    raw = bm + dn
    z = B.zscore_rows(bm) + B.zscore_rows(dn)
    assert raw.argmax() == 1                          # BM25 wins outright on the raw sum
    # after standardizing, the two disagreeing scorers cancel exactly rather than one dominating
    assert z[0, 0] == pytest.approx(-z[0, 1])
    assert abs(z[0, 0] - z[0, 1]) < abs(raw[0, 0] - raw[0, 1])


# --- the holdout filter, which is what makes a comparison possible -------------------------------

def _split_components(tmp_path, monkeypatch, n_calls=120):
    """A components shard whose calls straddle the holdout split."""
    monkeypatch.setattr(B.P, "components_shard",
                        lambda s: tmp_path / f"components_{s}.parquet")
    rows = []
    for c in range(1, n_calls + 1):
        order = 0
        for i in range(8):
            rows.append({"keydevid": c, "companyid": c * 10, "component_order": order,
                         "component_id": f"{c}-q{i}", "section": "qa",
                         "speaker_type": B.QUESTION_SPEAKER, "speaker_name": "an analyst",
                         "text": f"question {i} about margin and freight in call {c} " * 4,
                         "n_chars": 200})
            order += 1
            rows.append({"keydevid": c, "companyid": c * 10, "component_order": order,
                         "component_id": f"{c}-a{i}", "section": "qa",
                         "speaker_type": B.ANSWER_SPEAKER, "speaker_name": "the cfo",
                         "text": f"answer {i} on pricing and volume in call {c} " * 4,
                         "n_chars": 200})
            order += 1
    pl.DataFrame(rows).write_parquet(B.P.components_shard("2010Q1"))


def test_the_two_holdout_populations_partition_the_calls(tmp_path, monkeypatch):
    """Not overlapping and not losing calls: a comparison rests on the two sides being the split."""
    from ecalls.train import holdout as HO
    _split_components(tmp_path, monkeypatch)
    q_all, _ = B.qa_pairs("2010Q1", 10**9)
    q_ho, _ = B.qa_pairs("2010Q1", 10**9, holdout=True)
    q_tr, _ = B.qa_pairs("2010Q1", 10**9, holdout=False)

    all_calls = set(q_all["keydevid"].to_list())
    ho, tr = set(q_ho["keydevid"].to_list()), set(q_tr["keydevid"].to_list())
    assert ho and tr, "one side is empty, so the test proves nothing"
    assert not (ho & tr)
    assert ho | tr == all_calls
    assert all(HO.is_holdout(k) for k in ho)
    assert not any(HO.is_holdout(k) for k in tr)


def test_the_default_keeps_every_call_so_the_reference_numbers_still_reproduce(tmp_path, monkeypatch):
    """bge-small's 0.5378 was measured with no filter. Changing the default would silently move it."""
    _split_components(tmp_path, monkeypatch)
    q_none, a_none = B.qa_pairs("2010Q1", 10**9)
    q_ho, _ = B.qa_pairs("2010Q1", 10**9, holdout=True)
    assert q_none.height > q_ho.height


def test_a_holdout_filter_that_empties_the_shard_raises(tmp_path, monkeypatch):
    """A recall computed on nothing is a number, and it would be reported as one."""
    monkeypatch.setattr(B.P, "components_shard",
                        lambda s: tmp_path / f"components_{s}.parquet")
    from ecalls.train import holdout as HO
    trainable = next(k for k in range(1, 10000) if not HO.is_holdout(k))
    rows = [{"keydevid": trainable, "companyid": 1, "component_order": i,
             "component_id": f"c{i}", "section": "qa",
             "speaker_type": B.QUESTION_SPEAKER if i % 2 == 0 else B.ANSWER_SPEAKER,
             "speaker_name": "x", "text": "margin and freight " * 20, "n_chars": 380}
            for i in range(12)]
    pl.DataFrame(rows).write_parquet(B.P.components_shard("2010Q1"))
    with pytest.raises(ValueError, match="no call survives"):
        B.qa_pairs("2010Q1", 10**9, holdout=True)


# --- the fusion scan and the per-question ranks -----------------------------------------------------

def _fake_run(monkeypatch, n_calls=6, per_call=5, seed=0):
    """`run` on synthetic text with a stand-in dense model, so the scan and the ranks can be checked
    against the scorers they must reproduce."""
    rng = np.random.default_rng(seed)
    words = ["margin", "freight", "guidance", "inventory", "pricing", "capex", "churn", "backlog"]
    q, a = [], []
    for c in range(n_calls):
        for i in range(per_call):
            topic = words[(c + i) % len(words)]
            q.append({"keydevid": c, "answer_id": f"a{c}-{i}", "text": f"what about {topic} {c} {i}"})
            a.append({"keydevid": c, "answer_id": f"a{c}-{i}",
                      "text": f"{topic} moved in quarter {i} for call {c} " + " ".join(
                          rng.choice(words, 4))})
    queries, answers = pl.DataFrame(q), pl.DataFrame(a)
    monkeypatch.setattr(B, "qa_pairs", lambda *args, **kw: (queries, answers))

    def dense(qt, at, model_id, device=None):
        Vq, Va = rng.normal(size=(len(qt), 8)), rng.normal(size=(len(at), 8))
        return Vq, Va
    monkeypatch.setattr(B, "dense_matrix", dense)
    return B.run("2019Q3", n_calls, "fake", keep_ranks=True, fusion_scan=True)


def test_the_scan_reproduces_the_hybrid_and_its_two_ends(monkeypatch):
    """Weight 0.5 ranks exactly as the unweighted sum, 0 as dense alone and 1 as BM25 alone. If any of
    the three disagrees, the scan is measuring something other than the scorers it is compared with."""
    res = _fake_run(monkeypatch)
    scan, sc = res["fusion_scan"], res["scorers"]
    for pool in ("within_call", "cross_call"):
        assert scan["0.5"][pool] == sc["hybrid_bm25_dense"][pool]
        assert scan["0.0"][pool] == sc["dense"][pool]
        assert scan["1.0"][pool] == sc["bm25"][pool]
    assert list(scan) == [str(w) for w in B.FUSION_WEIGHTS]


def test_the_saved_ranks_reproduce_the_reported_metrics(monkeypatch, tmp_path):
    """A bootstrap reads the ranks file, so it must agree exactly with the JSON it sits beside."""
    res = _fake_run(monkeypatch)
    path = B.save_ranks(res, tmp_path / "r.npz")
    z = np.load(path)
    assert len(z["keydevid"]) == res["n_queries"]
    for name in ("tfidf", "bm25", "dense", "hybrid_bm25_dense", "fused_w0.3"):
        for pool, key in (("within", "within_call"), ("cross", "cross_call")):
            got = B.metrics_from_ranks(z[f"{name}__{pool}"])
            want = (res["scorers"][name][key] if name in res["scorers"]
                    else res["fusion_scan"]["0.3"][key])
            assert got == want, (name, pool)


def test_ranks_are_not_returned_unless_asked(monkeypatch):
    rng = np.random.default_rng(0)
    res = _fake_run(monkeypatch)
    assert "_ranks" in res
    monkeypatch.setattr(B, "dense_matrix",
                        lambda qt, at, m, device=None: (rng.normal(size=(len(qt), 4)),
                                                        rng.normal(size=(len(at), 4))))
    res2 = B.run("2019Q3", 6, "fake")
    assert "_ranks" not in res2 and "fusion_scan" not in res2


def test_rank_vector_and_rank_metrics_agree():
    rng = np.random.default_rng(3)
    S = rng.normal(size=(20, 30))
    gold = rng.integers(0, 30, 20)
    assert B.metrics_from_ranks(B.rank_vector(S, gold)) == B.rank_metrics(S, gold)


def test_dense_matrix_applies_the_query_and_passage_prefixes(monkeypatch):
    """E5 needs "query: " and "passage: ", BGE only the query instruction, so each side reads its own."""
    import numpy as np
    from ecalls.chunk import registry as R
    from ecalls.embed import encode as E
    from ecalls.diagnostics import bow_vs_dense as BV
    seen = []
    entry = R.Entry(model_id="stub", kind=R.ATEMPORAL, source="huggingface:stub/stub", dim=2,
                    training_cutoff=None, pooling="mean", normalized=True,
                    extra={"query_prefix": "query: ", "passage_prefix": "passage: "})
    class Reg:
        def by_name(self, name):
            return entry
    monkeypatch.setattr(BV.R, "load", lambda: Reg())
    monkeypatch.setattr(E, "load_encoder", lambda e, d: (None, None, "cpu"))
    monkeypatch.setattr(E, "encode_texts", lambda texts, *a, **k: seen.append(list(texts)) or np.zeros((len(texts), 2)))
    BV.dense_matrix(["q1"], ["d1", "d2"], "stub")
    assert seen == [["query: q1"], ["passage: d1", "passage: d2"]]
