"""The call-level bootstrap behind the paper's intervals."""
import numpy as np
import pytest

from ecalls.diagnostics import seed_summary as S


def test_a_constant_statistic_has_a_zero_width_interval():
    calls = np.repeat(np.arange(10), 5)
    draws = S.call_bootstrap([(calls, np.full(50, 0.3))], n_boot=200)
    assert np.allclose(draws, 0.3)


def test_the_bootstrap_centres_on_the_question_weighted_mean():
    rng = np.random.default_rng(0)
    calls = np.repeat(np.arange(200), rng.integers(3, 12, 200))
    vals = rng.random(len(calls))
    draws = S.call_bootstrap([(calls, vals)], n_boot=2000)
    assert abs(draws.mean() - vals.mean()) < 0.005


def test_resampling_calls_is_wider_than_resampling_questions_when_calls_cluster():
    """Questions in one call share an outcome here, so the effective sample is the calls. A
    question-level bootstrap would report about a third of the true width."""
    calls = np.repeat(np.arange(40), 10)
    vals = np.repeat((np.arange(40) % 2).astype(float), 10)
    by_call = S.call_bootstrap([(calls, vals)], n_boot=2000)
    by_question = S.call_bootstrap([(np.arange(400), vals)], n_boot=2000)
    assert by_call.std() > 2.5 * by_question.std()


def test_quarters_are_averaged_with_equal_weight():
    """The headline is the mean over quarters, so a quarter with more questions must not count more."""
    q1 = (np.repeat(np.arange(10), 2), np.ones(20))
    q2 = (np.repeat(np.arange(50), 20), np.zeros(1000))
    draws = S.call_bootstrap([q1, q2], n_boot=50)
    assert np.allclose(draws, 0.5)


def _write(dirpath, year, model, hits, calls):
    ranks = np.where(np.asarray(hits) == 1, 1, 2).astype(np.int32)
    np.savez_compressed(dirpath / f"{S.shard_of(year)}_{model}_s0.npz",
                        keydevid=np.asarray(calls), dense__within=ranks)


def test_a_missing_replicate_is_reported_missing_not_filled_with_seed_zero(tmp_path, monkeypatch):
    """Filling a missing replicate with seed 0 would make retraining look perfectly stable."""
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    calls = np.repeat(np.arange(10), 3)
    for y in S.TRAIN_YEARS:
        _write(tmp_path, y, S.FAMILIES["bge"], np.zeros(30), calls)
        for t in (0, 1):
            _write(tmp_path, y, S.TRAINED["ours"][t].format(y=y), np.ones(30), calls)
        _write(tmp_path, y, S.TRAINED["abl"][0].format(y=y), np.ones(30), calls)
    r = S.training_seeds(n_boot=20)
    assert r["per_training_seed"]["1"]["abl"] == "missing"
    assert r["per_training_seed"]["2"]["ours"] == "missing"
    assert "ours_minus_abl" not in r["per_training_seed"]["1"]
    assert r["per_training_seed"]["1"]["ours_minus_bge"]["mean"] == 1.0
    assert r["per_training_seed"]["0"]["ours_minus_abl"]["mean"] == 0.0


def test_later_vintage_names_the_2024_and_same_year_models():
    assert S.later_vintage("ecalls-ret", 2015, "leaky") == "ecalls-ret-20241231"
    assert S.later_vintage("ecalls-ret", 2015, "adjacent") == "ecalls-ret-20161231"
    with pytest.raises(ValueError):
        S.later_vintage("ecalls-ret", 2015, "both")


def test_lookahead_pairs_each_quarter_with_its_point_in_time_vintage(tmp_path, monkeypatch):
    """A later vintage that answers every question and a point-in-time one that answers half gives
    +0.5 in every quarter, and a quarter whose later file is absent is listed as missing."""
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    calls = np.repeat(np.arange(10), 2)
    half = np.tile([1, 0], 10)
    for fam in S.LOOKAHEAD_FAMILIES:
        for y in S.LOOKAHEAD_YEARS:
            _write(tmp_path, y, f"{fam}-{y}1231", half, calls)
            _write(tmp_path, y, f"{fam}-20241231", np.ones(20), calls)
            if y not in (2012, 2023):     # 2023 + 1 is the 2024 file written above
                _write(tmp_path, y, f"{fam}-{y + 1}1231", half, calls)
    r = S.lookahead_summary(n_boot=20)
    leaky = r["families"]["ecalls-ret"]["leaky"]
    assert leaky["mean"] == 0.5 and leaky["positive"] == leaky["n_quarters"] == len(S.LOOKAHEAD_YEARS)
    assert leaky["ci95"] == (0.5, 0.5)
    # The same-year vintage of the last quarter is the 2024 one, so only that quarter moves.
    adj = r["families"]["ecalls-ret"]["adjacent"]
    assert adj["missing"] == [S.shard_of(2012)] and adj["n_quarters"] == len(S.LOOKAHEAD_YEARS) - 1
    assert adj["positive"] == 1 and adj["per_quarter"][S.shard_of(2023)]["diff"] == 0.5
    assert np.isclose(adj["mean"], 0.5 / (len(S.LOOKAHEAD_YEARS) - 1))


def test_lookahead_refuses_vintages_scored_on_different_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    for y in S.LOOKAHEAD_YEARS:
        _write(tmp_path, y, f"ecalls-ret-{y}1231", np.ones(4), np.array([1, 1, 2, 2]))
        _write(tmp_path, y, "ecalls-ret-20241231", np.ones(4), np.array([1, 1, 3, 3]))
    with pytest.raises(ValueError, match="different questions"):
        S.lookahead_summary(n_boot=5)


def test_the_chain_starts_from_the_from_scratch_2010_retriever():
    assert S.pit_vintage(S.CHAIN, 2010) == "ecalls-ret-20101231"
    assert S.pit_vintage(S.CHAIN, 2011) == "ecalls-chain-ret-20111231"
    assert S.pit_vintage("ecalls-abl", 2010) == "ecalls-abl-20101231"
    assert S.later_vintage(S.CHAIN, 2010, "adjacent") == "ecalls-chain-ret-20111231"


def test_chain_summary_pairs_quarters_and_skips_the_shared_2010_model(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    calls = np.repeat(np.arange(5), 2)
    for y in range(2010, 2025):
        _write(tmp_path, y, f"ecalls-ret-{y}1231", np.tile([1, 0], 5), calls)
        _write(tmp_path, y, f"ecalls-abl-{y}1231", np.ones(10), calls)
        if y > 2010 and y != 2020:
            _write(tmp_path, y, f"ecalls-chain-ret-{y}1231", np.ones(10), calls)
    r = S.chain_summary(n_boot=10)
    v = r["chain_minus_ecalls-ret"]
    assert v["n_quarters"] == 13 and v["mean"] == 0.5 and v["wins"] == 13
    assert r["chain_minus_ecalls-abl"]["mean"] == 0.0
    assert f"{S.shard_of(2020)} ecalls-ret" in r["missing"]


def test_references_summary_pairs_ours_with_each_general_retriever(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    calls = np.repeat(np.arange(4), 2)
    for y in S.YEARS:
        for ref in S.REFERENCES:
            _write(tmp_path, y, ref, np.zeros(8), calls)
        _write(tmp_path, y, f"ecalls-ret-{y}1231", np.ones(8), calls)
        _write(tmp_path, y, f"ecalls-abl-{y}1231", np.tile([1, 0], 4), calls)
        if y > 2010:
            _write(tmp_path, y, f"ecalls-chain-ret-{y}1231", np.ones(8), calls)
    r = S.references_summary(n_boot=10)
    assert r["differences"]["ours_minus_e5-base-v2"]["mean"] == 1.0
    assert r["differences"]["abl_minus_bge-large-en-v1.5"]["mean"] == 0.5
    assert r["differences"]["chain_minus_bge-base-en-v1.5"]["wins"] == len(S.YEARS)


def test_budget_summary_pairs_the_twice_budget_vintage_with_both_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    calls = np.repeat(np.arange(4), 2)
    for y in S.YEARS:
        # Point in time first: for y = 2024 it is the same file as the 2024 vintage, which wins.
        _write(tmp_path, y, f"ecalls-ret-{y}1231", np.zeros(8), calls)
        _write(tmp_path, y, "ecalls-ret-20241231", np.tile([1, 0], 4), calls)
        if y != 2015:
            _write(tmp_path, y, S.TWICE, np.ones(8), calls)
    r = S.budget_summary(n_boot=10)
    a, b = r["twice_minus_ours2024"], r["twice_minus_pit"]
    assert a["mean"] == 0.5 and a["n_quarters"] == len(S.YEARS) - 1 and a["missing"] == [S.shard_of(2015)]
    assert b["mean"] == 1.0 and b["n_quarters"] == len(S.LOOKAHEAD_YEARS) - 1


def test_budget_summary_refuses_models_scored_on_different_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    _write(tmp_path, 2010, "ecalls-ret-20241231", np.ones(4), np.array([1, 1, 2, 2]))
    _write(tmp_path, 2010, S.TWICE, np.ones(4), np.array([1, 1, 3, 3]))
    with pytest.raises(ValueError, match="different questions"):
        S.budget_summary(n_boot=5)


def test_compute_prediction_reads_one_doubling_as_the_rate():
    tokens = {y: 1000 * 2 ** (y - 2010) for y in range(2010, 2025)}
    c = S.compute_prediction(0.02, tokens)
    assert np.isclose(c["adjacent_mean"], 0.02)
    assert np.isclose(c["leaky"][S.shard_of(2023)], 0.02) and np.isclose(c["leaky"][S.shard_of(2010)], 0.28)
    assert c["adjacent_token_ratio"] == [2.0, 2.0]


def test_chain_tokens_accumulate_from_the_2010_base(tmp_path):
    import json
    for name, t in [("ecalls-mlm-20101231", 100)] + [(f"ecalls-chain-mlm-{y}1231", 10) for y in range(2011, 2025)]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "_run.json").write_text(json.dumps({"tokens_seen": t}))
    t = S.chain_tokens(tmp_path)
    assert t[2010] == 100 and t[2011] == 110 and t[2024] == 240


def test_replay_summary_splits_the_chains_premium_into_compute_and_later_text(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    calls = np.repeat(np.arange(4), 2)
    for y in S.LOOKAHEAD_YEARS:
        _write(tmp_path, y, S.pit_vintage(S.CHAIN, y), np.zeros(8), calls)
        _write(tmp_path, y, f"{S.REPLAY}-{y}1231", np.tile([1, 0], 4), calls)
        _write(tmp_path, y, S.later_vintage(S.CHAIN, y, "adjacent"), np.ones(8), calls)
    r = S.replay_summary(n_boot=10)
    assert r["compute_rpl_minus_chain"]["mean"] == 0.5
    assert r["later_text_chain_next_minus_rpl"]["mean"] == 0.5
    assert r["chain_adjacent"]["mean"] == 1.0
    assert r["chain_adjacent"]["n_quarters"] == len(S.LOOKAHEAD_YEARS)


def test_replay_summary_pairs_the_control_with_vintage_y_on_quarter_y_plus_one(tmp_path, monkeypatch):
    """The control of 2015 is scored on 2016Q2, beside chained vintages 2015 and 2016."""
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    calls = np.repeat(np.arange(4), 2)
    y = 2015
    _write(tmp_path, y, f"ecalls-chain-ret-{y}1231", np.zeros(8), calls)
    _write(tmp_path, y, f"ecalls-rpl-ret-{y}1231", np.ones(8), calls)
    _write(tmp_path, y, f"ecalls-chain-ret-{y + 1}1231", np.ones(8), calls)
    r = S.replay_summary(n_boot=10)
    assert list(r["compute_rpl_minus_chain"]["per_quarter"]) == ["2016Q2"]
    assert r["compute_rpl_minus_chain"]["mean"] == 1.0 and r["later_text_chain_next_minus_rpl"]["mean"] == 0.0


def test_staleness_pairs_each_vintage_with_its_familys_2024_vintage_on_2025q2(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    calls = np.repeat(np.arange(4), 2)
    y0 = S.STALENESS_QUARTER_YEAR
    for fam in ("ecalls-ret", S.CHAIN):
        _write(tmp_path, y0, f"{fam}-20241231", np.ones(8), calls)
    for y in S.LOOKAHEAD_YEARS:
        # older vintages score worse in the from-scratch family only
        hits = np.ones(8) if y >= 2017 else np.tile([1, 0], 4)
        _write(tmp_path, y0, S.pit_vintage("ecalls-ret", y), hits, calls)
        if y > 2010:
            _write(tmp_path, y0, S.pit_vintage(S.CHAIN, y), np.ones(8), calls)
    r = S.staleness_summary(n_boot=10)
    ours, chain = r["families"]["ecalls-ret"], r["families"][S.CHAIN]
    assert r["quarter"] == "2025Q2"
    assert ours["per_vintage"]["2010"]["minus_2024"] == -0.5 and ours["per_vintage"]["2020"]["minus_2024"] == 0.0
    assert ours["slope_per_year"] < 0
    # the chain's 2010 member is the from-scratch 2010 retriever, so it inherits its score
    assert chain["per_vintage"]["2010"]["minus_2024"] == -0.5
    assert chain["per_vintage"]["2015"]["minus_2024"] == 0.0
    # Fig. 3b plots levels: each interval brackets its level, and a constant level has zero width
    lo, hi = ours["per_vintage"]["2010"]["level_ci95"]
    assert lo <= ours["per_vintage"]["2010"]["recall_at_1"] == 0.5 <= hi
    assert ours["latest_level_ci95"] == [1.0, 1.0] or tuple(ours["latest_level_ci95"]) == (1.0, 1.0)


def _write_both(dirpath, year, model, within, cross, calls):
    w = np.where(np.asarray(within) == 1, 1, 2).astype(np.int32)
    c = np.where(np.asarray(cross) == 1, 1, 2).astype(np.int32)
    np.savez_compressed(dirpath / f"{S.shard_of(year)}_{model}_s0.npz",
                        keydevid=np.asarray(calls), dense__within=w, dense__cross=c)


def test_references_summary_reads_cross_call_ranks_and_finberts_clean_quarters(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    calls = np.repeat(np.arange(4), 2)
    for y in S.YEARS:
        for ref in S.REFERENCES:
            # FinBERT does well on the quarters it may have read and poorly after them
            good = ref == "finbert-pretrain" and y < S.CLEAN_FROM[ref]
            _write_both(tmp_path, y, ref, np.ones(8) if good else np.zeros(8), np.zeros(8), calls)
        for fam in ("ecalls-ret", "ecalls-abl"):
            _write_both(tmp_path, y, f"{fam}-{y}1231", np.ones(8), np.tile([1, 0], 4), calls)
        if y > 2010:
            _write_both(tmp_path, y, f"ecalls-chain-ret-{y}1231", np.ones(8), np.tile([1, 0], 4), calls)
    within = S.references_summary(n_boot=10)
    cross = S.references_summary(n_boot=10, key=S.CROSS)
    assert within["differences"]["ours_minus_e5-base-v2"]["mean"] == 1.0
    assert cross["differences"]["ours_minus_e5-base-v2"]["mean"] == 0.5
    assert within["differences"]["ours_minus_finbert-pretrain_clean"]["mean"] == 1.0
    assert within["differences"]["ours_minus_finbert-pretrain_clean"]["n_quarters"] == 2024 - 2019 + 1
    assert within["differences"]["ours_minus_finbert-pretrain"]["mean"] < 1.0
    assert cross["mean_recall_at_1"]["ours"] == 0.5


def test_decomposition_rows_pair_the_right_models_in_both_scopes(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    calls = np.repeat(np.arange(4), 2)
    half = np.tile([1, 0], 4)
    for y in S.YEARS:
        # write the per-vintage files first: for y = 2024 they are the files the 2024 rows read
        _write_both(tmp_path, y, f"ecalls-abl-{y}1231", np.ones(8), np.ones(8), calls)
        _write_both(tmp_path, y, f"chrono-bert-v1-{y}1231", np.zeros(8), np.zeros(8), calls)
        _write_both(tmp_path, y, f"ecalls-ret-{y}1231", half, np.zeros(8), calls)
        _write_both(tmp_path, y, "e5-base-v2", np.zeros(8), np.zeros(8), calls)
        _write_both(tmp_path, y, "finbert-pretrain", np.zeros(8), np.zeros(8), calls)
        _write_both(tmp_path, y, S.TWICE, np.ones(8), np.ones(8), calls)
        if y > 2010:
            _write_both(tmp_path, y, f"ecalls-chain-ret-{y}1231", np.ones(8), np.zeros(8), calls)
        if y in S.DAPT_YEARS:
            _write_both(tmp_path, y, f"ecalls-dapt-ret-{y}1231", np.ones(8), np.ones(8), calls)
    r = S.decomposition_summary(n_boot=10)
    assert r["task_stage_on_chronobert"]["within"]["mean"] == 1.0
    assert r["ours_minus_e5"]["within"]["mean"] == 0.5 and r["ours_minus_e5"]["cross"]["mean"] == 0.0
    assert r["ours_minus_finbert_clean"]["within"]["n_quarters"] == len(S.CLEAN_YEARS)
    assert r["dapt_minus_abl"]["within"]["n_quarters"] == 3 and r["dapt_minus_abl"]["within"]["mean"] == 0.0
    assert r["chain_minus_ours"]["within"]["n_quarters"] == 14 and r["chain_minus_ours"]["within"]["mean"] == 0.5
    assert r["twice_minus_ours2024"]["cross"]["mean"] == 1.0
    assert r["ours_minus_abl"]["within"]["mean"] == -0.5


def test_seed_lookahead_pairs_each_replicate_with_the_seed_zero_point_in_time_vintages(tmp_path, monkeypatch):
    """Seed 0 is the canonical 2024 vintage, seeds 1 and 2 its replicates. Each is set against the same
    point-in-time vintages, so a replicate that answers fewer questions gives a smaller premium, and a
    replicate with no files is reported missing rather than replaced by seed 0."""
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    calls = np.repeat(np.arange(10), 2)
    half = np.tile([1, 0], 10)
    three_quarters = np.tile([1, 1, 1, 0], 5)
    for y in S.LOOKAHEAD_YEARS:
        _write(tmp_path, y, f"ecalls-ret-{y}1231", half, calls)
        _write(tmp_path, y, "ecalls-ret-20241231", np.ones(20), calls)
        _write(tmp_path, y, "ecalls-seed1-ret-20241231", three_quarters, calls)
        _write(tmp_path, y, "ecalls-seed2-ret-20241231", half, calls)
    r = S.seed_lookahead_summary(n_boot=20)["families"]
    assert r["ecalls-ret"]["0"]["mean"] == 0.5
    assert r["ecalls-ret"]["1"]["mean"] == 0.25
    assert r["ecalls-ret"]["2"]["mean"] == 0.0 and r["ecalls-ret"]["2"]["positive"] == 0
    assert r["ecalls-ret"]["1"]["model"] == "ecalls-seed1-ret-20241231"
    assert "mean" not in r["ecalls-abl"]["1"]
    assert r["ecalls-abl"]["1"]["missing"] == [S.shard_of(y) for y in S.LOOKAHEAD_YEARS]


# --- the pretraining-seed replicates of the chain (D34) -------------------------------------------

@pytest.fixture
def no_registry_guard(monkeypatch):
    """The replicates are registered only on the cluster. Record the guard's calls instead."""
    from ecalls.chunk import registry as R
    calls = []
    monkeypatch.setattr(R, "require_point_in_time", lambda model, date: calls.append((model, date)))
    return calls


def test_a_replicate_names_every_role_with_its_own_prefix(no_registry_guard):
    n1, n2 = S.pretrain_seed_names(1), S.pretrain_seed_names(2)
    assert n1["pit"](2010) == "ecalls-pseed1-ret-20101231"          # the reseeded base, not ecalls-ret
    assert n1["pit"](2015) == "ecalls-pseed1-chain-ret-20151231"
    assert n1["chain"](2024) == "ecalls-pseed1-chain-ret-20241231"
    assert n1["rpl"](2015) == "ecalls-pseed1-rpl-ret-20151231"
    assert all("pseed1" not in f(y) for f in n2.values() for y in (2010, 2015, 2023))


def test_seed_zero_keeps_the_reported_chain(no_registry_guard):
    n0 = S.pretrain_seed_names(0)
    assert n0["chain"](2015) == "ecalls-chain-ret-20151231" and n0["rpl"](2015) == "ecalls-rpl-ret-20151231"


def test_a_replicates_point_in_time_vintage_passes_the_registry_guard_at_its_quarter(no_registry_guard):
    S.pretrain_seed_names(2)["pit"](2013)
    import datetime as dt
    assert no_registry_guard == [("ecalls-pseed2-chain-ret-20131231", dt.date(2014, 4, 1))]


def test_pretrain_seed_summary_pairs_only_models_of_one_seed(tmp_path, monkeypatch, no_registry_guard):
    """Seed 1's premium must come from seed 1's models. Pairing a replicate with the seed-0 chain would
    report seed 0's premium twice and make retraining look perfectly stable."""
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    calls = np.repeat(np.arange(4), 2)
    for s, later in ((0, np.ones(8)), (1, np.tile([1, 0], 4))):
        nm = S.pretrain_seed_names(s)
        for y in S.LOOKAHEAD_YEARS:
            _write(tmp_path, y, nm["pit"](y), np.zeros(8), calls)
            _write(tmp_path, y, nm["rpl"](y), np.zeros(8), calls)
            _write(tmp_path, y, nm["chain"](y + 1), later, calls)
            if y + 1 != 2024:
                _write(tmp_path, y, nm["chain"](2024), later, calls)
    r = S.pretrain_seed_summary(n_boot=10)
    assert r["seeds"]["0"]["adjacent"]["mean"] == 1.0
    assert r["seeds"]["1"]["adjacent"]["mean"] == 0.5
    assert r["seeds"]["1"]["leaky"]["mean"] == 0.5
    assert r["seeds"]["1"]["compute_rpl_minus_chain"]["mean"] == 0.0
    assert r["seeds"]["1"]["later_text_chain_next_minus_rpl"]["mean"] == 0.5
    assert "mean" not in r["seeds"]["2"]["adjacent"]                  # missing, never filled
    assert len(r["seeds"]["2"]["adjacent"]["missing"]) == len(S.LOOKAHEAD_YEARS)


def test_pretrain_seed_summary_refuses_models_scored_on_different_questions(tmp_path, monkeypatch,
                                                                            no_registry_guard):
    monkeypatch.setattr(S, "RANKS_DIR", tmp_path)
    nm = S.pretrain_seed_names(1)
    _write(tmp_path, 2015, nm["pit"](2015), np.zeros(4), [1, 1, 2, 2])
    _write(tmp_path, 2015, nm["chain"](2016), np.ones(4), [1, 1, 3, 3])
    with pytest.raises(ValueError, match="different questions"):
        S.pretrain_seed_summary(n_boot=10, seeds=(1,))
