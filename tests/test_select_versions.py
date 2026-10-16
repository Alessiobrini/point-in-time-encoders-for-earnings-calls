"""Edge cases for the version rule (D4). Pure, no database."""
import datetime as dt
import sys

import pandas as pd
import pytest

from ecalls.config import constants as C
from ecalls.ingest.select_versions import choose_version

D = dt.date


def cand(keydevid, transcriptid, chars, ncomp=30, pres="Final", coll="Edited Copy",
         created=D(2024, 1, 2), companyid=7, call_date=D(2024, 1, 1)):
    return {"keydevid": keydevid, "transcriptid": transcriptid, "companyid": companyid,
            "call_date": call_date, "total_chars": chars, "n_components": ncomp,
            "presentation_type": pres, "collection_type": coll, "created_date": created}


def frame(rows):
    return pd.DataFrame(rows)


def test_longest_wins_regardless_of_collection_type():
    """The measured case. A Spellchecked/Preliminary copy that is longer must beat an
    Edited/Final one, which any fixed type hierarchy would get backwards."""
    out = choose_version(frame([
        cand(1, 10, 30_000, pres="Final", coll="Edited Copy"),
        cand(1, 11, 45_000, pres="Preliminary", coll="Spellchecked Copy"),
    ]))
    assert len(out) == 1
    r = out.iloc[0]
    assert r["chosen_transcriptid"] == 11
    assert r["chosen_collection_type"] == "Spellchecked Copy"
    assert r["tiebreak_used"] == "chars"
    assert r["n_versions"] == 2
    assert r["runner_up_chars"] == 30_000
    assert abs(r["margin_ratio"] - 1.5) < 1e-5


def test_single_version_passthrough():
    out = choose_version(frame([cand(1, 10, 30_000)])).iloc[0]
    assert out["chosen_transcriptid"] == 10
    assert out["n_versions"] == 1
    assert pd.isna(out["runner_up_chars"])
    assert pd.isna(out["margin_ratio"])
    assert out["tiebreak_used"] == "chars"


def test_component_count_breaks_a_character_tie():
    out = choose_version(frame([
        cand(1, 10, 30_000, ncomp=20), cand(1, 11, 30_000, ncomp=40),
    ])).iloc[0]
    assert out["chosen_transcriptid"] == 11
    assert out["tiebreak_used"] == "n_components"


def test_final_beats_preliminary_when_chars_and_components_tie():
    out = choose_version(frame([
        cand(1, 10, 30_000, ncomp=30, pres="Preliminary"),
        cand(1, 11, 30_000, ncomp=30, pres="Final"),
    ])).iloc[0]
    assert out["chosen_transcriptid"] == 11
    assert out["tiebreak_used"] == "presentation"


def test_earlier_creation_breaks_the_next_tie():
    out = choose_version(frame([
        cand(1, 10, 30_000, created=D(2024, 3, 1)),
        cand(1, 11, 30_000, created=D(2024, 1, 5)),
    ])).iloc[0]
    assert out["chosen_transcriptid"] == 11
    assert out["tiebreak_used"] == "created"


def test_transcriptid_is_the_final_tiebreak():
    out = choose_version(frame([cand(1, 77, 30_000), cand(1, 12, 30_000)])).iloc[0]
    assert out["chosen_transcriptid"] == 12
    assert out["tiebreak_used"] == "transcriptid"


def _tied_rows():
    """Candidates that force every tiebreak key to be reached.

    The earlier fixture used `chars = 30_000 + (j*7 % 3)*1000`, which gave every call a unique
    maximum, so keys 2 to 5 were never exercised. An audit mutation-tested it: reversing the
    transcriptid sort direction left that test passing. Here each call ties at a different depth.
    """
    rows = []
    for k in range(1, 26):
        depth = k % 4
        if depth == 0:        # decided on n_components
            rows += [cand(k, 10 * k + 1, 30_000, ncomp=20),
                     cand(k, 10 * k + 2, 30_000, ncomp=40)]
        elif depth == 1:      # decided on presentation type
            rows += [cand(k, 10 * k + 1, 30_000, ncomp=30, pres="Preliminary"),
                     cand(k, 10 * k + 2, 30_000, ncomp=30, pres="Final")]
        elif depth == 2:      # decided on creation date
            rows += [cand(k, 10 * k + 1, 30_000, ncomp=30, created=D(2024, 3, 1)),
                     cand(k, 10 * k + 2, 30_000, ncomp=30, created=D(2024, 1, 5))]
        else:                 # fully tied, decided on transcriptid alone
            rows += [cand(k, 10 * k + 7, 30_000, ncomp=30),
                     cand(k, 10 * k + 2, 30_000, ncomp=30)]
    return rows


def test_determinism_under_permutation():
    """Non-determinism here would silently change the corpus between runs, and no downstream check
    would catch it. The fixture ties at every depth so all five sort keys are actually reached."""
    rows = _tied_rows()
    base = choose_version(frame(rows))
    used = set(base["tiebreak_used"])
    assert used == {"n_components", "presentation", "created", "transcriptid"}, (
        f"fixture does not exercise every tiebreak, only reached {sorted(used)}")
    for seed in range(10):
        shuffled = frame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        pd.testing.assert_frame_equal(base, choose_version(shuffled))


def test_the_final_tiebreak_direction_is_load_bearing():
    """A mutation guard. Reversing the transcriptid sort must change the answer on a fully tied call,
    which is what proves the key is doing work rather than being decoration."""
    import ecalls.ingest.select_versions as sv
    rows = [cand(1, 77, 30_000, ncomp=30), cand(1, 12, 30_000, ncomp=30)]
    assert choose_version(frame(rows)).iloc[0]["chosen_transcriptid"] == 12
    orig = list(sv.ASCENDING)
    try:
        sv.ASCENDING[4] = False
        assert choose_version(frame(rows)).iloc[0]["chosen_transcriptid"] == 77
    finally:
        sv.ASCENDING[:] = orig
    assert choose_version(frame(rows)).iloc[0]["chosen_transcriptid"] == 12


def test_no_call_is_ever_dropped():
    rows = [cand(k, 10 * k, 30_000) for k in range(1, 26)]
    assert len(choose_version(frame(rows))) == 25


def test_suspect_flags_are_set_but_nothing_is_filtered():
    out = choose_version(frame([
        cand(1, 10, C.SUSPECT_SHORT_CHARS - 1),
        cand(2, 20, C.SUSPECT_LONG_CHARS + 1),
        cand(3, 30, 50_000),
    ])).set_index("keydevid")
    assert bool(out.loc[1, "suspect_short"]) and not bool(out.loc[1, "suspect_long"])
    assert bool(out.loc[2, "suspect_long"]) and not bool(out.loc[2, "suspect_short"])
    assert not bool(out.loc[3, "suspect_short"]) and not bool(out.loc[3, "suspect_long"])
    assert len(out) == 3


def test_a_zero_length_transcript_ranks_last_but_is_kept():
    """A transcript with no component rows is real. It must lose, not disappear."""
    out = choose_version(frame([cand(1, 10, 0, ncomp=0), cand(1, 11, 25_000)])).iloc[0]
    assert out["chosen_transcriptid"] == 11
    assert out["n_versions"] == 2


def test_all_candidates_are_recorded_so_the_rule_is_reversible():
    out = choose_version(frame([
        cand(1, 10, 30_000, coll="Edited Copy"), cand(1, 11, 45_000, coll="Audited Copy"),
    ])).iloc[0]
    assert sorted(out["candidate_transcriptids"]) == [10, 11]
    assert sorted(out["candidate_chars"]) == [30_000, 45_000]
    assert sorted(out["candidate_collection_types"]) == ["Audited Copy", "Edited Copy"]
    assert out["rule_version"] == C.VERSION_RULE


if __name__ == "__main__":
    print("run with pytest")
    sys.exit(0)


def test_load_candidates_refuses_to_silently_drop_transcripts_without_lengths(monkeypatch, tmp_path):
    """Regression, and worse than it looks. The inner join on transcriptid is where a gap in the
    length shards disappears: the calls never reach version_choice.parquet, pull_components never
    pulls them, and completeness reports UNEXPLAINED = 0 because it compares against
    version_choice itself. An earlier note here was deleted by accident when the parent/subsidiary
    dedup was added, leaving no signal at all."""
    import pandas as pd
    import ecalls.ingest.select_versions as sv

    idx = pd.DataFrame({"keydevid": [1, 2], "transcriptid": [10, 20], "companyid": [7, 7],
                        "call_date": [D(2024, 1, 1)] * 2, "collection_type": ["Edited Copy"] * 2,
                        "presentation_type": ["Final"] * 2, "created_date": [D(2024, 1, 2)] * 2})
    lens = pd.DataFrame({"transcriptid": [10], "n_components": [30], "total_chars": [30_000]})
    shard = tmp_path / "lengths_2024.parquet"
    lens.to_parquet(shard, index=False)

    monkeypatch.setattr(sv.pd, "read_parquet",
                        lambda pth, **kw: idx if "transcript_index" in str(pth) else lens)
    monkeypatch.setattr(sv.P, "LENGTHS_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="have no length row"):
        sv.load_candidates()
    out = sv.load_candidates(allow_missing=True)      # explicit opt-in still works
    assert len(out) == 1 and out.iloc[0]["transcriptid"] == 10
