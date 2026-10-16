"""Edge cases for the chunker. Pure except for the reconstruction tests, which read the corpus."""
import sys

import pytest

from ecalls.chunk.chunker import (
    MAX_CHARS, TARGET_CHARS, build_units, chunk_call, chunk_id, pack_unit, split_long_text,
)

V = "test-v1"


def comp(cid, order, section, type_id, text, speaker="Jane Doe", role="Executives", kd=111):
    return {"keydevid": kd, "component_id": cid, "component_order": order, "section": section,
            "component_type_id": type_id, "text": text, "speaker_name": speaker,
            "speaker_type": role, "text_sha1": f"h{cid}"}


# --- split_long_text ---------------------------------------------------------------------------

def test_short_text_is_one_piece():
    assert split_long_text("hello") == [(0, 5)]


def test_pieces_are_contiguous_and_cover_the_text_exactly():
    t = "Sentence one. Sentence two. " * 400
    sp = split_long_text(t)
    assert sp[0][0] == 0 and sp[-1][1] == len(t)
    assert all(sp[i][1] == sp[i + 1][0] for i in range(len(sp) - 1))
    assert "".join(t[a:b] for a, b in sp) == t


def test_no_piece_exceeds_the_ceiling_when_breaks_exist():
    t = "Sentence one. Sentence two. " * 400
    assert max(b - a for a, b in split_long_text(t)) <= MAX_CHARS + 1


def test_a_single_unbreakable_sentence_is_emitted_whole_rather_than_cut_mid_word():
    """A truncated word is worse for retrieval than an over-long chunk, so this is on purpose."""
    t = "word " * 2000            # 10,000 chars, no sentence terminator anywhere
    sp = split_long_text(t)
    assert "".join(t[a:b] for a, b in sp) == t
    assert len(sp) == 1 and sp[0] == (0, len(t))


def test_paragraph_boundaries_are_preferred_over_sentence_boundaries():
    para = "A. B. C. " * 100
    t = para + "\n\n" + para
    sp = split_long_text(t, target=len(para), max_=len(para) + 10)
    assert any(t[a:b].startswith("A.") for a, b in sp)


# --- build_units -------------------------------------------------------------------------------

def test_a_question_opens_a_unit_and_the_answers_join_it():
    cs = [comp(1, 1, "qa", 3, "Q1?"), comp(2, 2, "qa", 4, "A1."), comp(3, 3, "qa", 4, "A1 more."),
          comp(4, 4, "qa", 3, "Q2?"), comp(5, 5, "qa", 4, "A2.")]
    units = build_units(cs)
    assert [[c["component_id"] for c in u] for u in units] == [[1, 2, 3], [4, 5]]


def test_consecutive_presentation_components_form_one_unit():
    cs = [comp(1, 1, "presentation", 2, "a"), comp(2, 2, "presentation", 2, "b")]
    assert len(build_units(cs)) == 1


def test_operator_turns_are_dropped_by_default_and_split_the_units_around_them():
    cs = [comp(1, 1, "presentation", 2, "a"), comp(2, 2, "operator", 1, "next question"),
          comp(3, 3, "qa", 3, "Q?")]
    units = build_units(cs)
    assert [[c["component_id"] for c in u] for u in units] == [[1], [3]]


def test_operator_turns_are_kept_when_asked():
    cs = [comp(1, 1, "presentation", 2, "a"), comp(2, 2, "operator", 1, "next")]
    units = build_units(cs, drop_operator=False)
    assert [[c["component_id"] for c in u] for u in units] == [[1], [2]]


def test_no_non_operator_component_is_lost():
    cs = [comp(i, i, "qa", 3 if i % 2 else 4, f"t{i}") for i in range(1, 21)]
    kept = {c["component_id"] for u in build_units(cs) for c in u}
    assert kept == {c["component_id"] for c in cs}


# --- pack_unit ---------------------------------------------------------------------------------

def test_small_components_are_grouped_into_one_chunk():
    unit = [comp(1, 1, "qa", 3, "Q?"), comp(2, 2, "qa", 4, "A.")]
    out = pack_unit(unit)
    assert len(out) == 1
    assert out[0]["component_ids"] == [1, 2]
    assert out[0]["text"] == "Q?\nA."


def test_an_over_long_component_is_split_and_keeps_its_own_offsets():
    long = "Sentence. " * 600                      # 6,000 chars
    out = pack_unit([comp(9, 1, "presentation", 2, long)])
    assert len(out) > 1
    assert all(c["component_ids"] == [9] for c in out)
    assert "".join(c["text"] for c in out) == long
    assert out[0]["char_start"] == 0 and out[-1]["char_end"] == len(long)


def test_the_slice_invariant_holds_for_every_chunk_a_unit_produces():
    """text must equal the components joined by a newline, sliced by char_start:char_end. This is
    what makes a chunk reconstructible from the corpus rather than a copy that can drift."""
    unit = [comp(1, 1, "qa", 3, "Q? " * 400), comp(2, 2, "qa", 4, "A. " * 400)]
    for ch in pack_unit(unit):
        joined = "\n".join(c["text"] for c in unit if c["component_id"] in ch["component_ids"])
        assert ch["text"] == joined[ch["char_start"]:ch["char_end"]]


# --- chunk_id and chunk_call -------------------------------------------------------------------

def test_chunk_id_is_deterministic_and_version_sensitive():
    a = chunk_id(111, 9, 0, "v1")
    assert a == chunk_id(111, 9, 0, "v1")
    assert a != chunk_id(111, 9, 0, "v2")
    assert a != chunk_id(111, 9, 1, "v1")
    assert a != chunk_id(112, 9, 0, "v1")
    assert len(a) == 40


def test_chunk_ids_are_unique_within_a_call():
    cs = [comp(i, i, "qa", 3 if i % 2 else 4, "Sentence. " * 300) for i in range(1, 13)]
    ids = [c["chunk_id"] for c in chunk_call(cs, V)]
    assert len(ids) == len(set(ids))


def test_speaker_is_named_when_single_and_mixed_when_not():
    same = chunk_call([comp(1, 1, "qa", 3, "Q?", speaker="A", role="Analysts")], V)[0]
    assert same["speaker_name"] == "A" and same["speaker_type"] == "Analysts"
    mixed = chunk_call([comp(1, 1, "qa", 3, "Q?", speaker="A", role="Analysts"),
                        comp(2, 2, "qa", 4, "R.", speaker="B", role="Executives")], V)[0]
    assert mixed["speaker_name"] is None and mixed["speaker_type"] == "Mixed"


def test_empty_call_gives_no_chunks():
    assert chunk_call([], V) == []


def test_a_call_of_only_operator_turns_gives_no_chunks_by_default():
    assert chunk_call([comp(1, 1, "operator", 1, "welcome")], V) == []


# --- against the real corpus -------------------------------------------------------------------

def test_reconstruction_invariant_on_real_calls(components_path):
    """Every chunk of real calls must slice exactly out of its own components, and no component the
    chunker is supposed to keep may be lost.

    The call set is FIXED, not sampled. An earlier version used `using sample 8 rows`, which made the
    test flaky: it failed on a draw containing a long operator component and passed on the retry. A
    test whose result depends on the draw hides exactly the bug it should catch, so the ids below are
    pinned and include two calls with long operator components (2544783 is the 105,420-character
    WebSideStory turn) plus a spread of ordinary ones.
    """
    import duckdb
    from ecalls.chunk.chunker import OPERATOR_KEEP_CHARS
    con = duckdb.connect(); con.execute("set memory_limit='4GB'")
    pinned = [2544783, 3430579, 6170256, 4861740]
    ordinary = [r[0] for r in con.execute(
        f"select keydevid from read_parquet('{components_path}') "
        f"where keydevid not in ({','.join(map(str, pinned))}) "
        f"group by keydevid order by keydevid limit 8").fetchall()]
    kds = pinned + ordinary
    assert len(kds) == 12
    for kd in kds:
        rows = con.execute(
            f"select keydevid, component_id, component_order, section, component_type_id, text, "
            f"speaker_name, speaker_type, text_sha1 from read_parquet('{components_path}') "
            f"where keydevid = {kd} order by component_order").df().to_dict("records")
        by_id = {int(r["component_id"]): r["text"] for r in rows}
        chunks = chunk_call(rows, V)
        assert chunks, f"call {kd} produced no chunks"
        for ch in chunks:
            joined = "\n".join(by_id[i] for i in ch["component_ids"])
            assert ch["text"] == joined[ch["char_start"]:ch["char_end"]], f"call {kd} chunk drifted"
            assert ch["n_chars"] == len(ch["text"])
        covered = {i for ch in chunks for i in ch["component_ids"]}
        # The rule is length-based, not section-based: short operator turns are dropped as
        # boilerplate and long ones are kept because they are misclassified content.
        expected = {int(r["component_id"]) for r in rows
                    if r["section"] != "operator" or len(r["text"]) >= OPERATOR_KEEP_CHARS}
        assert covered == expected, (
            f"call {kd}: lost {len(expected - covered)}, gained {len(covered - expected)}")


if __name__ == "__main__":
    print("run with pytest")
    sys.exit(0)


def test_a_long_operator_component_is_kept_because_it_is_content_not_boilerplate():
    """Regression. Dropping by section alone left 57 calls with zero chunks, including one PepsiCo
    call, because Capital IQ sometimes attributes a whole transcript to one operator component.
    keydevid 2544783 is a 105,420-character operator turn that is the entire WebSideStory Q4 2005
    call. The boilerplate is short (88% under 200 chars) and the misclassified content is long."""
    from ecalls.chunk.chunker import OPERATOR_KEEP_CHARS
    short = comp(1, 1, "operator", 1, "Your next question comes from Jane at Acme.")
    long = comp(2, 2, "operator", 1, "Welcome to the call. " * 200)   # 4,200 chars
    assert len(short["text"]) < OPERATOR_KEEP_CHARS < len(long["text"])
    units = build_units([short, long])
    assert [[c["component_id"] for c in u] for u in units] == [[2]]
    assert chunk_call([short], V) == []
    assert len(chunk_call([long], V)) >= 1


def test_keep_operator_still_keeps_the_short_ones():
    short = comp(1, 1, "operator", 1, "next question please")
    assert len(build_units([short], drop_operator=False)) == 1
