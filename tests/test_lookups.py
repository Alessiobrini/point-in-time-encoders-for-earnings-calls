"""Edge cases for the component decode tables and text normalization. Pure, no database."""
import sys

from ecalls.ingest.lookups import (
    COMPONENT_TYPE, SECTION, SPEAKER_TYPE, normalize_text, text_sha1,
)


def test_every_component_type_has_a_section():
    """An unmapped type would become a null section and silently drop out of a Q&A filter."""
    assert set(SECTION) == set(COMPONENT_TYPE)
    assert all(v in {"presentation", "qa", "operator"} for v in SECTION.values())


def test_the_section_mapping_is_what_we_intend():
    """A change-detector on the constant, named honestly. The real validation against the corpus is
    `test_section_mapping_against_the_corpus` below, and against the live WRDS lookup tables it is
    `verify_measurements.check_lookups`. An earlier version of this test claimed to be 'measured over
    1,500 calls' while reading no data at all."""
    assert SECTION[1] == "operator"        # Presentation Operator Message
    assert SECTION[7] == "operator"        # Question and Answer Operator Message
    assert SECTION[2] == "presentation"    # Presenter Speech
    assert SECTION[3] == SECTION[4] == "qa"  # Question, Answer
    assert SECTION[5] == "presentation" and SECTION[6] == "qa" and SECTION[8] == "qa"


def test_section_mapping_against_the_corpus(components_path):
    """The real check: every component type present in the corpus decodes, no section is null, and
    each section's derived value agrees with SECTION for the type that produced it. Skips when the
    corpus is not on this machine."""
    import duckdb
    con = duckdb.connect()
    con.execute("set memory_limit='4GB'")
    rows = con.execute(
        f"select component_type_id, any_value(section) s, count(distinct section) n_sections, "
        f"count(*) n from read_parquet('{components_path}') group by 1 order by 1").fetchall()
    assert rows, "corpus has no components"
    for type_id, section, n_sections, _n in rows:
        assert n_sections == 1, f"type {type_id} maps to {n_sections} sections in the data"
        assert section is not None, f"type {type_id} produced a null section"
        assert section == SECTION[int(type_id)], (
            f"type {type_id} is {section!r} in the corpus but {SECTION[int(type_id)]!r} in SECTION")
    assert {int(r[0]) for r in rows} <= set(SECTION), "corpus holds a component type we do not map"


def test_speaker_types_are_the_six_ciq_defines():
    assert len(SPEAKER_TYPE) == 6
    assert SPEAKER_TYPE[2] == "Executives" and SPEAKER_TYPE[3] == "Analysts"


def test_crlf_becomes_lf_and_bare_cr_too():
    assert normalize_text("a\r\nb") == "a\nb"
    assert normalize_text("a\rb") == "a\nb"


def test_trailing_whitespace_stripped_per_line_and_at_the_end():
    assert normalize_text("a   \nb\t\n\n\n") == "a\nb"


def test_leading_whitespace_is_preserved():
    """ciq-raw-v1 strips trailing only. Leading indentation is content we do not judge."""
    assert normalize_text("   a\nb") == "   a\nb"


def test_nfc_normalization_collapses_decomposed_accents():
    decomposed = "café"          # e + combining acute
    composed = "café"             # precomposed e-acute
    assert normalize_text(decomposed) == composed
    assert text_sha1(normalize_text(decomposed)) == text_sha1(composed)


def test_empty_and_none_both_give_empty_string():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("   \n  ") == ""


def test_hash_is_deterministic_and_differs_on_different_text():
    assert text_sha1("abc") == text_sha1("abc")
    assert text_sha1("abc") != text_sha1("abd")
    assert len(text_sha1("abc")) == 40


def test_hash_is_computed_on_normalized_text_so_crlf_does_not_change_it():
    """Otherwise the chunk-layer cache key would miss whenever a line ending differed."""
    assert text_sha1(normalize_text("a\r\nb")) == text_sha1(normalize_text("a\nb"))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


def test_a_nan_returns_empty_rather_than_raising():
    """Regression. componenttext is nullable, pandas turns NULL into NaN when building the column,
    and `not float("nan")` is False, so an `if not s` guard let NaN reach .replace and killed the
    pull 29 shards in."""
    assert normalize_text(float("nan")) == ""
    assert normalize_text(0.0) == ""
    assert normalize_text(123) == ""
