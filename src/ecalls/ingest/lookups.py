"""Decode tables and text normalization for the component layer.

The lookup tables are tiny and never change, so they are pinned here rather than joined per batch.
Read from WRDS on 2026-09-25 and asserted against the live tables by
`ecalls.ingest.verify_measurements`.
"""
import hashlib
import unicodedata

# ciq_transcripts.ciqtranscriptcomponenttype, read 2026-09-25
COMPONENT_TYPE = {
    1: "Presentation Operator Message",
    2: "Presenter Speech",
    3: "Question",
    4: "Answer",
    5: "Presentation Section",
    6: "Question And Answer Section",
    7: "Question and Answer Operator Message",
    8: "Unknown Question and Answer Message",
}

# ciq_transcripts.ciqtranscriptspeakertype, read 2026-09-25
SPEAKER_TYPE = {
    1: "Operator",
    2: "Executives",
    3: "Analysts",
    4: "Shareholders",
    5: "Attendees",
    6: "Unknown Speaker",
}

# Derived once at ingest, because "give me the Q and A and drop the operator boilerplate" is the
# most common query against this corpus and every consumer would otherwise re-derive it slightly
# differently.
#
# Measured over ALL 22,790,065 components on 2026-09-26, which corrects an earlier note here that
# generalized from a 1,500-call sample: every one of the eight types occurs. Answer 10,320,503,
# Question 7,371,495, Q&A Operator Message 3,016,161, Presenter Speech 1,586,663, Presentation
# Operator Message 373,452, Unknown Q&A Message 118,372, Presentation Section 2,002, Q&A Section
# 1,417. The sample contained none of the last three, which is why they were called unused.
#
# The same note claimed the presentation section carries only Executives. It does not: 1,529,465
# Executive turns, but also 34,418 Analysts, 21,354 Attendees, 1,313 Unknown Speaker and 101
# Shareholders, plus about 58,000 turns across all sections with no speaker type at all. The mapping
# itself is sound (zero unmapped types, zero null sections), and `tests/test_lookups.py` now checks
# it against the corpus rather than against itself.
SECTION = {
    1: "operator",
    2: "presentation",
    3: "qa",
    4: "qa",
    5: "presentation",
    6: "qa",
    7: "operator",
    8: "qa",
}


def normalize_text(s: str | None) -> str:
    """cleaning_version "ciq-raw-v1": CRLF to LF, Unicode NFC, trailing whitespace stripped per
    component. Nothing else. No entity unescaping, no boilerplate removal, no speaker-prefix
    stripping. The corpus layer is archival, so every lossy step belongs in the chunk layer where
    redoing it costs CPU rather than a WRDS pull (docs/decisions.md, D6).

    A non-string input returns "". The type check is not decoration: `componenttext` is nullable in
    Capital IQ, pandas turns a NULL into NaN when it builds the column, and `not float("nan")` is
    False, so an `if not s` guard passes NaN straight through to `.replace` and the pull dies. A
    component with no text is a real row and is kept with n_chars 0 for the audit to flag, never
    dropped."""
    if not isinstance(s, str) or not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = unicodedata.normalize("NFC", s)
    return "\n".join(line.rstrip() for line in s.split("\n")).rstrip()


def text_sha1(s: str) -> str:
    """Stable cache key for the chunk and embed layers, computed on the NORMALIZED text."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()
