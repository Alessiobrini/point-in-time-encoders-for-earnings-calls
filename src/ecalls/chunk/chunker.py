"""Phase 3: turn components into retrieval chunks.

Three properties this is built around.

**Chunks respect component boundaries, and a Question is kept with the Answers that follow it.**
Splitting an analyst's question away from the CFO's reply is the characteristic retrieval failure of
fixed-window chunking on this text, and the component structure hands us the fix for free.

**Every chunk is exactly reconstructible from the corpus.** `text` equals the concatenation of the
components named in `component_ids`, sliced by `char_start:char_end`. `tests/test_chunker.py` asserts
this on real calls, so a chunk can never drift from its source.

**Chunk ids are deterministic.** `sha1(keydevid, first component_id, char_start, chunker_version)`, so
the same corpus and chunker always give the same ids and an embedding cache stays valid.

Sizing is by characters, not tokens, using the measured 4.703 chars/token (cl100k over a 77 M-char
sample). Chunk boundaries fall at component or sentence boundaries, so exact token counts are not
needed to place them, and `n_tokens_est` is labelled an estimate. Presentation components average
4,605 characters with a maximum of 161,393 and 896,738 of 1.59M exceed one window, so a component
longer than `max_chars` is split at paragraph and then sentence boundaries, never mid-word.
"""
import hashlib
import re

CHARS_PER_TOKEN = 4.703          # measured, cl100k over 77 M chars from 6 shards
TARGET_TOKENS = 512
TARGET_CHARS = int(TARGET_TOKENS * CHARS_PER_TOKEN)       # 2408
MAX_CHARS = int(TARGET_CHARS * 1.25)                      # 3010, a soft ceiling before splitting

QUESTION_TYPE = 3
ANSWER_TYPES = (4, 8)            # Answer, Unknown Question and Answer Message

# Operator turns are boilerplate on average and content in the tail, so the drop is by length rather
# than by section. Measured over all 3,389,613 operator components: 88.1% are under 200 characters
# (mean 83) and carry the "your next question comes from" text, while 1,214 exceed 3,000 characters
# averaging 7,081. Capital IQ sometimes attributes an entire transcript to one operator component,
# most often in 2005 and 2006: keydevid 2544783 is a 105,420-character operator turn that is the whole
# WebSideStory Q4 2005 call. Dropping by section alone left 57 calls with no chunks at all, PepsiCo's
# among them. A 1,000-character floor keeps the 32,824 components (1.0%) that carry 56.5 M characters.
OPERATOR_KEEP_CHARS = 1000

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+")


def split_long_text(text: str, target: int = TARGET_CHARS, max_: int = MAX_CHARS) -> list[tuple[int, int]]:
    """Offsets of pieces no longer than `max_`, breaking at paragraphs then sentences.

    Returns (start, end) pairs into `text`, contiguous and covering it exactly, so the caller can
    slice without losing a character. A single sentence longer than `max_` is emitted whole rather
    than cut mid-word, because a truncated word is worse for retrieval than an over-long chunk.
    """
    if len(text) <= max_:
        return [(0, len(text))]

    # Candidate break points: paragraph boundaries first, then sentence boundaries inside them.
    breaks = [0]
    for m in _PARA.finditer(text):
        breaks.append(m.end())
    breaks.append(len(text))
    pieces = _pack_spans(text, breaks, target, max_)

    out = []
    for a, b in pieces:
        if b - a <= max_:
            out.append((a, b))
            continue
        sub = text[a:b]
        sb = [0] + [m.end() for m in _SENT.finditer(sub)] + [len(sub)]
        for c, d in _pack_spans(sub, sb, target, max_):
            out.append((a + c, a + d))
    return out


def _pack_spans(text: str, breaks: list[int], target: int, max_: int) -> list[tuple[int, int]]:
    """Greedily join adjacent break-delimited segments up to `target`, never past `max_`."""
    breaks = sorted(set(breaks))
    spans, start = [], breaks[0]
    for i in range(1, len(breaks)):
        end = breaks[i]
        nxt = breaks[i + 1] if i + 1 < len(breaks) else None
        if end - start >= target or nxt is None or (nxt - start) > max_:
            spans.append((start, end))
            start = end
    if start < breaks[-1]:
        spans.append((start, breaks[-1]))
    return [(a, b) for a, b in spans if b > a]


def build_units(components: list[dict], drop_operator: bool = True) -> list[list[dict]]:
    """Group a call's components into retrieval units.

    A `Question` opens a unit and the `Answer` components after it join it, so a question and its
    replies stay together. Consecutive presentation components form one unit.

    Operator turns are dropped by default, but only the short ones. See `OPERATOR_KEEP_CHARS`: the
    boilerplate is short and the misclassified whole-transcript cases are long, so the split is by
    length. Dropping is recorded in `chunker_version`, and the components themselves are untouched in
    the corpus, so this is reversible without a re-pull.
    """
    units: list[list[dict]] = []
    cur: list[dict] = []
    cur_kind = None

    for c in components:
        section = c["section"]
        if section == "operator":
            if drop_operator and len(c["text"]) < OPERATOR_KEEP_CHARS:
                if cur:
                    units.append(cur); cur, cur_kind = [], None
                continue
            kind = "operator"
        elif section == "qa":
            kind = "qa"
        else:
            kind = "presentation"

        starts_new = (
            kind != cur_kind
            or (kind == "qa" and int(c["component_type_id"]) == QUESTION_TYPE)
            or kind == "operator"
        )
        if starts_new and cur:
            units.append(cur); cur = []
        cur.append(c); cur_kind = kind
    if cur:
        units.append(cur)
    return units


def pack_unit(unit: list[dict], target: int = TARGET_CHARS,
              max_: int = MAX_CHARS) -> list[dict]:
    """Chunks for one unit. Whole components are grouped; an over-long one is split.

    `char_start` and `char_end` index into the concatenation of the components in `component_ids`,
    joined by a single newline, which is the invariant the reconstruction test relies on.
    """
    chunks, buf, buf_len = [], [], 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        joined = "\n".join(c["text"] for c in buf)
        chunks.append({"component_ids": [int(c["component_id"]) for c in buf],
                       "char_start": 0, "char_end": len(joined), "text": joined})
        buf, buf_len = [], 0

    for c in unit:
        t = c["text"]
        if len(t) > max_:
            flush()
            for a, b in split_long_text(t, target, max_):
                chunks.append({"component_ids": [int(c["component_id"])],
                               "char_start": a, "char_end": b, "text": t[a:b]})
            continue
        extra = len(t) + (1 if buf else 0)
        if buf and buf_len + extra > max_:
            flush()
            extra = len(t)
        buf.append(c); buf_len += extra
        if buf_len >= target:
            flush()
    flush()
    return chunks


def chunk_id(keydevid: int, first_component_id: int, char_start: int, version: str) -> str:
    """Deterministic, so the same corpus and chunker always give the same ids."""
    key = f"{int(keydevid)}|{int(first_component_id)}|{int(char_start)}|{version}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def chunk_call(components: list[dict], version: str, drop_operator: bool = True,
               target: int = TARGET_CHARS, max_: int = MAX_CHARS) -> list[dict]:
    """All chunks for one call. `components` must be ordered by `component_order`."""
    if not components:
        return []
    kd = int(components[0]["keydevid"])
    out = []
    for unit in build_units(components, drop_operator=drop_operator):
        first = unit[0]
        for ch in pack_unit(unit, target, max_):
            ids = ch["component_ids"]
            members = [c for c in unit if int(c["component_id"]) in set(ids)]
            speakers = {c.get("speaker_name") for c in members if c.get("speaker_name")}
            roles = {c.get("speaker_type") for c in members if c.get("speaker_type")}
            out.append({
                "chunk_id": chunk_id(kd, ids[0], ch["char_start"], version),
                "keydevid": kd,
                "section": first["section"],
                "component_ids": ids,
                "n_components": len(ids),
                "char_start": ch["char_start"],
                "char_end": ch["char_end"],
                "text": ch["text"],
                "n_chars": len(ch["text"]),
                "n_tokens_est": int(round(len(ch["text"]) / CHARS_PER_TOKEN)),
                "speaker_name": sorted(speakers)[0] if len(speakers) == 1 else None,
                "speaker_type": sorted(roles)[0] if len(roles) == 1 else "Mixed",
                "source_sha1": hashlib.sha1(
                    "|".join(str(c.get("text_sha1", "")) for c in members).encode()).hexdigest(),
                "chunker_version": version,
            })
    return out
