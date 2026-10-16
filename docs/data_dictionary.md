# Data dictionary

Three tables and the registry. Paths resolve from `$ECALLS_DATA_ROOT`, default `<repo>/data`.

## `processed/calls.parquet` — one row per call (412,255)

| Variable | Definition | Units | Source |
| --- | --- | --- | --- |
| `keydevid` | Capital IQ key development id. The real-world call event and the primary key. | id | `wrds_transcript_detail` |
| `transcriptid` | The version chosen for this call by rule `longest-total-chars-v1` (D4). | id | derived |
| `companyid` | Capital IQ company id. A call can map to several; this is the lowest after dedup. | id | `wrds_transcript_detail` |
| `gvkey`, `permno` | Compustat and CRSP identifiers, via CCM. `permno` is null outside the S&P 500 universe and is arbitrary for the 21 companyids mapping to more than one permno (L8). | id | link chain |
| `call_date`, `call_time_utc` | Date and time of the call. | date, time | `mostimportantdateutc` |
| `event_type_id`, `event_type` | 48, "Earnings Calls", the only type in this corpus. | id, text | lookup |
| `collection_type`, `presentation_type` | Capital IQ's version labels for the chosen transcript, e.g. "Audited Copy", "Final". | text | `wrds_transcript_detail` |
| `n_components` | Speaker turns in the chosen version, counted from what was pulled. | count | derived |
| `n_chars` | Characters after `ciq-raw-v1` normalization. | count | derived |
| `expected_components`, `expected_chars` | Stage 5b's server-side counts on raw text, before normalization. Kept beside the actuals so the 0.094% gap is computable (L4). | count | `ciqtranscriptcomponent` |
| `n_versions` | Candidate transcripts for this call. Mean 3.18. | count | derived |
| `margin_ratio` | Chosen characters over runner-up characters. Null when only one version. | ratio | derived |
| `tiebreak_used` | Which sort key decided: chars, n_components, presentation, created, transcriptid. Reads "chars" for single-version calls, which overstates that share (L8). | text | derived |
| `in_index_at_call` | **The point-in-time label.** True when any company route for the call belongs to a firm inside an S&P 500 membership spell on the call date, using the intersected link window. 37,920 calls. | bool | derived |
| `suspect_short`, `suspect_long` | Flags at 5,000 and 300,000 characters. Descriptive, never filters. | bool | derived |
| `shard`, `pull_ts`, `git_sha`, `cleaning_version`, `rule_version` | Provenance. | text | derived |

## `processed/components.parquet` — one row per speaker turn (22,790,065)

| Variable | Definition | Units | Source |
| --- | --- | --- | --- |
| `component_id` | `transcriptcomponentid`. Primary key. | id | `ciqtranscriptcomponent` |
| `component_order` | Position within the call. | ordinal | same |
| `component_type_id`, `component_type` | One of eight Capital IQ types, e.g. 3 "Question", 4 "Answer". All eight occur. | id, text | lookup, pinned in `lookups.py` |
| `section` | **Derived once at ingest** into `presentation`, `qa` or `operator`. The column most retrieval filters on. | text | derived |
| `person_id`, `speaker_name`, `speaker_company` | The speaker and their employer per Capital IQ. Null on about 58,000 turns. | id, text | `ciqtranscriptperson` |
| `speaker_type_id`, `speaker_type` | Operator, Executives, Analysts, Shareholders, Attendees, Unknown Speaker. | id, text | lookup |
| `text` | The turn, normalized by `ciq-raw-v1`. | text | `componenttext` |
| `n_chars` | `len(text)`. Zero on the 438 rows whose source text is NULL. | count | derived |
| `text_sha1` | SHA1 of the normalized text. Cache key for the chunk layer. | hex | derived |

## `processed/chunks/chunks_<YYYY>Q<Q>.parquet` — one row per retrieval chunk (12,102,243)

| Variable | Definition | Units | Source |
| --- | --- | --- | --- |
| `chunk_id` | `sha1(keydevid, first component_id, char_start, chunker_version)`. Deterministic, so an embedding cache survives a rebuild. | hex | derived |
| `component_ids` | The components this chunk draws on, in order. Exact provenance. | list of id | derived |
| `char_start`, `char_end` | Offsets into those components joined by a single newline. **`text` equals that join sliced by these offsets**, which is the reconstruction invariant. | count | derived |
| `text`, `n_chars` | The chunk and its length. Mean 1,293 characters. | text, count | derived |
| `n_tokens_est` | `n_chars / 4.703`. **An estimate**, from a measured rate (G1). | count | derived |
| `section`, `speaker_name`, `speaker_type` | Copied down for pre-filtered search. `speaker_type` reads "Mixed" when the chunk spans roles, which a question-and-answer chunk does by design. | text | derived |
| `keydevid`, `companyid`, `permno`, `call_date`, `in_index_at_call` | Filter columns copied from `calls.parquet` so retrieval needs no join. | mixed | join |
| `source_sha1` | SHA1 over the contributing `text_sha1` values. Invalidates an embedding when the source text changes. | hex | derived |
| `chunker_version` | `qa-pair-v2`. A bump changes every `chunk_id`. | text | derived |

## `registry/embeddings.json` — tracked in git, not under `data/`

One entry per embedding set: `model_id`, `kind` (`vintage` or `atemporal`), `training_cutoff`, `dim`,
`pooling`, `normalized`, `source`, and notes. Resolved by date or by name through
`ecalls.chunk.registry`, never by a supplied path. Currently the 26 ChronoBERT vintages, 1999 to 2024.
