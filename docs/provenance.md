# Provenance of raw data

No raw vendor file is stored on disk. Every pull aggregates or filters server-side on WRDS and returns
rows already reduced to this project's schema, so the provenance record is the manifest rather than a
file checksum.

## The record

`data/raw/_manifest.jsonl` carries one JSON object per artifact written, with the stage, the artifact
name, row counts, the pull timestamp and the git SHA of the code that produced it. It is JSONL rather
than CSV because each stage records different fields and a single CSV header cannot describe
heterogeneous rows. `data/raw/_manifest_legacy.csv` is the earlier ragged CSV, kept unaltered.

`data/raw/ciq/_run_state.json` carries the incremental-refresh state: the watermark
(`max_creation_date`), the margin applied, and per-year transcript counts used as the reconciliation
backstop.

## Sources

| Artifact | WRDS source | Coverage | Access |
| --- | --- | --- | --- |
| `universe/sp500_membership.parquet` | `crsp.dsp500list_v2` | 2,084 spells, 1,956 permnos, horizon 2025-12-31 | SQLAlchemy over psycopg2, credentials from `~/.pgpass` |
| `universe/_unlinked_permnos.csv` | derived | the 9 index permnos with no usable CCM link | computed |
| `ciq/transcript_index.parquet` | `ciq_transcripts.wrds_transcript_detail` | 1,311,639 earnings-call transcript versions, 412,255 calls | filtered on `keydeveventtypeid = 48` |
| `ciq/lengths/lengths_<year>.parquet` | `ciq_transcripts.ciqtranscriptcomponent` | 1,305,765 rows | server-side `count(*)` and `sum(length(componenttext))`, so the text crosses the wire once |
| `ciq/components/components_<YYYY>Q<Q>.parquet` | `ciq_transcripts.ciqtranscriptcomponent` joined to `ciqtranscriptperson` | 22,790,065 components, 16.01 G characters, 88 shards | batches of 250 transcriptids, one connection |
| link chain | `crsp.ccmxpf_lnkhist`, `ciq_common.wrds_gvkey`, `comp.idxcst_his` | 996 permnos in, 987 linked, 992 companyids | `linktype` in LU/LC/LS and `linkprim` in P/C |

## Cleaning

`cleaning_version = "ciq-raw-v1"`: CRLF to LF, Unicode NFC, trailing whitespace stripped per component.
Nothing else. Bump the string rather than changing what it means.

## No silent re-pulls

A re-pull writes a new manifest row with its own timestamp and git SHA. Any change to cleaning takes a
new `cleaning_version`. A resume marker compares coverage rather than file existence, so a re-run with
wider scope rewrites rather than skipping, and says so.

## Licensing

The transcripts and the market data are licensed, and none of them is redistributed here.
