"""Declared pyarrow schemas, one per artifact, enforced at write time.

The component pull writes roughly 88 shards that DuckDB later unions. A single shard whose
`person_id` came back all null would be typed float64 by pandas and would silently coerce the union,
so every shard is written against the schema below rather than against whatever pandas inferred.
"""
import pyarrow as pa

COMPONENTS = pa.schema([
    # identifiers
    ("keydevid", pa.int64()),
    ("transcriptid", pa.int64()),
    ("companyid", pa.int64()),
    ("call_date", pa.date32()),
    ("component_id", pa.int64()),
    ("component_order", pa.int32()),
    # what kind of turn this is
    ("component_type_id", pa.int16()),
    ("component_type", pa.string()),
    ("section", pa.string()),          # derived: presentation / qa / operator
    # who is speaking
    ("person_id", pa.int64()),         # nullable
    ("speaker_type_id", pa.int16()),   # nullable
    ("speaker_type", pa.string()),     # nullable
    ("speaker_name", pa.string()),     # nullable
    ("speaker_company", pa.string()),  # nullable, the person's employer per CIQ
    # the text
    ("text", pa.large_string()),
    ("n_chars", pa.int32()),
    ("text_sha1", pa.string()),
    # provenance
    ("shard", pa.string()),
    ("cleaning_version", pa.string()),
    ("pull_ts", pa.string()),
    ("git_sha", pa.string()),
])


CHUNKS = pa.schema([
    ("chunk_id", pa.string()),
    # filter columns, copied down so retrieval can pre-filter without a join
    ("keydevid", pa.int64()),
    ("companyid", pa.int64()),
    ("permno", pa.int32()),          # nullable
    ("call_date", pa.date32()),
    ("in_index_at_call", pa.bool_()),
    ("section", pa.string()),
    ("speaker_name", pa.string()),   # nullable, null when the chunk spans speakers
    ("speaker_type", pa.string()),   # "Mixed" when the chunk spans roles
    # provenance back to components.parquet, exact
    ("component_ids", pa.list_(pa.int64())),
    ("n_components", pa.int32()),
    ("char_start", pa.int32()),
    ("char_end", pa.int32()),
    ("source_sha1", pa.string()),
    # the text
    ("text", pa.large_string()),
    ("n_chars", pa.int32()),
    ("n_tokens_est", pa.int32()),
    ("chunker_version", pa.string()),
    ("shard", pa.string()),
    ("pull_ts", pa.string()),
    ("git_sha", pa.string()),
])
