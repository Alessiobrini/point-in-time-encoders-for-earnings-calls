"""Edge cases for the component frame builder. Pure, no database."""
import datetime as dt
import sys

import pandas as pd
import pyarrow as pa
import pytest

from ecalls.config import constants as C
from ecalls.config.schemas import COMPONENTS
from ecalls.ingest.pull_components import build_frame

D = dt.date

# (transcriptid, component_id, component_order, component_type_id, person_id, raw_text,
#  speaker_name, speaker_type_id, speaker_company)
ROWS = [
    (500, 9001, 1, 1, 70, "Good morning.\r\n", "Operator", 1, None),
    (500, 9002, 2, 2, 71, "Thanks.  Revenue grew.   ", "Jane Doe", 2, "Acme Inc."),
    (500, 9003, 3, 3, 72, "What about margins?", "John Ray", 3, "Big Bank"),
    (500, 9004, 4, 4, 71, "Margins improved.", "Jane Doe", 2, "Acme Inc."),
]
META = pd.DataFrame([{"keydevid": 111, "companyid": 222, "call_date": D(2024, 2, 1),
                      "chosen_transcriptid": 500, "shard": "2024Q1"}])


@pytest.fixture
def df():
    return build_frame(ROWS, META, "2024Q1")


def test_call_identifiers_are_attached_to_every_component(df):
    assert len(df) == 4
    assert set(df["keydevid"]) == {111}
    assert set(df["companyid"]) == {222}
    assert set(df["call_date"]) == {D(2024, 2, 1)}


def test_text_is_normalized_and_measured(df):
    row = df.loc[df.component_id == 9002].iloc[0]
    assert row["text"] == "Thanks.  Revenue grew."      # trailing stripped, interior kept
    assert row["n_chars"] == len(row["text"])
    assert len(row["text_sha1"]) == 40


def test_types_and_sections_are_decoded(df):
    got = dict(zip(df.component_id, df.section))
    assert got == {9001: "operator", 9002: "presentation", 9003: "qa", 9004: "qa"}
    assert df.loc[df.component_id == 9003, "component_type"].iloc[0] == "Question"
    assert df.loc[df.component_id == 9003, "speaker_type"].iloc[0] == "Analysts"


def test_output_is_ordered_by_call_then_turn(df):
    assert list(df["component_order"]) == [1, 2, 3, 4]


def test_provenance_columns_are_set(df):
    assert set(df["shard"]) == {"2024Q1"}
    assert set(df["cleaning_version"]) == {C.CLEANING_VERSION}
    assert df["pull_ts"].notna().all() and df["git_sha"].notna().all()


def test_frame_conforms_to_the_declared_schema(df):
    """The reason schemas.py exists: a shard typed wrong would silently coerce the union of 88."""
    cols = [f.name for f in COMPONENTS]
    tbl = pa.Table.from_pandas(df[cols], schema=COMPONENTS, preserve_index=False)
    assert tbl.num_rows == 4
    assert tbl.schema.equals(COMPONENTS)


def test_an_all_null_person_column_still_types_as_int64():
    """The exact failure schemas.py guards: pandas would infer float64 from all-null and the
    union across shards would then coerce."""
    rows = [(500, 9001, 1, 2, None, "text", None, None, None)]
    out = build_frame(rows, META, "2024Q1")
    cols = [f.name for f in COMPONENTS]
    tbl = pa.Table.from_pandas(out[cols], schema=COMPONENTS, preserve_index=False)
    assert tbl.schema.field("person_id").type == pa.int64()
    assert tbl.column("person_id").null_count == 1


def test_a_component_whose_transcript_is_not_in_meta_is_dropped():
    """An inner join, so a stray transcriptid cannot enter the corpus without call identifiers."""
    rows = ROWS + [(999, 9099, 1, 2, 70, "orphan", "X", 2, None)]
    out = build_frame(rows, META, "2024Q1")
    assert len(out) == 4
    assert 999 not in set(out["transcriptid"])


def test_empty_rows_returns_empty_without_raising():
    assert build_frame([], META, "2024Q1").empty


if __name__ == "__main__":
    print("run with pytest")
    sys.exit(0)


def test_shard_is_complete_requires_coverage_not_just_existence(tmp_path):
    """Regression. Shard names are calendar quarters with no universe in them, so the S&P 500
    checkpoint writes components_2024Q1.parquet holding only S&P 500 calls. A later full-corpus run
    that skipped on existence alone would drop every other company's calls in that quarter and
    report success."""
    import pyarrow.parquet as pq
    from ecalls.ingest.pull_components import shard_is_complete

    path = tmp_path / "components_2024Q1.parquet"
    assert shard_is_complete(path, 10) is False          # missing file

    df = build_frame(ROWS, META, "2024Q1")               # one call, keydevid 111
    cols = [f.name for f in COMPONENTS]
    pq.write_table(pa.Table.from_pandas(df[cols], schema=COMPONENTS, preserve_index=False), path)

    assert shard_is_complete(path, 1) is True            # holds the one call expected
    assert shard_is_complete(path, 2) is False           # a wider universe expects more


def test_shard_is_complete_treats_an_unreadable_file_as_incomplete(tmp_path):
    path = tmp_path / "components_2024Q1.parquet"
    path.write_bytes(b"not parquet")
    from ecalls.ingest.pull_components import shard_is_complete
    assert shard_is_complete(path, 1) is False
