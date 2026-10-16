"""The searchable index: LanceDB, hybrid lexical plus dense, with point-in-time refusals built in.

The index is disposable and rebuildable, which is why it can afford opinions the corpus cannot. Two of
them are load-bearing.

**Retrieval is hybrid, not pure dense (D17).** Measured on this corpus, BM25 with a real retriever
reaches 0.5544 within-call recall@1 against 0.5102 for that retriever alone and 0.4804 for BM25 alone,
and the margin is widest exactly where a dense index is weakest, cross-call 0.3849 against 0.2967. So the
table carries a full-text index beside the vectors and `search` combines them.

**A point-in-time query refuses rather than warns.** `search(..., as_of=d)` enforces two separate things
that are easy to confuse:

1. **No future documents.** Results are filtered to `call_date <= as_of`. This is the obvious leak and
   the easy one.
2. **No future model.** The embedding model's own training cutoff must be at or before `as_of`, or the
   query raises. This is the leak the whole project exists to control, and it cannot be fixed by
   filtering rows: a 2024-cutoff encoder embedding a 2015 call has already read 2016 to 2024 language,
   so every vector in the table is contaminated for a 2015 decision no matter which rows come back.

An index built from an `ATEMPORAL` model, which is any strong retriever since none of them report a
cutoff, therefore refuses **every** `as_of` query. That is correct rather than inconvenient: it is
usable for search and teaching and barred from backtests, which is exactly what D16 assigns it.

Run:
    python -m ecalls.chunk.index build --model bge-small-en-v1.5
    python -m ecalls.chunk.index search --model bge-small-en-v1.5 --query "margin pressure from freight"
"""
import argparse
import datetime as dt
import json

import numpy as np
import polars as pl

from ecalls.chunk import registry as R
from ecalls.config import paths as P
from ecalls.embed import encode as E
from ecalls.ingest.manifest import git_sha, utc_now

# Columns copied into the index so retrieval can pre-filter without joining back to the chunk layer.
FILTER_COLUMNS = ["keydevid", "companyid", "permno", "call_date", "in_index_at_call",
                 "section", "speaker_type", "speaker_name"]
TEXT_COLUMN = "text"
META_FILE = "_index_meta.json"
MIN_ANN_ROWS = 50_000   # below this, PQ training is unreliable or impossible and an exact scan is fast

# Query-time ANN defaults, measured rather than inherited from the library. See DECISIONS D18: at the
# library default of nprobes=20 with no refine, recall against exact search is 0.62 mean and 0.00 at
# worst, which would silently return half the wrong neighbours. Raising nprobes alone barely helps,
# because the loss is PQ quantization error rather than too few partitions probed.
NPROBES = 400
REFINE_FACTOR = 50

# Chunks shorter than this are left out of the index. **Not out of the corpus**: the project rule is that
# ingest is maximal and filtering is downstream, and the index is the downstream. 50 is chosen by reading
# the length bands rather than by rounding. Under 50 characters is almost entirely pleasantries: "Okay.",
# "Yes.", "Okay. Thank you.", and "Thank you." alone appears 12,404 times. The 50-to-100 band is mixed,
# holding real content such as "Is there a 2007 target for multi-branding? I think it's in the several
# hundred range." beside transitions, so filtering at 100 would discard genuine short answers to save
# 0.18% more characters. At 50 the index loses 3.07% of chunks holding 0.069% of the corpus characters.
MIN_INDEX_CHARS = 50


def index_dir(model_id: str):
    return P.INDEX_DIR / model_id


def table_name(model_id: str) -> str:
    """LanceDB table names are filesystem-ish, and a model id with dots is legal but noisy."""
    return "chunks"


# --- building ------------------------------------------------------------------------------------

def table_list(db) -> list[str]:
    """Table names, across LanceDB versions.

    `table_names()` is deprecated in favour of `list_tables()`, and the replacement is **not** a drop-in:
    it returns a `ListTablesResponse` rather than a list, so `name in db.list_tables()` is always False.
    Swapping one for the other to silence the warning made every build recreate the table with
    `mode="overwrite"`, silently discarding the shards already indexed. A test caught it; nothing else
    would have.
    """
    try:
        resp = db.list_tables()
    except AttributeError:
        return list(db.table_names())
    return list(getattr(resp, "tables", resp))


def arrow_schema(dim: int):
    import pyarrow as pa
    return pa.schema([
        ("chunk_id", pa.string()),
        ("vector", pa.list_(pa.float32(), dim)),
        (TEXT_COLUMN, pa.large_string()),
        ("keydevid", pa.int64()),
        ("companyid", pa.int64()),
        ("permno", pa.int32()),
        ("call_date", pa.date32()),
        ("in_index_at_call", pa.bool_()),
        ("section", pa.string()),
        ("speaker_type", pa.string()),
        ("speaker_name", pa.string()),
    ])


def shard_frame(shard: str, model_id: str, min_chars: int = MIN_INDEX_CHARS) -> pl.DataFrame:
    """One shard's rows, joining the chunk layer to its embeddings on `chunk_id`.

    The join is inner and then checked, because a silent row drop here is the failure this project keeps
    finding: the index would simply be short and every recall number computed from it would be wrong
    with nothing reporting a gap.
    """
    chunks = pl.read_parquet(P.chunks_shard(shard),
                             columns=["chunk_id", TEXT_COLUMN] + FILTER_COLUMNS)
    part = E.part_path(model_id, shard)
    if not part.exists():
        raise FileNotFoundError(
            f"no embeddings for {shard} under {part}. Build the embedding set first, and run "
            "`python -m ecalls.audit.embeddings` to confirm it is complete.")
    ids, vecs = E.read_vectors(part)
    emb = pl.DataFrame({"chunk_id": ids, "vector": [v for v in vecs.astype(np.float32)]})
    out = chunks.join(emb, on="chunk_id", how="inner")
    if out.height != chunks.height:
        raise ValueError(
            f"{shard}: {chunks.height} chunks but only {out.height} joined to an embedding. The "
            "embedding set does not cover this shard; the index would be silently short.")
    # The coverage check above runs before this filter, on purpose. Dropping trivial chunks is a
    # deliberate choice about the index, and folding it into the same comparison would hide a genuine
    # gap in the embedding set behind an intended one.
    if min_chars:
        out = out.filter(pl.col("text").str.len_chars() >= min_chars)
    return out


def read_meta(model_id: str) -> dict:
    f = index_dir(model_id) / META_FILE
    return json.loads(f.read_text()) if f.exists() else {}


def write_meta(model_id: str, entry: R.Entry, n_rows: int, dim: int,
               shards: list[str], fts: bool, ann: bool = False,
               min_chars: int = MIN_INDEX_CHARS) -> None:
    d = index_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / META_FILE).write_text(json.dumps({
        "model_id": model_id,
        "kind": entry.kind,
        "training_cutoff": entry.training_cutoff,
        "dim": dim,
        "pooling": entry.pooling,
        "normalized": entry.normalized,
        "n_rows": n_rows,
        "shards": sorted(shards),          # which shards are in, so a resume adds rather than repeats
        "hybrid": fts,
        "ann": ann,
        "min_chars": min_chars,
        "built_at": utc_now(),
        "git_sha": git_sha(),
    }, indent=2) + "\n")


def build_ann(tbl, quiet: bool = False) -> bool:
    """An approximate-nearest-neighbour index on the vector column.

    Without it a vector query is a brute-force scan: **measured** 2.26s over 12,102,243 vectors, against
    0.005s for the lexical half, which has an inverted index. The hybrid search inherits the slower of
    the two, so the dense side governs interactive latency.

    IVF_PQ rather than IVF_FLAT, because flat keeps every full vector in the index and the point of the
    index is to not read them all. **ANN is approximate, so this trades recall for latency, and the
    trade is measured rather than assumed**: see `ann_recall` and the figure recorded in DECISIONS D18.
    """
    from lancedb.index import IvfPq

    n = tbl.count_rows()
    if n < MIN_ANN_ROWS:
        # PQ training needs at least 256 rows and fails outright below that, and well past 256 a brute
        # force scan is faster than the index lookup anyway. Skipping is the right answer rather than an
        # error, so building an index over one quarter for a test or a spot check just works.
        if not quiet:
            print(f"  skipping the ANN index: {n:,} rows is below {MIN_ANN_ROWS:,}, and an exact scan "
                  "at that size is already fast", flush=True)
        return False
    partitions = max(1, min(4096, int(n ** 0.5)))   # the usual sqrt(n) heuristic, capped
    if not quiet:
        print(f"  building the ANN index over {n:,} vectors, {partitions} partitions", flush=True)
    tbl.create_index("vector", config=IvfPq(num_partitions=partitions, num_sub_vectors=48),
                     replace=True)
    return True


def ann_recall(model_id: str, n_queries: int = 50, k: int = 10, seed: int = 0,
               nprobes: int = NPROBES, refine_factor: int = REFINE_FACTOR) -> dict:
    """Overlap between the ANN top-k and the exact top-k, on queries drawn from the table itself.

    Reported rather than trusted, because an approximate index silently returns slightly wrong answers
    and nothing downstream can tell. Queries are existing vectors, so the exact answer is known to
    contain the query itself and the comparison is against a real neighbour list.
    """
    import numpy as np

    tbl = open_table(model_id)
    rng = np.random.default_rng(seed)
    sample = tbl.search().limit(max(n_queries * 20, 1000)).select(["chunk_id", "vector"]).to_polars()
    idx = rng.choice(sample.height, size=min(n_queries, sample.height), replace=False)

    overlaps = []
    for i in idx:
        v = np.asarray(sample["vector"][int(i)], dtype=np.float32)
        approx = (tbl.search(v).nprobes(nprobes).refine_factor(refine_factor)
                  .limit(k).to_polars()["chunk_id"].to_list())
        exact = tbl.search(v).bypass_vector_index().limit(k).to_polars()["chunk_id"].to_list()
        overlaps.append(len(set(approx) & set(exact)) / k)
    a = np.array(overlaps)
    return {"n_queries": len(a), "k": k, "nprobes": nprobes, "refine_factor": refine_factor,
            "recall_mean": float(a.mean()), "recall_min": float(a.min()),
            "recall_p10": float(np.percentile(a, 10))}


def build(model_id: str, shards: list[str] | None = None, fts: bool = True, ann: bool = True,
          overwrite: bool = False, quiet: bool = False,
          min_chars: int = MIN_INDEX_CHARS) -> dict:
    """Build or extend the index for one embedding set.

    Idempotent per shard, tracked by an explicit shard list in the sidecar. That is deliberate rather
    than lazy: appending a shard twice duplicates rows, and no count-based check on a 12 M row table
    would notice, so the marker has to name what is in rather than how much.
    """
    import lancedb

    entry = R.load().by_name(model_id)
    shards = shards or E.shard_keys()
    d = index_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(d))
    name = table_name(model_id)

    existing = name in table_list(db)
    if existing and overwrite:
        db.drop_table(name)
        existing = False

    # Which shards are already in, from the sidecar rather than from a scan of 12 M rows. Appending a
    # shard twice would duplicate rows, which no count-based check on the whole table would notice, so
    # the resume marker is an explicit shard list written at the end of each successful build.
    done = set(read_meta(model_id).get("shards", [])) if existing else set()
    todo = [s for s in shards if s not in done]
    if not todo:
        tbl = db.open_table(name)
        if not quiet:
            print(f"  all {len(shards)} requested shards are already indexed, {tbl.count_rows():,} rows")
        write_meta(model_id, entry, tbl.count_rows(), entry.dim, sorted(done), fts, ann, min_chars)
        return {"model_id": model_id, "n_rows": tbl.count_rows(), "shards": 0, "dir": str(d)}

    tbl = db.open_table(name) if existing else None
    total = tbl.count_rows() if tbl is not None else 0
    for s in todo:
        frame = shard_frame(s, model_id, min_chars)
        total += frame.height
        arrow = frame.select(["chunk_id", "vector", TEXT_COLUMN] + FILTER_COLUMNS).to_arrow()
        arrow = arrow.cast(arrow_schema(entry.dim))
        if tbl is None:
            tbl = db.create_table(name, data=arrow, mode="overwrite")
        else:
            tbl.add(arrow)
        done.add(s)
        if not quiet:
            print(f"  {s}: +{frame.height:,} rows, {total:,} total", flush=True)

    if tbl is None:
        raise SystemExit("no shards were indexed")

    if fts:
        from lancedb.index import FTS
        if not quiet:
            print("  building the full-text index, which is what makes the search hybrid (D17)",
                  flush=True)
        tbl.create_index(TEXT_COLUMN, config=FTS(), replace=True)

    ann_built = build_ann(tbl, quiet=quiet) if ann else False

    n = tbl.count_rows()
    write_meta(model_id, entry, n, entry.dim, sorted(done), fts, ann_built, min_chars)
    return {"model_id": model_id, "n_rows": n, "shards": len(todo), "dir": str(d)}


# --- searching -----------------------------------------------------------------------------------

_ENCODER_CACHE: dict = {}


def cached_encoder(entry: R.Entry, device=None):
    """The encoder, loaded once per process.

    `search` used to load it on every call, which is **measured** at 2.74s against 0.006s to actually
    embed the query. For a batch job that is noise; for anything interactive it is the whole latency.
    Keyed on model and device so switching either still reloads.
    """
    key = (entry.model_id, device)
    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE[key] = E.load_encoder(entry, device)
    return _ENCODER_CACHE[key]


def open_table(model_id: str):
    import lancedb
    d = index_dir(model_id)
    if not (d / META_FILE).exists():
        raise FileNotFoundError(
            f"no index for {model_id} at {d}. Build it with "
            f"`python -m ecalls.chunk.index build --model {model_id}`.")
    db = lancedb.connect(str(d))
    return db.open_table(table_name(model_id))


def check_point_in_time(entry: R.Entry, as_of: dt.date) -> None:
    """Refuse a query whose embedding model saw text after the decision date.

    Filtering rows to `call_date <= as_of` does not fix this and is the mistake worth naming. If a
    2024-cutoff encoder produced the vectors, then a 2015 call's vector was written by a model that had
    read 2016 to 2024 language, so the *representation* carries the future even when the *document* does
    not. The only fix is a different model, which is why the registry resolves one by date.
    """
    if entry.kind != R.VINTAGE or entry.cutoff is None:
        raise ValueError(
            f"{entry.model_id} is registered as {entry.kind} with no training cutoff, so it cannot "
            f"support an as_of query. Use it without as_of for search and teaching, and use a vintage "
            f"for anything predictive: registry.for_date({as_of}) resolves one.")
    if entry.cutoff > as_of:
        raise ValueError(
            f"{entry.model_id} was trained on text through {entry.cutoff}, which is after "
            f"as_of={as_of}. Every vector in this index is contaminated for that decision date, and "
            f"filtering rows by call_date does not undo it. Resolve the right model with "
            f"registry.for_date({as_of}) and build its index.")


def search(query: str, model_id: str, k: int = 10, as_of=None, mode: str = "hybrid",
           where: str | None = None, device=None, nprobes: int = NPROBES,
           refine_factor: int = REFINE_FACTOR):
    """Top-k chunks for a text query.

    `as_of` turns on the point-in-time discipline: it filters to calls on or before that date **and**
    requires the embedding model's cutoff to be at or before it. Omitting it is exploration mode, which
    is the right thing for search and teaching and the wrong thing for a backtest.

    `nprobes` and `refine_factor` default to values measured against exact search rather than to the
    library's, which return 0.62 recall here. Lowering them trades accuracy for latency, and `ann_recall`
    is how to find out what the trade costs before making it.
    """
    if mode not in ("hybrid", "vector", "fts"):
        raise ValueError(f"mode must be hybrid, vector or fts, got {mode!r}")
    entry = R.load().by_name(model_id)
    tbl = open_table(model_id)

    clauses = [where] if where else []
    if as_of is not None:
        as_of = dt.date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
        check_point_in_time(entry, as_of)
        clauses.append(f"call_date <= date '{as_of.isoformat()}'")
    predicate = " AND ".join(f"({c})" for c in clauses) if clauses else None

    if mode == "fts":
        if not read_meta(model_id).get("hybrid", False):
            raise ValueError(
                f"the index for {model_id} was built with --no-fts, so there is no lexical index to "
                "search. Rebuild without that flag, or use mode='vector'.")
        q = tbl.search(query, query_type="fts")
    else:
        tok, model, dev = cached_encoder(entry, device)
        prefix = (entry.extra or {}).get("query_prefix", "")
        vec = E.encode_texts([prefix + query], tok, model, dev,
                             max_length=E.effective_max_length(entry),
                             pooling=entry.pooling)[0]
        if mode == "vector":
            q = tbl.search(vec, query_type="vector")
        else:
            q = tbl.search(query_type="hybrid").vector(vec).text(query)
        # Applied only when an ANN index exists. On an exact scan they are meaningless, and on a table
        # small enough to have skipped the index they would raise.
        if read_meta(model_id).get("ann"):
            q = q.nprobes(nprobes).refine_factor(refine_factor)

    if predicate:
        q = q.where(predicate, prefilter=True)
    return q.limit(k).to_polars()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--model", required=True)
    b.add_argument("--shards", nargs="*", default=None)
    b.add_argument("--no-fts", action="store_true", help="skip the lexical index, leaving it dense only")
    b.add_argument("--no-ann", action="store_true",
                    help="skip the approximate vector index, leaving vector search exact but slow")
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--min-chars", type=int, default=MIN_INDEX_CHARS,
                    help="leave chunks shorter than this out of the index. 0 indexes everything. The "
                         "corpus always keeps them; this is an index-side choice")

    s = sub.add_parser("search")
    s.add_argument("--model", required=True)
    s.add_argument("--query", required=True)
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--as-of", default=None, help="enforce point-in-time: filters rows AND checks the "
                                                "model's cutoff, raising rather than warning")
    s.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "fts"])
    s.add_argument("--where", default=None, help="extra SQL predicate, e.g. \"section = 'qa'\"")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        if args.shards is not None and not args.shards:
            raise SystemExit("--shards was given with no values. Omit it to index every shard.")
        res = build(args.model, args.shards, fts=not args.no_fts, ann=not args.no_ann,
                    overwrite=args.overwrite, min_chars=args.min_chars)
        print(f"{res['model_id']}: {res['n_rows']:,} rows over {res['shards']} shards in {res['dir']}")
        return 0

    df = search(args.query, args.model, args.k, args.as_of, args.mode, args.where)
    cols = [c for c in ("call_date", "companyid", "section", "speaker_name", "_relevance_score",
                        "_distance", "_score") if c in df.columns]
    for row in df.iter_rows(named=True):
        head = "  ".join(f"{c}={row[c]}" for c in cols)
        print(f"- {head}\n  {row[TEXT_COLUMN][:300]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
