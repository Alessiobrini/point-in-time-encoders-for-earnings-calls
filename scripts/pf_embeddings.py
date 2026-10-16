"""Call embeddings for the portfolio test (D26): one vintage embeds every in-index call up to a date.

For test year t, the vintage with cutoff t-1 embeds every call the regression trains or predicts on,
which is every in-index call from 2011 through the end of year t (D11). So the task for vintage v
embeds calls through December 31 of v+1. The 2024 vintage's task covers every call, and it is also
what the leaky variant reads, so the look-ahead needs no separate embedding job. That the leaky
assignment is made at all is the backtest's business and is flagged there.

A call is the average of its chunk vectors, L2 normalized, as in the premise test.

On the full universe (D28) one model's task can run most of a day, which the scavenger queue rarely
fits. `--parts N --part K` embeds only the quarterly shards dealt to part K, and `--merge N` joins the
N parts into the model's file once each part and the union cover every call. Pooling is per shard, so
the parts give the same vectors as one pass.

    PYTHONPATH=src python3 scripts/pf_embeddings.py --model ecalls-ret-20171231 --through 2018-12-31
    PYTHONPATH=src python3 scripts/pf_embeddings.py --model ecalls-ret-20171231 --through 2018-12-31 \
        --universe all --parts 4 --part 0
    PYTHONPATH=src python3 scripts/pf_embeddings.py --model ecalls-ret-20171231 --through 2018-12-31 \
        --universe all --merge 4
"""
import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path

import polars as pl

from ecalls.chunk import registry as R
from ecalls.config import paths as P
from ecalls.ingest.manifest import git_sha, utc_now

OUT_DIR = P.PROCESSED_DIR / "pf_embeddings"
OUT_DIR_ALL = P.PROCESSED_DIR / "pf_embeddings_all"     # D28, the full linked universe
PARTS_DIR = OUT_DIR_ALL / "parts"
FIRST_CALL = dt.date(2011, 1, 1)


def universe(through: dt.date) -> pl.DataFrame:
    """In-index calls with a permno, from 2011 through `through`."""
    c = pl.read_parquet(P.PROCESSED_DIR / "calls.parquet",
                        columns=["keydevid", "call_date", "shard", "permno", "in_index_at_call"])
    c = c.with_columns(pl.col("call_date").cast(pl.Date))
    return c.filter(pl.col("in_index_at_call") & pl.col("permno").is_not_null()
                    & (pl.col("call_date") >= FIRST_CALL) & (pl.col("call_date") <= through))


def universe_all(through: dt.date) -> pl.DataFrame:
    """D28: the next-day panel's events, which passed the common-stock, price and window filters."""
    ev = pl.read_parquet(P.PROCESSED_DIR / "pf_panel_daily.parquet", columns=["keydevid"])
    c = pl.read_parquet(P.PROCESSED_DIR / "calls.parquet", columns=["keydevid", "call_date", "shard"])
    c = c.with_columns(pl.col("call_date").cast(pl.Date)).join(ev, on="keydevid", how="inner")
    return c.filter((pl.col("call_date") >= FIRST_CALL) & (pl.col("call_date") <= through))


def select_part(calls: pl.DataFrame, part: int, parts: int) -> pl.DataFrame:
    """The calls in the shards dealt to `part`: shards in calendar order, dealt round robin.

    Dealing rather than cutting into blocks keeps the parts near equal, since later quarters hold
    more calls.
    """
    if not 0 <= part < parts:
        raise ValueError(f"part {part} is outside 0 to {parts - 1}")
    shards = sorted(calls["shard"].unique().to_list())
    if len(shards) < parts:
        raise ValueError(f"{len(shards)} shards cannot fill {parts} parts")
    return calls.filter(pl.col("shard").is_in(shards[part::parts]))


def part_path(model: str, part: int, parts: int) -> Path:
    return PARTS_DIR / f"{model}.p{part}of{parts}.parquet"


def merge_parts(want: set, frames: list[pl.DataFrame], metas: list[dict], through: str) -> pl.DataFrame:
    """Join the parts, or raise if any part is short or the union is not exactly the calls wanted.

    The expectation comes from the universe, never from the parts: a part that skipped a shard would
    otherwise pass as complete.
    """
    for i, m in enumerate(metas):
        if m.get("through") != through:
            raise ValueError(f"part {i} covers calls through {m.get('through')}, not {through}")
        if m.get("n_missing") or m.get("n_calls_embedded") != m.get("n_calls_requested"):
            raise ValueError(f"part {i} embedded {m.get('n_calls_embedded')} of "
                             f"{m.get('n_calls_requested')} calls, {m.get('n_missing')} missing")
    df = pl.concat(frames, how="vertical")
    if df["keydevid"].n_unique() != df.height:
        raise ValueError(f"{df.height - df['keydevid'].n_unique()} calls appear in more than one part")
    got = set(df["keydevid"].to_list())
    if got != want:
        raise ValueError(f"the parts hold {len(got - want)} calls outside the universe and miss "
                         f"{len(want - got)} of its {len(want)}")
    return df.sort("keydevid")


def _write(df: pl.DataFrame, out: Path, meta: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def _covers(meta_path: Path, through: str, n: int) -> bool:
    """Resume on coverage, not existence: a file from an earlier, shorter universe is not this one."""
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return meta.get("through") == through and meta.get("n_calls_embedded") == n and not meta.get("n_missing")


def _call_embeddings():
    p = Path(__file__).parent / "call_embeddings.py"
    spec = importlib.util.spec_from_file_location("call_embeddings", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--through", required=True, help="last call date to embed, YYYY-MM-DD")
    ap.add_argument("--device", default=None)
    ap.add_argument("--universe", choices=("sp500", "all"), default="sp500")
    ap.add_argument("--parts", type=int, default=None, help="split the shards into this many parts")
    ap.add_argument("--part", type=int, default=None, help="embed this part, 0 to parts - 1")
    ap.add_argument("--merge", type=int, default=None, metavar="N", help="join N finished parts")
    args = ap.parse_args(argv)
    full = args.universe == "all"
    if (args.parts is None) != (args.part is None):
        raise SystemExit("--parts and --part go together")
    if args.merge is not None and args.parts is not None:
        raise SystemExit("--merge joins parts, it does not embed one")

    through = dt.date.fromisoformat(args.through)
    reg = R.load()
    entry = reg.by_name(args.model)
    if entry.kind != R.VINTAGE:
        raise SystemExit(f"{args.model} is {entry.kind}; the portfolio test needs a known cutoff")
    calls = (universe_all(through) if full else universe(through)).with_columns(model_id=pl.lit(args.model))
    out = (OUT_DIR_ALL if full else OUT_DIR) / f"{args.model}.parquet"
    base = {"model": args.model, "training_cutoff": entry.training_cutoff, "through": args.through,
            "pooling": entry.pooling}

    if _covers(out.with_suffix(".json"), args.through, calls.height):
        print(f"{out.name} already covers all {calls.height:,} calls through {args.through}")
        return 0

    if args.merge is not None:
        paths = [part_path(args.model, k, args.merge) for k in range(args.merge)]
        absent = [p.name for p in paths if not p.exists()]
        if absent:
            raise SystemExit(f"parts not written yet: {', '.join(absent)}")
        metas = [json.loads(p.with_suffix(".json").read_text()) for p in paths]
        df = merge_parts(set(calls["keydevid"].to_list()), [pl.read_parquet(p) for p in paths],
                         metas, args.through)
        _write(df, out, {**base, "n_calls_requested": calls.height, "n_calls_embedded": df.height,
                         "n_missing": 0, "dim": len(df["embedding"][0]), "merged_from": args.merge,
                         "git_sha": git_sha(), "built_at": utc_now()})
        print(f"{df.height:,} calls from {args.merge} parts -> {out}")
        return 0

    if args.parts is not None:
        calls = select_part(calls, args.part, args.parts)
        out = part_path(args.model, args.part, args.parts)
        if _covers(out.with_suffix(".json"), args.through, calls.height):
            print(f"{out.name} already covers its {calls.height:,} calls")
            return 0

    print(f"{args.model}: {calls.height:,} calls from {FIRST_CALL} through {through}", flush=True)
    rows, missing = _call_embeddings().encode_calls(calls, reg, args.device, sections=full)
    df = pl.DataFrame(rows)
    _write(df, out, {**base, "n_calls_requested": calls.height, "n_calls_embedded": df.height,
                     "n_missing": missing, "dim": len(rows[0]["embedding"]) if rows else 0,
                     "part": args.part, "parts": args.parts, "git_sha": git_sha(), "built_at": utc_now()})
    print(f"{df.height:,} of {calls.height:,} calls embedded, {missing} missing -> {out}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
