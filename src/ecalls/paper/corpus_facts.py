"""Table 1's counts: calls, held-out calls, chunks, pretraining tokens and question-answer pairs by window.

    python -m ecalls.paper.corpus_facts      # on the cluster, where the packed corpus is

Calls and chunks come from the chunk layer, counting calls dated on or before each cutoff and splitting
them by the shared held-out rule. Tokens are the packed window's sequences times the context, read with
`pretrain_data.sequences_up_to`, the function that sizes every vintage. Pairs are the task stage's own
count from each from-scratch retriever's run record.
"""
import json

import duckdb

from ecalls.config import paths as P
from ecalls.ingest.manifest import git_sha, utc_now
from ecalls.train import holdout as HO
from ecalls.train import pretrain_data as PD

CUTOFFS = tuple(range(2010, 2025))   # every cutoff with a vintage
WINDOW = "ecalls-bpe-32768-20101231-ctx768-20241231"
OUT = P.REPORTS_DIR / "paper" / "corpus_facts.json"


def main() -> int:
    glob = str(P.CHUNKS_DIR / "chunks_*.parquet")
    calls = duckdb.sql(f"select keydevid, min(call_date) as d, count(*) as n from read_parquet('{glob}') "
                       "group by keydevid").pl()
    held = calls["keydevid"].map_elements(lambda k: HO.is_holdout(int(k)), return_dtype=bool)
    calls = calls.with_columns(held=held)
    meta = json.loads((PD.window_dir(WINDOW) / "_meta.json").read_text())
    out = {}
    for y in CUTOFFS:
        c = f"{y}-12-31"
        w = calls.filter(calls["d"] <= __import__("datetime").date(y, 12, 31))
        run = json.loads((P.run_records_dir() / f"ecalls-ret-{y}1231" / "_run.json").read_text())
        out[str(y)] = {"calls": w.height, "held_out_calls": int(w["held"].sum()), "chunks": int(w["n"].sum()),
                       "tokens": int(PD.sequences_up_to(meta, c) * meta["context"]),
                       "pairs": int(run["n_pairs"]), "pair_calls": int(run["n_calls"])}
    out_all = {"total_calls": calls.height, "first_call": str(calls["d"].min()), "last_call": str(calls["d"].max())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"by_cutoff": out, **out_all, "git_sha": git_sha(), "ts": utc_now()}, indent=2) + "\n")
    for y, v in out.items():
        print(y, v)
    print(out_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
