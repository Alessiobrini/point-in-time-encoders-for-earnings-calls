"""Stage 5b: choose one transcript per call. Pure, offline, deterministic.

Capital IQ publishes several versions of the same call (3.18 on average) and no collection type
consistently carries the most text, so the version is selected by measured length rather than by a
guessed hierarchy (D4). Over 91 measured events the within-event winner was Edited/Final 32 times,
EV/Final 28, Spellchecked/Preliminary 22, Proofed/Final 6 and Audited/Final 3, so a fixed ranking
of types would truncate roughly two thirds of calls.

This module touches no database. Changing the rule is a recomputation over the stage 5a shards and
the transcript index, and never a re-pull.

Run:
    python -m ecalls.ingest.select_versions
"""
import sys

import pandas as pd

from ecalls.config import constants as C
from ecalls.config import paths as P
from ecalls.ingest.manifest import append_manifest, atomic_write_parquet, git_sha, utc_now

STAGE = "select_versions"

# Only key 1 carries information. Keys 2 to 5 exist so the output cannot depend on the order the
# database happened to return rows in, and key 5 alone makes it total.
SORT_KEYS = ["total_chars", "n_components", "presentation_rank", "created_rank", "transcriptid"]
ASCENDING = [False, False, True, True, True]
TIEBREAK_NAME = {0: "chars", 1: "n_components", 2: "presentation", 3: "created", 4: "transcriptid"}


def _presentation_rank(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.strip().str.lower() != "final").astype("int8")


def choose_version(cands: pd.DataFrame) -> pd.DataFrame:
    """One row per keydevid, with the chosen transcript and the diagnostics to audit the rule.

    `cands` needs: keydevid, transcriptid, companyid, call_date, total_chars, n_components,
    presentation_type, collection_type, created_date.
    """
    df = cands.copy()
    df["presentation_rank"] = _presentation_rank(df["presentation_type"])
    df["created_rank"] = pd.to_datetime(df["created_date"], errors="coerce").rank(method="dense")
    df = df.sort_values(["keydevid"] + SORT_KEYS, ascending=[True] + ASCENDING,
                        kind="mergesort").reset_index(drop=True)

    g = df.groupby("keydevid", sort=True)
    best = g.head(1).set_index("keydevid")
    second = g.nth(1).set_index("keydevid") if len(df) > len(best) else best.iloc[0:0]

    out = pd.DataFrame({
        "keydevid": best.index,
        "companyid": best["companyid"].to_numpy(),
        "call_date": best["call_date"].to_numpy(),
        "chosen_transcriptid": best["transcriptid"].to_numpy(),
        "n_versions": g.size().reindex(best.index).to_numpy(),
        "chosen_chars": best["total_chars"].to_numpy(),
        "chosen_components": best["n_components"].to_numpy(),
        "chosen_collection_type": best["collection_type"].to_numpy(),
        "chosen_presentation_type": best["presentation_type"].to_numpy(),
    })
    out["runner_up_chars"] = second["total_chars"].reindex(out["keydevid"]).to_numpy()
    out["margin_ratio"] = (out["chosen_chars"] / out["runner_up_chars"].replace(0, pd.NA)
                           ).astype("Float32")
    out["tiebreak_used"] = _tiebreak(best, second, out["keydevid"])

    agg = g.agg(candidate_transcriptids=("transcriptid", list),
                candidate_chars=("total_chars", list),
                candidate_collection_types=("collection_type", list))
    for c in agg.columns:
        out[c] = agg[c].reindex(out["keydevid"]).to_numpy()

    out["suspect_short"] = out["chosen_chars"] < C.SUSPECT_SHORT_CHARS
    out["suspect_long"] = out["chosen_chars"] > C.SUSPECT_LONG_CHARS
    out["rule_version"] = C.VERSION_RULE
    return out.reset_index(drop=True)


def _tiebreak(best: pd.DataFrame, second: pd.DataFrame, keys) -> pd.Series:
    """Which sort key actually decided. If it reads 'chars' almost everywhere the tiebreaks are
    decoration, and if 'transcriptid' shows up often there is a real tie population to look at."""
    s2 = second.reindex(keys)
    b = best.reindex(keys)
    res = pd.Series("chars", index=range(len(keys)), dtype="object")
    undecided = s2["total_chars"].notna() & (b["total_chars"].to_numpy() == s2["total_chars"].to_numpy())
    for i, col in enumerate(SORT_KEYS[1:], start=1):
        still = undecided.to_numpy()
        if not still.any():
            break
        differs = b[col].to_numpy() != s2[col].to_numpy()
        res.loc[still & differs] = TIEBREAK_NAME[i]
        undecided = pd.Series(still & ~differs, index=undecided.index)
    return res.to_numpy()


def load_candidates(allow_missing: bool = False) -> pd.DataFrame:
    idx = pd.read_parquet(P.TRANSCRIPT_INDEX, columns=[
        "keydevid", "transcriptid", "companyid", "call_date",
        "collection_type", "presentation_type", "created_date"])
    shards = sorted(P.LENGTHS_DIR.glob("lengths_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no length shards in {P.LENGTHS_DIR}; run transcript_lengths")
    lens = pd.concat([pd.read_parquet(s, columns=["transcriptid", "n_components", "total_chars"])
                      for s in shards], ignore_index=True)
    lens = lens.drop_duplicates("transcriptid", keep="last")
    merged = idx.merge(lens, on="transcriptid", how="inner")

    # This inner join is where a gap in the length shards vanishes. Affected calls never reach
    # version_choice.parquet, so pull_components never pulls them, and completeness reports
    # UNEXPLAINED = 0 because it compares against version_choice itself. Silent end to end.
    # An earlier version of this function printed a note here; the note was deleted by accident when
    # the parent/subsidiary dedup below was added, so for a while there was no signal at all.
    missing = len(idx) - len(merged)
    if missing:
        if not allow_missing:
            raise RuntimeError(
                f"{missing:,} indexed transcripts have no length row and would be dropped here "
                f"without trace. Run `python -m ecalls.ingest.transcript_lengths` first, or pass "
                f"--allow-missing-lengths if the gap is intentional and recorded.")
        print(f"  WARNING: dropping {missing:,} transcripts with no length row, as permitted",
              flush=True)

    # The index grain is (transcriptid, companyid), so a call covering a parent and a subsidiary
    # carries the same transcript twice. Counting it twice would inflate n_versions and duplicate
    # the candidate lists, so candidates are collapsed to one row per (keydevid, transcriptid).
    # The company fan-out is preserved separately in call_companies.parquet.
    before = len(merged)
    merged = (merged.sort_values(["keydevid", "transcriptid", "companyid"])
                    .drop_duplicates(["keydevid", "transcriptid"], keep="first")
                    .reset_index(drop=True))
    if before != len(merged):
        print(f"  collapsed {before - len(merged):,} parent/subsidiary duplicate rows", flush=True)
    return merged


def write_call_companies() -> int:
    """Every (keydevid, companyid) route to a call, kept so the index label can use any of them."""
    idx = pd.read_parquet(P.TRANSCRIPT_INDEX, columns=["keydevid", "companyid", "company_name"])
    cc = (idx.drop_duplicates(["keydevid", "companyid"])
             .sort_values(["keydevid", "companyid"]).reset_index(drop=True))
    atomic_write_parquet(cc, P.CALL_COMPANIES)
    return len(cc)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing-lengths", action="store_true",
                    help="proceed even though some indexed transcripts have no length row")
    args = ap.parse_args(argv)

    P.ensure_dirs()
    n_cc = write_call_companies()
    print(f"call/company routes: {n_cc:,}", flush=True)
    cands = load_candidates(allow_missing=args.allow_missing_lengths)
    print(f"candidates: {len(cands):,} transcripts over {cands.keydevid.nunique():,} calls",
          flush=True)
    out = choose_version(cands)
    print(f"chosen: {len(out):,} calls, {out.chosen_chars.sum()/1e9:.2f} GB of text", flush=True)
    print("  tiebreak_used: " + ", ".join(
        f"{k}={v:,}" for k, v in out["tiebreak_used"].value_counts().items()), flush=True)
    print(f"  suspect_short={int(out.suspect_short.sum()):,}  "
          f"suspect_long={int(out.suspect_long.sum()):,}", flush=True)

    out["pull_ts"] = utc_now(); out["git_sha"] = git_sha()
    atomic_write_parquet(out, P.VERSION_CHOICE)
    append_manifest({"stage": STAGE, "artifact": P.VERSION_CHOICE.name, "n_rows": len(out),
                     "total_chars": int(out.chosen_chars.sum()),
                     "pull_ts": utc_now(), "git_sha": git_sha()})
    print(f"wrote {P.VERSION_CHOICE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
