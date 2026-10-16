"""Stage 8: reconcile every call stage 5b expected against what actually landed.

Each expected call is KEPT, DROPPED with a logged reason, or UNEXPLAINED. **UNEXPLAINED is the real
gap and the target is zero.** That is the reproducibility guarantee: after this stage, no expected
call may be silently absent, and a call that is absent must be absent for a reason written down
somewhere.

Runs entirely offline from the artifacts on disk. No WRDS access.

Run:
    python -m ecalls.audit.completeness
"""
import sys

import pandas as pd

from ecalls.config import paths as P
from ecalls.ingest.manifest import append_manifest, git_sha, utc_now

STAGE = "completeness"


def failed_transcript_ids() -> set[int]:
    """Transcript ids from batches that exhausted their retries during the pull."""
    if not P.FAILED_BATCHES.exists():
        return set()
    out: set[int] = set()
    df = pd.read_csv(P.FAILED_BATCHES)
    for cell in df.get("transcriptids", pd.Series(dtype=str)).dropna().astype(str):
        out.update(int(x) for x in cell.split() if x.strip().isdigit())
    return out


def classify(expected: pd.DataFrame, present: set[int], failed_tx: set[int]) -> pd.DataFrame:
    """One row per expected call with its verdict.

    `expected` needs keydevid and chosen_transcriptid.
    """
    out = expected.copy()
    is_present = out["keydevid"].isin(present)
    is_failed = out["chosen_transcriptid"].isin(failed_tx)
    out["verdict"] = "UNEXPLAINED"
    out.loc[is_failed & ~is_present, "verdict"] = "DROPPED"
    out.loc[is_present, "verdict"] = "KEPT"
    out["reason"] = ""
    out.loc[out["verdict"] == "DROPPED", "reason"] = "pull batch failed after retries"
    return out


def main() -> int:
    if not P.CALLS.exists():
        print("FAIL: run ecalls.ingest.consolidate first", flush=True)
        return 1

    expected = pd.read_parquet(P.VERSION_CHOICE,
                               columns=["keydevid", "chosen_transcriptid", "call_date",
                                        "companyid", "suspect_short", "suspect_long"])
    calls = pd.read_parquet(P.CALLS, columns=["keydevid", "n_components", "n_chars",
                                             "in_index_at_call"])
    present = set(calls["keydevid"])
    failed_tx = failed_transcript_ids()

    audit = classify(expected, present, failed_tx)
    counts = audit["verdict"].value_counts()
    print("=== completeness ===")
    for v in ("KEPT", "DROPPED", "UNEXPLAINED"):
        print(f"  {v:12s} {int(counts.get(v, 0)):>9,}")
    n_unexpl = int(counts.get("UNEXPLAINED", 0))

    audit.to_csv(P.COMPLETENESS_AUDIT, index=False)
    print(f"\nwrote {P.COMPLETENESS_AUDIT}")

    print("\n=== flags, which are descriptive and never filters ===")
    print(f"  suspect_short calls      {int(expected.suspect_short.sum()):>9,}")
    print(f"  suspect_long calls       {int(expected.suspect_long.sum()):>9,}")
    print(f"  calls in index at call   {int(calls.in_index_at_call.sum()):>9,}")
    print(f"  components with no text  "
          f"{int((pd.read_parquet(P.CALLS, columns=['n_components']).n_components == 0).sum()):>9,}"
          "  (calls with zero components)")

    append_manifest({"stage": STAGE, "artifact": P.COMPLETENESS_AUDIT.name,
                     "n_rows": len(audit), "n_kept": int(counts.get("KEPT", 0)),
                     "n_dropped": int(counts.get("DROPPED", 0)), "n_unexplained": n_unexpl,
                     "pull_ts": utc_now(), "git_sha": git_sha()})

    if n_unexpl:
        print(f"\nFAIL: {n_unexpl:,} calls are UNEXPLAINED. The target is zero.", flush=True)
        ex = audit.loc[audit.verdict == "UNEXPLAINED"].head(5)
        print(ex[["keydevid", "chosen_transcriptid", "call_date"]].to_string(index=False))
        return 1
    print("\nPASS: UNEXPLAINED = 0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
