"""Stage 2: point-in-time S&P 500 membership as interval spells.

Membership is stored as spells, one row per (permno, contiguous window), never as a daily panel.
Containment answers "was firm X in the index on date D", `distinct permno` answers "every firm ever
in the index", and a firm removed in 2011 and re-added in 2019 stays two spells with a real
non-member gap between them.

The source is CRSP `dsp500list_v2`. Compustat `idxcst_his` is NOT usable for history: with
gvkeyx='000003' it holds only current members, so it reports 216 firms on 2005-06-30 where CRSP
reports 500. See docs/decisions.md, D1.

CRSP stops at a coverage horizon, `max(mbrenddt)`, measured 2025-12-31 on 2026-09-18, past which it
returns zero members silently. That horizon is computed at pull time, never hardcoded, and carried
on every row. See docs/decisions.md, D3.

Run:
    python -m ecalls.ingest.universe_sp500
"""
import datetime as dt
import sys

import pandas as pd

from ecalls.config import constants as C
from ecalls.config import paths as P
from ecalls.ingest.manifest import atomic_write_parquet, append_manifest, git_sha, utc_now


# --- pure helpers, unit-tested with no database -----------------------------------------------

def coverage_horizon(memb: pd.DataFrame) -> dt.date:
    """The date past which the source knows nothing. Raises if the frame disagrees with itself."""
    vals = set(memb["crsp_cut_date"].dropna().unique())
    if len(vals) != 1:
        raise ValueError(f"crsp_cut_date is not unique across rows: {sorted(vals)}")
    return pd.Timestamp(vals.pop()).date()


def members_on(memb: pd.DataFrame, date: dt.date) -> set[int]:
    """Permnos whose spell contains `date`, inclusive at both ends.

    Raises past the coverage horizon when carry-forward has not been applied. A silently empty
    universe for a 2026 date is the exact failure this function exists to prevent.
    """
    date = pd.Timestamp(date).date()
    # The frame knows nothing past its own furthest end date, so asking beyond it must raise rather
    # than return an empty set. Checking the coverage horizon alone was not enough: carry-forward
    # pushes mbr_end out to CORPUS_END, which silenced the guard, and CORPUS_END is itself baked from
    # the wall-clock year at build time. On an artifact built in 2026 this returned 503 members for
    # 2026-12-31 and zero for 2027-01-01, with no error, which is the exact failure this prevents.
    furthest = pd.Timestamp(pd.to_datetime(memb["mbr_end"]).max()).date()
    if date > furthest:
        raise ValueError(
            f"{date} is past the furthest date this membership frame covers ({furthest}). "
            f"Its coverage horizon is {coverage_horizon(memb)} and carry-forward was applied to "
            f"{int(memb['carryforward'].sum())} spells. Rebuild the artifact rather than trusting "
            "an empty answer."
        )
    start = pd.to_datetime(memb["mbr_start"]).dt.date
    end = pd.to_datetime(memb["mbr_end"]).dt.date
    hit = (start <= date) & (end >= date)
    return set(memb.loc[hit, "permno"].astype(int))


def ever_members(memb: pd.DataFrame) -> set[int]:
    return set(memb["permno"].dropna().astype(int))


def is_member(memb: pd.DataFrame, permno: int, date: dt.date) -> bool:
    return int(permno) in members_on(memb, date)


def member_windows(memb: pd.DataFrame, permno: int) -> list[tuple[dt.date, dt.date]]:
    rows = memb.loc[memb["permno"].astype("Int64") == int(permno)].sort_values("mbr_start")
    return [(pd.Timestamp(a).date(), pd.Timestamp(b).date())
            for a, b in zip(rows["mbr_start"], rows["mbr_end"])]


def build_spells(rows, corpus_end: dt.date, cut: dt.date | None = None,
                 min_open: int = 2) -> pd.DataFrame:
    """Assemble spells from raw dsp500list_v2 rows and apply the carry-forward rule.

    `rows` is an iterable of (permno, indno, mbrstartdt, mbrenddt, mbrflg, indfam).

    The coverage horizon is the date where the source stops knowing anything, which shows up as
    many spells ending on the same day at once. Measured 2026-09-18: 503 rows share 2025-12-31.
    A single spell ending on the latest date is something else entirely, the last removal, and
    carrying it forward would leave a removed firm in the index indefinitely. So the maximum end
    date is only treated as a horizon when at least `min_open` spells share it, and otherwise no
    spell is open and nothing is carried forward. Pass `cut` to override.
    """
    df = pd.DataFrame(
        list(rows), columns=["permno", "indno", "mbr_start", "mbr_end_raw", "mbr_flg", "indfam"]
    )
    if df.empty:
        raise ValueError("dsp500list_v2 returned no rows")
    df["permno"] = df["permno"].astype("int32")
    df["indno"] = df["indno"].astype("int32")
    for c in ("mbr_start", "mbr_end_raw"):
        df[c] = pd.to_datetime(df[c]).dt.date

    max_end = max(df["mbr_end_raw"])                   # computed, never hardcoded
    if cut is None:
        n_at_max = int((df["mbr_end_raw"] == max_end).sum())
        # The horizon is the last date the source speaks to either way. What the shared-max test
        # decides is whether any spell continues PAST it, which is what carry-forward acts on.
        cut = max_end
        df["is_open"] = (df["mbr_end_raw"] == max_end) & (n_at_max >= min_open)
    else:
        df["is_open"] = df["mbr_end_raw"] == cut
    df["mbr_end"] = df.apply(
        lambda r: corpus_end if r["is_open"] else r["mbr_end_raw"], axis=1
    )
    df["carryforward"] = df["mbr_end"] != df["mbr_end_raw"]
    df["crsp_cut_date"] = cut
    df["source"] = C.TBL_SP500_CRSP
    return df


# --- the stage --------------------------------------------------------------------------------

def fetch(cur) -> pd.DataFrame:
    cur.execute(
        f"select permno, indno, mbrstartdt, mbrenddt, mbrflg, indfam from {C.TBL_SP500_CRSP}"
    )
    return build_spells(cur.fetchall(), C.CORPUS_END)


def main() -> int:
    from ecalls.ingest.wrds_conn import connect

    P.ensure_dirs()
    if P.SP500_MEMBERSHIP.exists():
        print(f"skip(exists): {P.SP500_MEMBERSHIP}", flush=True)
        return 0

    conn = connect()
    df = fetch(conn.cursor())
    df["pull_ts"] = utc_now()
    df["git_sha"] = git_sha()

    cut = coverage_horizon(df)
    print(f"spells={len(df):,}  permnos={df.permno.nunique():,}  "
          f"open={int(df.is_open.sum())}  horizon={cut}", flush=True)
    for d in (dt.date(2005, 6, 30), dt.date(2015, 6, 30), dt.date(2024, 6, 30)):
        print(f"  members on {d}: {len(members_on(df, d))}", flush=True)
    print(f"  ever-members since {C.CORPUS_START}: "
          f"{len({p for p in df.loc[df.mbr_end >= C.CORPUS_START, 'permno']}):,}", flush=True)

    atomic_write_parquet(df, P.SP500_MEMBERSHIP)
    append_manifest({"stage": "universe", "artifact": P.SP500_MEMBERSHIP.name,
                     "n_rows": len(df), "n_permnos": int(df.permno.nunique()),
                     "crsp_cut_date": cut, "pull_ts": utc_now(), "git_sha": git_sha()})
    print(f"wrote {P.SP500_MEMBERSHIP}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
