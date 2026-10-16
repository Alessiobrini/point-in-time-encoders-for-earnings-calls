"""Turn the facts this pipeline was designed around into a regression harness.

Every number in `constants.MEASURED_2026_09_18` came from a live query on 2026-09-18 and a design
decision rests on it. WRDS changes under us, so these are re-measured rather than trusted, with
tolerances where drift is expected and exact equality where a change would invalidate a decision.

The queries here reproduce the ORIGINAL measurements, which means they deliberately do NOT apply the
date-window intersection the real link stage applies. The pipeline gets 985 permnos and 990
companyids because it intersects link validity with membership spells, and the pinned 987 and 1,025
are the un-intersected counts. Both are correct and they answer different questions, so this harness
checks WRDS rather than checking our code.

Run:
    python -m ecalls.ingest.verify_measurements
"""
import json
import sys

from ecalls.config import constants as C
from ecalls.config import paths as P
from ecalls.ingest import lookups as L
from ecalls.ingest.manifest import git_sha, utc_now

UNIV = f"""
with univ as (select distinct permno from {C.TBL_SP500_CRSP} where mbrenddt >= '2004-01-01'),
lnk as (select distinct u.permno, l.gvkey from univ u
        join {C.TBL_CCM_LINK} l on l.lpermno = u.permno
        where l.linktype = any(%(lt)s) and l.linkprim = any(%(lp)s)),
cid as (select distinct lnk.permno, lnk.gvkey, g.companyid
        from lnk join {C.TBL_GVKEY_LINK} g on g.gvkey = lnk.gvkey)
"""
PARAMS = {"lt": list(C.CCM_LINKTYPES), "lp": list(C.CCM_LINKPRIMS),
          "gx": C.GVKEYX_SP500, "ev": list(C.EARNINGS_EVENT_TYPE_IDS)}


def measure(cur) -> dict:
    out = {}

    def one(sql, params=None):
        cur.execute(sql, params if params is not None else PARAMS)
        return cur.fetchone()[0]

    out["dsp500list_v2_rows"] = one(f"select count(*) from {C.TBL_SP500_CRSP}")
    for d in ("2005-06-30", "2015-06-30", "2024-06-30"):
        out[f"members_{d.replace('-', '_')}"] = one(
            f"select count(distinct permno) from {C.TBL_SP500_CRSP} "
            f"where mbrstartdt <= '{d}' and mbrenddt >= '{d}'")
    out["idxcst_his_sp500_rows_with_thru"] = one(
        f"select count(*) from {C.TBL_SP500_COMP} where gvkeyx = %(gx)s and thru is not null")
    out["ever_member_permnos_since_2004"] = one(
        f"select count(distinct permno) from {C.TBL_SP500_CRSP} where mbrenddt >= '2004-01-01'")
    out["permnos_linked_to_gvkey"] = one(UNIV + "select count(distinct permno) from lnk")
    out["permnos_linked_to_companyid"] = one(UNIV + "select count(distinct permno) from cid")
    out["distinct_companyids"] = one(UNIV + "select count(distinct companyid) from cid")
    out["sp500_earnings_keydevids"] = one(UNIV + f"""
        select count(distinct d.keydevid) from {C.TBL_DETAIL} d
        join cid on cid.companyid = d.companyid
        where d.keydeveventtypeid = any(%(ev)s) and d.mostimportantdateutc >= '2004-01-01'""")
    out["all_earnings_keydevids"] = one(
        f"select count(distinct keydevid) from {C.TBL_DETAIL} "
        "where keydeveventtypeid = any(%(ev)s)")
    return out


def check_lookups(cur) -> list[str]:
    """The decode tables are pinned in lookups.py, so a change there would silently mislabel
    sections. Exact equality, no tolerance."""
    problems = []
    cur.execute(f"select transcriptcomponenttypeid, transcriptcomponenttypename "
                f"from {C.TBL_COMPONENT_TYPE}")
    live = {int(a): b for a, b in cur.fetchall()}
    if live != L.COMPONENT_TYPE:
        problems.append(f"component type table changed: live={live} pinned={L.COMPONENT_TYPE}")
    cur.execute(f"select speakertypeid, speakertypename from {C.TBL_SPEAKER_TYPE}")
    live = {int(a): b for a, b in cur.fetchall()}
    if live != L.SPEAKER_TYPE:
        problems.append(f"speaker type table changed: live={live} pinned={L.SPEAKER_TYPE}")
    cur.execute(f"select distinct keydeveventtypeid from {C.TBL_DETAIL} "
                "where keydeveventtypename = %s", (C.EARNINGS_EVENT_TYPE_NAME,))
    live_ids = tuple(sorted(int(r[0]) for r in cur.fetchall()))
    if live_ids != C.EARNINGS_EVENT_TYPE_IDS:
        problems.append(f"earnings event type ids changed: live={live_ids} "
                        f"pinned={C.EARNINGS_EVENT_TYPE_IDS}")
    return problems


def verdict(name: str, got: int, expected: int, tol) -> tuple[str, str]:
    if tol is None:
        return ("PASS", "exact") if got == expected else ("FAIL", "exact, hard failure")
    if tol == "grows_only":
        return ("PASS", "grows only") if got >= expected else ("FAIL", "shrank")
    return ("PASS", f"+/-{tol}") if abs(got - expected) <= tol else ("FAIL", f"outside +/-{tol}")


def main() -> int:
    from ecalls.ingest.wrds_conn import connect

    P.ensure_dirs()
    cur = connect(statement_timeout_ms=900_000).cursor()

    print("=== pinned measurements, re-measured ===")
    print(f"  {'check':38s} {'measured':>10s} {'expected':>10s} {'rule':>16s}  verdict")
    got = measure(cur)
    fails = []
    for name, (exp, tol) in C.MEASURED_2026_09_18.items():
        g = int(got[name])
        v, rule = verdict(name, g, exp, tol)
        if v == "FAIL":
            fails.append(f"{name}: measured {g:,}, expected {exp:,} ({rule})")
        print(f"  {name:38s} {g:>10,} {exp:>10,} {rule:>16s}  {v}")

    print("\n=== pinned decode tables ===")
    lk = check_lookups(cur)
    if lk:
        fails.extend(lk)
        for p_ in lk:
            print(f"  FAIL {p_[:150]}")
    else:
        print("  component types, speaker types, earnings event ids: all unchanged  PASS")

    P.MEASUREMENT_CHECK.write_text(json.dumps(
        {"measured": {k: int(v) for k, v in got.items()},
         "expected": {k: v[0] for k, v in C.MEASURED_2026_09_18.items()},
         "failures": fails, "checked_at": utc_now(), "git_sha": git_sha()}, indent=2))
    print(f"\nwrote {P.MEASUREMENT_CHECK}")

    if fails:
        print(f"\nFAIL: {len(fails)} check(s) out of tolerance", flush=True)
        return 1
    print("\nPASS: every pinned fact holds", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
