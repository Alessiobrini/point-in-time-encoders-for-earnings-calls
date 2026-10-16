"""Stage 1: probe WRDS access and pin the identifiers the pipeline hardcodes.

A table appearing in `information_schema` is NOT proof that we can read it, so every probe runs a
real `select ... limit 1`. The stage also reads the earnings `keydeveventtypeid` out of the live
lookup, because matching that by string against a 1.8M-row table at query time is both slower and
more fragile than pinning an integer.

Run:
    python -m ecalls.ingest.check_wrds_access
"""
import json
import sys

from ecalls.config import constants as C
from ecalls.config import paths as P
from ecalls.ingest.wrds_conn import connect

PROBE_TABLES = [
    C.TBL_DETAIL, C.TBL_COMPONENT, C.TBL_COMPONENT_TYPE, C.TBL_SPEAKER_TYPE,
    C.TBL_PERSON, C.TBL_WRDS_PERSON, C.TBL_GVKEY_LINK,
    C.TBL_SP500_CRSP, C.TBL_SP500_COMP, C.TBL_CCM_LINK,
]


def probe(cur, qualified: str) -> str:
    """Return "OK" or "DENIED/ERR: <first line>" for a real one-row read."""
    schema, table = qualified.split(".", 1)
    try:
        cur.execute(f'select * from "{schema}"."{table}" limit 1')
        cur.fetchall()
        return "OK"
    except Exception as e:  # noqa: BLE001 -- the message is the finding
        return f"DENIED/ERR: {str(e).splitlines()[0][:160]}"


def earnings_event_type_ids(cur) -> list[int]:
    """The keydeveventtypeid values whose name is the earnings-call event type."""
    cur.execute(
        f"select distinct keydeveventtypeid, keydeveventtypename from {C.TBL_DETAIL} "
        "where keydeveventtypename = %s",
        (C.EARNINGS_EVENT_TYPE_NAME,),
    )
    return sorted(int(r[0]) for r in cur.fetchall())


def main() -> int:
    P.ensure_dirs()
    conn = connect()
    cur = conn.cursor()

    results = {}
    print("=== access probe ===", flush=True)
    for t in PROBE_TABLES:
        results[t] = probe(cur, t)
        print(f"  {t:44s} {results[t]}", flush=True)

    denied = [t for t, v in results.items() if v != "OK"]

    print("\n=== pinning earnings event type ids ===", flush=True)
    ids = earnings_event_type_ids(cur) if not denied else []
    print(f"  {C.EARNINGS_EVENT_TYPE_NAME!r} -> {ids}", flush=True)
    if C.EARNINGS_EVENT_TYPE_IDS and tuple(ids) != C.EARNINGS_EVENT_TYPE_IDS:
        print(f"  DRIFT: constants.EARNINGS_EVENT_TYPE_IDS = {C.EARNINGS_EVENT_TYPE_IDS}", flush=True)

    out = {"probes": results, "earnings_event_type_ids": ids, "denied": denied}
    P.ACCESS_PROBE.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {P.ACCESS_PROBE}", flush=True)

    if denied:
        print(f"FAIL: {len(denied)} table(s) unreadable", flush=True)
        return 1
    print("PASS: all probed tables readable", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
