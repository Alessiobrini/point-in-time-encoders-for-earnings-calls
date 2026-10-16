"""Shared read-only WRDS connection helper.

Copied VERBATIM from tsfmvol/src/tsfmvol/ingest/wrds_conn.py on 2026-09-18 so the two stay
diffable. The per-day timings quoted below are tsfmvol's TAQ pull, not ours. Do not edit the
body here. Fix it upstream and re-copy.

Credentials are read from ~/.pgpass (the wrds-pgdata line). The `wrds` Python
package requires interactive input, so we use psycopg2 directly (see global setup).
"""
import os

HOST = "wrds-pgdata.wharton.upenn.edu"
PORT = 9737
DB = "wrds"


def _creds():
    path = os.path.expanduser("~/.pgpass")
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                parts = line.split(":")
                if len(parts) >= 5 and HOST in parts[0]:
                    return parts[3], ":".join(parts[4:])  # user, password
    raise RuntimeError(f"no '{HOST}' line found in ~/.pgpass")


def connect(readonly: bool = True, retries: int = 4, statement_timeout_ms: int = 300_000):
    """Open a (by default read-only, autocommit) psycopg2 connection to WRDS.

    Self-healing for long unattended pulls:
      - TCP keepalives so a connection killed while the Mac sleeps (the #1 cause of
        the corpus build "stalling" — recv() blocking on a dead socket forever) is
        detected within ~2 min and raises, so the caller's per-day reconnect recovers
        instead of hanging.
      - statement_timeout so a hung/runaway server-side query is cancelled (raises)
        rather than blocking; 300s >> a normal ~40-60s per-day pull.
      - connect retry with exponential backoff for transient connect timeouts / WRDS
        throttling (observed when many connections are opened in a session).
    """
    import time
    import psycopg2

    user, pw = _creds()
    last = None
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(
                host=HOST, port=PORT, dbname=DB, user=user, password=pw,
                sslmode="require", connect_timeout=30,
                keepalives=1, keepalives_idle=60, keepalives_interval=10, keepalives_count=5,
                options=f"-c statement_timeout={statement_timeout_ms}",
            )
            conn.set_session(readonly=readonly, autocommit=True)
            return conn
        except psycopg2.OperationalError as e:  # transient: timeout / throttle
            last = e
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))  # 5s, 10s, 15s
    raise last
