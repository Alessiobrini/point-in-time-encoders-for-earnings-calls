"""Read-only WRDS connection helper.

Credentials are read from the wrds-pgdata line of ~/.pgpass. The `wrds` Python package asks for
interactive input, so the code connects with psycopg2 directly.
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
      - TCP keepalives, so a connection that dies while the client sleeps raises within about 2
        minutes instead of blocking forever in recv(), and the caller can reconnect.
      - statement_timeout, so a hung server-side query is cancelled.
      - connect retries with a linear backoff (5, 10, 15 s) for transient timeouts and WRDS
        throttling.
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
