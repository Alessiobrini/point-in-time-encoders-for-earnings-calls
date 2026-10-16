"""The next-day event panel of D28: one row per call in the full linked universe.

**The clock.** A call's start is converted to New York time. The position enters at the first close at
least one hour after the call starts, since a call lasts about an hour: the call date's close for a call
starting by 15:00 on a trading day, the next trading day's close otherwise. The last close before the
call is the reference: the call date's close for a call starting at or after 16:00, the previous
trading day's close otherwise.

**Targets.** The primary target is the stock's return on the trading day after entry minus the market's
(`mktrf + rf`). The 5-day target compounds the five days after entry, a missing day earning the market
as in D26. For calls starting before 08:30 or at or after 16:00, the open-to-close return of the first
session after the call is kept too, since that session opens after the call is public. It is raw,
because the factor file has no open-to-close market return.

**Baselines** use no day after entry. For the open-to-close target the reaction is replaced by the
overnight gap, from the reference close to the session's open, since the reaction contains the session: the market-adjusted return from the reference close to the entry
close, reversal over the 5 trading days ending at the reference close, momentum over days -252 to -22
before entry, log market cap and two-digit SIC at the reference close.

**Filters,** counted when they drop a call: no CRSP row near the call, not a US common stock on NYSE,
AMEX or Nasdaq at the reference close, a reference close below $5, or a window past the data.

    python -m ecalls.portfolio.panel_daily
"""
import datetime as dt

import numpy as np
import polars as pl

from ecalls.config import paths as P
from ecalls.portfolio import panel as PN

PANEL_ALL = P.PROCESSED_DIR / "pf_panel_daily.parquet"
CUTOFF_ENTER_SAME_DAY = dt.time(15, 0)
CLOSE = dt.time(16, 0)
PRE_OPEN = dt.time(8, 30)
MIN_PRICE = 5.0
EXCHANGES = ("N", "A", "Q")
REV_DAYS = 5
MOM = (-252, -22)


def clock(calendar: np.ndarray, local_date: dt.date, start: dt.time) -> tuple[int, int, int]:
    """Entry index, reference index, and the open-to-close session index (-1 when not defined).

    `calendar` holds trading dates as datetime64[D]. An index past the calendar's end is returned as
    is and caught by the caller.
    """
    d = np.datetime64(local_date, "D")
    left = int(np.searchsorted(calendar, d, side="left"))
    right = int(np.searchsorted(calendar, d, side="right"))
    on_day = right > left                     # the call date is itself a trading day
    entry = left if (start <= CUTOFF_ENTER_SAME_DAY and on_day) else right
    ref = (right - 1) if (start >= CLOSE and on_day) else (left - 1)
    if start < PRE_OPEN:
        session = left
    elif start >= CLOSE:
        session = right
    else:
        session = -1
    return entry, ref, session


def compound(r: np.ndarray) -> float:
    return float(np.prod(1.0 + r) - 1.0)


def event(R, m, cap, prc, opn, entry: int, ref: int, session: int) -> dict | None:
    """Targets and baselines for one call on one permno's calendar-aligned series."""
    T = len(m)
    if ref < 0 or entry + 5 >= T or entry + MOM[0] < 0:
        return None
    r1 = R[entry + 1]
    if np.isnan(r1):
        return None
    w = R[entry + 1: entry + 6]
    w = np.where(np.isnan(w), m[entry + 1: entry + 6], w)
    react = R[ref + 1: entry + 1]
    react = np.where(np.isnan(react), m[ref + 1: entry + 1], react)
    rev = R[ref - REV_DAYS + 1: ref + 1]
    mom = R[entry + MOM[0]: entry + MOM[1] + 1]
    mom = mom[~np.isnan(mom)]
    past_cap = cap[: ref + 1]
    okc = np.flatnonzero(~np.isnan(past_cap) & (past_cap > 0))
    o2c = gap = np.nan
    if 0 <= session < T and not np.isnan(opn[session]) and opn[session] > 0 and not np.isnan(prc[session]):
        o2c = abs(prc[session]) / opn[session] - 1.0
        # The open-to-close target must be predicted from what is known at that open. The reaction
        # runs to the entry close and contains the session itself, so it would leak the target. The
        # overnight gap from the reference close to the session's open is the reaction known in time.
        if not np.isnan(prc[ref]) and prc[ref] != 0:
            gap = opn[session] / abs(prc[ref]) - 1.0
    return {"target_1d": float(r1 - m[entry + 1]),
            "target_5d": compound(w) - compound(m[entry + 1: entry + 6]),
            "target_o2c": float(o2c), "gap": float(gap),
            "reaction": compound(react) - compound(m[ref + 1: entry + 1]),
            "rev5": compound(rev[~np.isnan(rev)]) if np.any(~np.isnan(rev)) else np.nan,
            "mom": compound(mom) if len(mom) >= PN.MIN_MOM_DAYS else np.nan,
            "logcap": float(np.log(past_cap[okc[-1]])) if len(okc) else np.nan,
            "entry_idx": entry, "ref_idx": ref, "session_idx": session}


def is_common(row: dict) -> bool:
    return (row.get("sharetype") == "NS" and row.get("securitytype") == "EQTY"
            and row.get("securitysubtype") == "COM" and row.get("primaryexch") in EXCHANGES)


def build(calls: pl.DataFrame, crsp: pl.DataFrame, ff: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """`calls`: keydevid, permno, call_date, call_time_utc. One permno at a time, to keep memory small."""
    ff = ff.sort("date")
    calendar = ff["date"].to_numpy().astype("datetime64[D]")
    m = (ff["mktrf"] + ff["rf"]).to_numpy()
    pos = {d: i for i, d in enumerate(calendar)}
    T = len(calendar)
    drops = {"no_crsp": 0, "window_past_data": 0, "not_common_stock": 0, "price_below_5": 0,
             "no_return_after_entry": 0}
    rows = []
    by_permno = {k[0]: g for k, g in crsp.group_by(["permno"])}
    for (permno,), cg in calls.group_by(["permno"]):
        g = by_permno.get(permno)
        if g is None:
            drops["no_crsp"] += cg.height
            continue
        idx = np.array([pos.get(d, -1) for d in g["dlycaldt"].to_numpy().astype("datetime64[D]")])
        keep = idx >= 0
        series = {}
        for col in ("dlyret", "dlycap", "dlyprc", "dlyopen"):
            a = np.full(T, np.nan)
            a[idx[keep]] = g[col].to_numpy()[keep]
            series[col] = a
        flags = {int(i): r for i, r in zip(idx[keep], g.select(
            "sharetype", "securitytype", "securitysubtype", "primaryexch", "siccd").iter_rows(named=True))}
        for c in cg.iter_rows(named=True):
            local, _ = PN.local_event_date(c["call_date"], c["call_time_utc"])
            ts = dt.datetime.combine(c["call_date"], c["call_time_utc"], tzinfo=dt.timezone.utc).astimezone(PN.NY)
            entry, ref, session = clock(calendar, local, ts.time())
            if entry + 5 >= T or ref < 0:
                drops["window_past_data"] += 1
                continue
            f = flags.get(ref)
            if f is None:
                drops["no_crsp"] += 1
                continue
            if not is_common(f):
                drops["not_common_stock"] += 1
                continue
            px = series["dlyprc"][ref]
            if np.isnan(px) or abs(px) < MIN_PRICE:
                drops["price_below_5"] += 1
                continue
            ev = event(series["dlyret"], m, series["dlycap"], series["dlyprc"], series["dlyopen"],
                       entry, ref, session)
            if ev is None:
                drops["no_return_after_entry"] += 1
                continue
            rows.append({"keydevid": c["keydevid"], "permno": int(permno), "call_date": c["call_date"],
                         "start_ny": ts.time(), "entry_date": calendar[entry].astype(dt.date),
                         "known_1d": calendar[entry + 1].astype(dt.date),
                         "known_5d": calendar[entry + 5].astype(dt.date),
                         "sic2": int(f["siccd"] // 100) if f.get("siccd") is not None else -1, **ev})
    df = pl.DataFrame(rows)
    return df, {"calls_in": calls.height, "events": df.height, "dropped": drops}


def main() -> int:
    import json

    from ecalls.ingest import manifest as M
    uni = pl.read_parquet(P.PROCESSED_DIR / "pf_universe_all.parquet", columns=["keydevid", "permno", "call_date"])
    times = pl.read_parquet(P.CALLS, columns=["keydevid", "call_time_utc"])
    calls = uni.join(times, on="keydevid", how="inner")
    crsp = pl.read_parquet(P.RAW_DIR / "crsp_all" / "dsf_v2_*.parquet")
    ff = pl.read_parquet(P.RAW_DIR / "ff" / "ff_daily.parquet")
    df, stats = build(calls, crsp, ff)
    M.atomic_write_table(df.to_arrow(), PANEL_ALL)
    M.atomic_write_json({**stats, "built_at": M.utc_now(), "git_sha": M.git_sha()}, PANEL_ALL.with_suffix(".json"))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
