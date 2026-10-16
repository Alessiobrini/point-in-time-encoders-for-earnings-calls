"""The event panel for the portfolio test (D26): one row per call, with its target and controls.

**The event clock.** The call's UTC timestamp is converted to New York time. Day 0 is the first trading
day on or after that local date, or the next trading day when the call starts at or after the 16:00
close, since the market cannot react before the next open. Trading days are the Fama-French daily
file's dates.

**The target** is the buy-and-hold return over days +2 to +61 minus the buy-and-hold market return
(`mktrf + rf`) over the same days. A day with no stock return inside the window, which after a
delisting is every remaining day, takes the market's return, so it adds no abnormal return. The
delisting day itself carries the delisting return, which `dlyret` includes.

**Controls** use only days before day +2: the market-adjusted announcement return over days -1 to +1,
momentum over days -252 to -22, short-term reversal over days -21 to -2, log market cap on the last
day at or before -2 with one, and the two-digit SIC code.

A call is dropped, and counted, when its stock has no return on day -1 or day 0, which is how a call
whose permno was linked outside its listed life shows up, or when its window runs past the last
trading day in the data.
"""
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from ecalls.config import paths as P

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
CLOSE = dt.time(16, 0)
TARGET = (2, 61)
EAR = (-1, 1)
MOM = (-252, -22)
REV = (-21, -2)
MIN_MOM_DAYS = 200
# D27, abnormal volatility: log mean squared return over +2..+21 against -21..-2, and the
# heterogeneous autoregressive baselines of Corsi over a week, a month and a year before the call.
AV_POST, AV_PRE = (2, 21), (-21, -2)
RV_WEEK, RV_YEAR = (-6, -2), (-252, -22)
MIN_RV_DAYS = 15
RV_FLOOR = 1e-12
PANEL_PATH = P.PROCESSED_DIR / "pf_panel.parquet"
# D28 amendment of 2026-10-09: D26's target and baselines on the full linked universe's events.
PANEL_QUARTER = P.PROCESSED_DIR / "pf_panel_quarter.parquet"
CRSP_SP500 = P.RAW_DIR / "crsp" / "dsf_v2_*.parquet"
CRSP_ALL = P.RAW_DIR / "crsp_all" / "dsf_v2_*.parquet"


def local_event_date(call_date: dt.date, call_time_utc: dt.time) -> tuple[dt.date, bool]:
    """The call's New York date and whether it starts at or after the close."""
    ts = dt.datetime.combine(call_date, call_time_utc, tzinfo=UTC).astimezone(NY)
    return ts.date(), ts.time() >= CLOSE


def day0_index(calendar: np.ndarray, local_date: dt.date, after_close: bool) -> int:
    """Position in `calendar` (sorted trading dates as datetime64[D]) of day 0, or -1 past its end."""
    d = np.datetime64(local_date, "D")
    i = int(np.searchsorted(calendar, d, side="right" if after_close else "left"))
    return i if i < len(calendar) else -1


def compound(r: np.ndarray) -> float:
    return float(np.prod(1.0 + r) - 1.0)


def window(R: np.ndarray, m: np.ndarray, i0: int, a: int, b: int, fill_market: bool) -> tuple[np.ndarray, int]:
    """Stock returns on days i0+a .. i0+b, with missing days filled by the market or dropped."""
    seg = R[i0 + a: i0 + b + 1]
    miss = np.isnan(seg)
    if fill_market:
        seg = np.where(miss, m[i0 + a: i0 + b + 1], seg)
    else:
        seg = seg[~miss]
    return seg, int(miss.sum())


def log_rv(r: np.ndarray, min_days: int) -> float:
    """Log of the mean squared return over the non-missing days, NaN with fewer than `min_days`."""
    r = r[~np.isnan(r)]
    if len(r) < min_days:
        return float("nan")
    return float(np.log(max(np.mean(r ** 2), RV_FLOOR)))


def vol_features(R: np.ndarray, m: np.ndarray, i0: int) -> dict:
    """D27's abnormal volatility target and its baselines, every baseline from days before +2."""
    seg = lambda x, a, b: x[i0 + a: i0 + b + 1]
    post = log_rv(seg(R, *AV_POST), MIN_RV_DAYS)
    pre = log_rv(seg(R, *AV_PRE), MIN_RV_DAYS)
    return {"av_target": post - pre, "lrv_month": pre,
            "lrv_week": log_rv(seg(R, *RV_WEEK), 3),
            "lrv_year": log_rv(seg(R, *RV_YEAR), MIN_MOM_DAYS),
            "mkt_lrv_month": log_rv(seg(m, *AV_PRE), MIN_RV_DAYS),
            "av_known_idx": i0 + AV_POST[1]}


def event_row(R: np.ndarray, m: np.ndarray, cap: np.ndarray, i0: int) -> dict | None:
    """Target and controls for one call whose stock series is `R` and day 0 sits at `i0`."""
    T = len(m)
    if i0 < 0 or i0 + TARGET[1] >= T or i0 + MOM[0] < 0:
        return None
    if np.isnan(R[i0]) and np.isnan(R[i0 - 1]):
        return None
    stock, n_fill = window(R, m, i0, *TARGET, fill_market=True)
    target = compound(stock) - compound(m[i0 + TARGET[0]: i0 + TARGET[1] + 1])
    ear_s, _ = window(R, m, i0, *EAR, fill_market=True)
    ear = compound(ear_s) - compound(m[i0 + EAR[0]: i0 + EAR[1] + 1])
    mom_s, _ = window(R, m, i0, *MOM, fill_market=False)
    rev_s, _ = window(R, m, i0, *REV, fill_market=False)
    past_cap = cap[: i0 + REV[1] + 1]
    ok = np.flatnonzero(~np.isnan(past_cap))
    return {"target": target, "ear": ear,
            "mom": compound(mom_s) if len(mom_s) >= MIN_MOM_DAYS else np.nan,
            "rev": compound(rev_s) if len(rev_s) else np.nan,
            "logcap": float(np.log(past_cap[ok[-1]])) if len(ok) and past_cap[ok[-1]] > 0 else np.nan,
            "n_target_days_filled": n_fill,
            "entry_idx": i0 + TARGET[0], "exit_idx": i0 + TARGET[1],
            "abs_ear": abs(ear), **vol_features(R, m, i0)}


def load_returns(crsp_glob=None) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray, dict]:
    """Trading calendar, market return, and per-permno return, cap and SIC series on it."""
    ff = pl.read_parquet(P.RAW_DIR / "ff" / "ff_daily.parquet").sort("date")
    calendar = ff["date"].to_numpy().astype("datetime64[D]")
    m = (ff["mktrf"] + ff["rf"]).to_numpy()
    pos = {d: i for i, d in enumerate(calendar)}
    r = pl.read_parquet(crsp_glob or CRSP_SP500, columns=["permno", "dlycaldt", "dlyret", "dlycap", "siccd"])
    R, CAP, SIC = {}, {}, {}
    for (permno,), g in r.group_by(["permno"]):
        idx = np.array([pos.get(d, -1) for d in g["dlycaldt"].to_numpy().astype("datetime64[D]")])
        keep = idx >= 0
        ser = np.full(len(calendar), np.nan)
        ser[idx[keep]] = g["dlyret"].to_numpy()[keep]
        cser = np.full(len(calendar), np.nan)
        cser[idx[keep]] = g["dlycap"].to_numpy()[keep]
        sic = np.full(len(calendar), np.nan)
        sic[idx[keep]] = g["siccd"].cast(pl.Float64).to_numpy()[keep]
        R[int(permno)], CAP[int(permno)], SIC[int(permno)] = ser, cser, sic
    return calendar, m, R, CAP, SIC, pos


def build(calls: pl.DataFrame, quiet: bool = False, crsp_glob=None) -> tuple[pl.DataFrame, dict]:
    calendar, m, R, CAP, SIC, _ = load_returns(crsp_glob)
    rows = []
    dropped = {"no_returns_for_permno": 0, "window_past_data_end": 0, "no_trade_at_call": 0}
    for row in calls.iter_rows(named=True):
        p = int(row["permno"])
        if p not in R:
            dropped["no_returns_for_permno"] += 1
            continue
        local, after = local_event_date(row["call_date"], row["call_time_utc"])
        i0 = day0_index(calendar, local, after)
        if i0 < 0 or i0 + TARGET[1] >= len(m):
            dropped["window_past_data_end"] += 1
            continue
        ev = event_row(R[p], m, CAP[p], i0)
        if ev is None:
            dropped["no_trade_at_call"] += 1
            continue
        sic_past = SIC[p][: i0 + 1]
        ok = np.flatnonzero(~np.isnan(sic_past))
        rows.append({"keydevid": row["keydevid"], "permno": p, "call_date": row["call_date"],
                     "day0": calendar[i0].astype(dt.date), "year": int(str(calendar[i0])[:4]),
                     "sic2": int(sic_past[ok[-1]] // 100) if len(ok) else -1, **ev})
    df = pl.DataFrame(rows)
    stats = {"n_calls_in": calls.height, "n_events": df.height, "dropped": dropped,
             "share_target_days_filled": float(df["n_target_days_filled"].sum() / (df.height * 60))}
    if not quiet:
        print(stats)
    return df, stats


def reproduction(quarter: pl.DataFrame, sp500: pl.DataFrame) -> dict:
    """D28 amendment check: the quarter panel's S&P 500 calls must carry D26's target exactly.

    Both panels run `event_row` on the same CRSP returns, so any difference is a bug in the inputs or
    the clock, never noise.
    """
    j = quarter.select("keydevid", "day0", "target").join(
        sp500.select("keydevid", pl.col("day0").alias("day0_d26"), pl.col("target").alias("target_d26")),
        on="keydevid", how="inner")
    diff = (j["target"] - j["target_d26"]).abs()
    return {"n_common": j.height, "n_sp500": sp500.height,
            "max_abs_target_diff": float(diff.max()) if j.height else None,
            "n_day0_differs": int((j["day0"].cast(pl.Date) != j["day0_d26"].cast(pl.Date)).sum())}


def main(argv=None) -> int:
    import argparse
    import json

    from ecalls.ingest import manifest as M
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--universe", choices=("sp500", "all"), default="sp500")
    args = ap.parse_args(argv)
    if args.universe == "all":
        # The D28 events, already filtered to common stock above $5, with D26's clock and target.
        from ecalls.portfolio import panel_daily as PDL
        ev = pl.read_parquet(PDL.PANEL_ALL, columns=["keydevid", "permno"])
        times = pl.read_parquet(P.CALLS, columns=["keydevid", "call_date", "call_time_utc"])
        c = ev.join(times, on="keydevid", how="inner").with_columns(pl.col("call_date").cast(pl.Date))
        df, stats = build(c, crsp_glob=CRSP_ALL)
        check = reproduction(df, pl.read_parquet(PANEL_PATH))
        stats = {**stats, "reproduction_d26": check}
        print(json.dumps(check, indent=2))
        M.atomic_write_table(df.to_arrow(), PANEL_QUARTER)
        M.atomic_write_json({**stats, "git_sha": M.git_sha(), "built_at": M.utc_now()},
                            PANEL_QUARTER.with_suffix(".json"))
        if check["max_abs_target_diff"] is None or check["max_abs_target_diff"] > 1e-9 or check["n_day0_differs"]:
            raise SystemExit("the quarter panel does not reproduce D26 on the S&P 500 calls")
        print(f"wrote {PANEL_QUARTER}")
        return 0
    c = pl.read_parquet(P.CALLS, columns=["keydevid", "call_date", "call_time_utc", "permno",
                                          "in_index_at_call"])
    c = c.with_columns(pl.col("call_date").cast(pl.Date)).filter(
        pl.col("in_index_at_call") & pl.col("permno").is_not_null()
        & (pl.col("call_date") >= dt.date(2011, 1, 1)))
    df, stats = build(c)
    M.atomic_write_table(df.to_arrow(), PANEL_PATH)
    M.atomic_write_json({**stats, "git_sha": M.git_sha(), "built_at": M.utc_now()},
                        PANEL_PATH.with_suffix(".json"))
    print(f"wrote {PANEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
