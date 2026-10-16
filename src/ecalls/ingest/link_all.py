"""Link every call since 2011 to a CRSP permno valid on the call date (D28).

`link_chain.py` links in the other direction, from S&P 500 members to Capital IQ companies, so only
index members' calls carry a permno. The next-day return test needs every US-listed company, so this
runs companyid to gvkey through `ciq_common.wrds_gvkey`, then gvkey to permno through the
CRSP/Compustat link history, both date-ranged, and keeps a link only if the call date falls inside
both windows. A permno outside its link window is never used: 17 permnos in the S&P panel carried
calls from outside their listed life, which is what an undated link produces.

When a call resolves to more than one permno, the primary link (`linkprim` P before C, then the
Capital IQ `primaryflag`) wins and the call is flagged, so the fan-out is visible rather than collapsed
silently.

Runs on the Mac, where the WRDS credential lives:

    python -m ecalls.ingest.link_all
"""
import datetime as dt

import polars as pl

from ecalls.config import constants as C
from ecalls.config import paths as P
from ecalls.ingest import manifest as M

OUT = P.PROCESSED_DIR / "pf_universe_all.parquet"
FIRST_CALL = dt.date(2011, 1, 1)
LINKTYPES = ("LU", "LC")
LINKPRIMS = ("P", "C")
FAR_PAST, FAR_FUTURE = dt.date(1900, 1, 1), dt.date(2999, 12, 31)


def resolve(calls: pl.DataFrame, cid: pl.DataFrame, ccm: pl.DataFrame) -> pl.DataFrame:
    """One row per call with a permno whose two link windows both contain the call date.

    `calls`: keydevid, companyid, call_date. `cid`: companyid, gvkey, cid_start, cid_end,
    primaryflag. `ccm`: gvkey, permno, link_start, link_end, linkprim. Null window ends are open.
    """
    cid = cid.with_columns(pl.col("cid_start").fill_null(FAR_PAST), pl.col("cid_end").fill_null(FAR_FUTURE))
    ccm = ccm.with_columns(pl.col("link_start").fill_null(FAR_PAST), pl.col("link_end").fill_null(FAR_FUTURE))
    a = calls.join(cid, on="companyid", how="inner").filter(
        (pl.col("call_date") >= pl.col("cid_start")) & (pl.col("call_date") <= pl.col("cid_end")))
    b = a.join(ccm, on="gvkey", how="inner").filter(
        (pl.col("call_date") >= pl.col("link_start")) & (pl.col("call_date") <= pl.col("link_end")))
    b = b.with_columns(prim_rank=pl.when(pl.col("linkprim") == "P").then(0).otherwise(1),
                       cid_rank=pl.when(pl.col("primaryflag") == 1).then(0).otherwise(1))
    n = b.group_by("keydevid").agg(pl.col("permno").n_unique().alias("n_permnos"))
    best = (b.sort(["keydevid", "prim_rank", "cid_rank", "permno"])
              .unique("keydevid", keep="first", maintain_order=True))
    return (best.join(n, on="keydevid")
                .with_columns(link_ambiguous=pl.col("n_permnos") > 1)
                .select("keydevid", "companyid", "call_date", "gvkey", "permno", "linkprim",
                        "linktype", "n_permnos", "link_ambiguous"))


def fetch(companyids: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    from ecalls.ingest import wrds_conn as W
    conn = W.connect()
    cur = conn.cursor()
    cur.execute(f"SELECT companyid, gvkey, startdate, enddate, primaryflag FROM {C.TBL_GVKEY_LINK} "
                "WHERE companyid = ANY(%s)", (companyids,))
    cid = pl.DataFrame(cur.fetchall(), orient="row",
                       schema={"companyid": pl.Int64, "gvkey": pl.String, "cid_start": pl.Date,
                               "cid_end": pl.Date, "primaryflag": pl.Int32})
    gv = sorted(set(cid["gvkey"].drop_nulls().to_list()))
    cur.execute(f"SELECT gvkey, lpermno, linkdt, linkenddt, linkprim, linktype FROM {C.TBL_CCM_LINK} "
                "WHERE gvkey = ANY(%s) AND linktype = ANY(%s) AND linkprim = ANY(%s)",
                (gv, list(LINKTYPES), list(LINKPRIMS)))
    ccm = pl.DataFrame(cur.fetchall(), orient="row",
                       schema={"gvkey": pl.String, "permno": pl.Int32, "link_start": pl.Date,
                               "link_end": pl.Date, "linkprim": pl.String, "linktype": pl.String})
    return cid, ccm


def main() -> int:
    calls = pl.read_parquet(P.CALLS, columns=["keydevid", "companyid", "call_date"]).with_columns(
        pl.col("call_date").cast(pl.Date), pl.col("companyid").cast(pl.Int64)).filter(
        pl.col("call_date") >= FIRST_CALL)
    cid, ccm = fetch(sorted(set(calls["companyid"].drop_nulls().to_list())))
    out = resolve(calls, cid, ccm)
    M.atomic_write_table(out.to_arrow(), OUT)
    stats = {"calls_since_2011": calls.height, "companies": calls["companyid"].n_unique(),
             "companyid_to_gvkey_rows": cid.height, "gvkey_to_permno_rows": ccm.height,
             "calls_linked": out.height, "permnos": out["permno"].n_unique(),
             "ambiguous": int(out["link_ambiguous"].sum()), "built_at": M.utc_now(), "git_sha": M.git_sha()}
    M.atomic_write_json(stats, OUT.with_suffix(".json"))
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
