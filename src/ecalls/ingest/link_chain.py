"""Stage 3: link point-in-time membership to Capital IQ company ids.

The chain is permno -> gvkey (CRSP/Compustat Merged) -> companyid (`ciq_common.wrds_gvkey`). Both
hops are date-ranged, so each link window is intersected with the membership spell rather than
merged onto it. A permno can legitimately resolve to more than one companyid (measured 987 permnos
to 1,025 companyids), so the fan-out is kept and flagged, never collapsed.

The Compustat reconciliation of D3 runs here: CRSP stops at its coverage horizon, so firms added to
the index after that date are visible only in Compustat's current snapshot.

Run:
    python -m ecalls.ingest.link_chain
"""
import datetime as dt
import sys

import pandas as pd

from ecalls.config import constants as C
from ecalls.config import paths as P
from ecalls.ingest.manifest import append_manifest, atomic_write_parquet, git_sha, utc_now

FAR_FUTURE = dt.date(2999, 12, 31)
FAR_PAST = dt.date(1900, 1, 1)


# --- pure helpers ------------------------------------------------------------------------------

def _as_date(s, fill=None):
    out = pd.to_datetime(s, errors="coerce").dt.date
    return out.map(lambda v: fill if pd.isna(v) else v) if fill is not None else out


def intersect_windows(left: pd.DataFrame, right: pd.DataFrame, key: str,
                      l_start="mbr_start", l_end="mbr_end",
                      r_start="link_start", r_end="link_end") -> pd.DataFrame:
    """Range join: keep pairs whose windows overlap, and carry the intersection.

    An open-ended window means the link is unbounded on that side, so a null end is filled with a
    far future date and a null start with a far past one. Both sides matter: 137,542 of the 144,806
    rows in `ciq_common.wrds_gvkey` carry a null `startdate`, so filling only the end silently
    dropped 90% of the universe (985 permnos down to 91) while looking like it worked.
    """
    if left.empty or right.empty:
        return left.iloc[0:0].assign(**{r_start: pd.Series(dtype="object"),
                                        r_end: pd.Series(dtype="object")})
    m = left.merge(right, on=key, how="inner")
    overlap = (m[l_start] <= m[r_end]) & (m[r_start] <= m[l_end])
    m = m.loc[overlap].copy()
    m["win_start"] = m[[l_start, r_start]].max(axis=1)
    m["win_end"] = m[[l_end, r_end]].min(axis=1)
    return m


def reconcile_compustat(universe: pd.DataFrame, comp_current: pd.DataFrame,
                        corpus_end: dt.date, gvkey_to_cid: dict | None = None) -> pd.DataFrame:
    """D3, both directions.

    Compustat-current minus CRSP-open are index additions CRSP has not recorded yet, and become new
    spells with a null permno. CRSP-open minus Compustat-current are firms removed after the CRSP
    cut, and they keep their end date with `possibly_removed` set. Truncating those on a guess
    would reintroduce survivorship bias, so the bias direction is over-inclusion plus a flag.
    """
    universe = universe.copy()
    open_gvkeys = set(universe.loc[universe["is_open"], "gvkey"].dropna())
    comp_gvkeys = set(comp_current["gvkey"].dropna())

    # A row is one (permno, gvkey, companyid, window) tuple, and its link can have ended decades
    # before the spell closes. Flagging on `is_open` alone asserted that Moody's may have been
    # dropped from the index after 2025 on two rows whose CCM link ended in 1998 and 2000. Require
    # the row's own window to still be live at the CRSP cut.
    cut = pd.to_datetime(universe["crsp_cut_date"]).max()
    win_live = pd.to_datetime(universe.get("win_end")) >= cut if "win_end" in universe else True
    universe["possibly_removed"] = (universe["is_open"].fillna(False)
                                    & win_live
                                    & ~universe["gvkey"].isin(comp_gvkeys))

    missing = comp_gvkeys - open_gvkeys
    add = comp_current.loc[comp_current["gvkey"].isin(missing)].copy()
    if not add.empty:
        add = add.rename(columns={"from_date": "mbr_start"})
        # Resolve the companyid for these gvkeys. Without it the spell is inert: index_label() in
        # consolidate.py matches calls through companyid, so a Compustat-only row with a null
        # companyid labels nothing and D3's whole correction does no work. All 15 such gvkeys do
        # have a companyid in ciq_common.wrds_gvkey, holding 724 earnings calls between them.
        add["companyid"] = (add["gvkey"].map(gvkey_to_cid) if gvkey_to_cid
                            else pd.Series(pd.NA, index=add.index))
        add["permno"] = pd.NA
        add["mbr_end"] = corpus_end
        add["mbr_end_raw"] = pd.NA
        add["is_open"] = True
        add["carryforward"] = True
        add["possibly_removed"] = False
        add["source"] = C.TBL_SP500_COMP
        add["crsp_cut_date"] = universe["crsp_cut_date"].iloc[0] if len(universe) else pd.NA
    return pd.concat([universe, add], ignore_index=True) if not add.empty else universe


# --- fetches -----------------------------------------------------------------------------------

def fetch_gvkey_links(cur, permnos) -> pd.DataFrame:
    cur.execute(
        f"select lpermno, gvkey, linkdt, linkenddt, linkprim, linktype from {C.TBL_CCM_LINK} "
        "where lpermno = any(%s) and linktype = any(%s) and linkprim = any(%s)",
        (list(map(int, permnos)), list(C.CCM_LINKTYPES), list(C.CCM_LINKPRIMS)),
    )
    df = pd.DataFrame(cur.fetchall(),
                      columns=["permno", "gvkey", "link_start", "link_end", "linkprim", "linktype"])
    df["permno"] = df["permno"].astype("int32")
    df["link_start"] = _as_date(df["link_start"], fill=FAR_PAST)
    df["link_end"] = _as_date(df["link_end"], fill=FAR_FUTURE)
    return df


def fetch_companyid_links(cur, gvkeys) -> pd.DataFrame:
    cur.execute(
        f"select gvkey, companyid, startdate, enddate, primaryflag from {C.TBL_GVKEY_LINK} "
        "where gvkey = any(%s)", (list(gvkeys),),
    )
    df = pd.DataFrame(cur.fetchall(),
                      columns=["gvkey", "companyid", "cid_start", "cid_end", "link_primaryflag"])
    df["companyid"] = df["companyid"].astype("int64")
    df["cid_start"] = _as_date(df["cid_start"], fill=FAR_PAST)
    df["cid_end"] = _as_date(df["cid_end"], fill=FAR_FUTURE)
    return df


def fetch_compustat_current(cur) -> pd.DataFrame:
    """Current S&P 500 members per Compustat. Only rows with a null `thru` are current."""
    cur.execute(
        f'select gvkey, "from" from {C.TBL_SP500_COMP} where gvkeyx = %s and thru is null',
        (C.GVKEYX_SP500,),
    )
    df = pd.DataFrame(cur.fetchall(), columns=["gvkey", "from_date"])
    df["from_date"] = _as_date(df["from_date"])
    return df


# --- the stage ---------------------------------------------------------------------------------

def main() -> int:
    from ecalls.ingest.wrds_conn import connect

    P.ensure_dirs()
    if P.SP500_UNIVERSE.exists():
        print(f"skip(exists): {P.SP500_UNIVERSE}", flush=True)
        return 0
    if not P.SP500_MEMBERSHIP.exists():
        print("FAIL: run ecalls.ingest.universe_sp500 first", flush=True)
        return 1

    memb = pd.read_parquet(P.SP500_MEMBERSHIP)
    memb = memb.loc[memb["mbr_end"] >= C.CORPUS_START].copy()
    print(f"membership spells in corpus window: {len(memb):,} "
          f"({memb.permno.nunique():,} permnos)", flush=True)

    conn = connect(); cur = conn.cursor()

    gl = fetch_gvkey_links(cur, memb["permno"].unique())
    step1 = intersect_windows(memb, gl, "permno")
    print(f"  permno -> gvkey:     {step1.permno.nunique():,} permnos, "
          f"{step1.gvkey.nunique():,} gvkeys", flush=True)
    # Record what the link filter drops. This loss used to pass in silence, which is how Eaton
    # Corporation plc went missing: 996 index permnos in, 985 out, and the stage just printed 985.
    lost = sorted(set(memb["permno"].astype("int64")) - set(step1["permno"].astype("int64")))
    if lost:
        print(f"  UNLINKED: {len(lost)} index permnos have no "
              f"{'/'.join(C.CCM_LINKTYPES)} + {'/'.join(C.CCM_LINKPRIMS)} CCM link: {lost}",
              flush=True)
        pd.DataFrame({"permno": lost, "reason": "no CCM link of an accepted type",
                      "linktypes_accepted": ",".join(C.CCM_LINKTYPES),
                      "pull_ts": utc_now(), "git_sha": git_sha()}).to_csv(
            P.UNIVERSE_DIR / "_unlinked_permnos.csv", index=False)

    cl = fetch_companyid_links(cur, step1["gvkey"].unique())
    step2 = intersect_windows(step1.rename(columns={"win_start": "mbr_start2",
                                                    "win_end": "mbr_end2"}),
                              cl, "gvkey", l_start="mbr_start2", l_end="mbr_end2",
                              r_start="cid_start", r_end="cid_end")
    print(f"  gvkey -> companyid:  {step2.permno.nunique():,} permnos, "
          f"{step2.companyid.nunique():,} companyids", flush=True)

    fan = step2.groupby(["permno", "mbr_start"], dropna=False)["companyid"].transform("nunique")
    step2["n_companyids"] = fan.astype("int16")
    step2["link_ambiguous"] = step2["n_companyids"] > 1

    comp = fetch_compustat_current(cur)
    print(f"  Compustat current members: {len(comp):,}", flush=True)
    # Resolve companyids for the Compustat-only gvkeys before adding their spells, so the D3
    # correction can actually label calls rather than sitting inert with a null companyid.
    comp_links = fetch_companyid_links(cur, comp["gvkey"].unique())
    comp_links = (comp_links.sort_values(["gvkey", "link_primaryflag", "companyid"])
                            .drop_duplicates("gvkey", keep="first"))
    g2c = dict(zip(comp_links["gvkey"], comp_links["companyid"]))
    uni = reconcile_compustat(step2, comp, C.CORPUS_END, gvkey_to_cid=g2c)
    added_linked = int(uni.loc[uni["source"] == C.TBL_SP500_COMP, "companyid"].notna().sum())

    added = int((uni["source"] == C.TBL_SP500_COMP).sum())
    removed = int(uni["possibly_removed"].fillna(False).sum())
    print(f"  D3 reconciliation: +{added} Compustat-only additions "
          f"({added_linked} with a resolved companyid), "
          f"{removed} flagged possibly_removed", flush=True)

    uni["pull_ts"] = utc_now(); uni["git_sha"] = git_sha()
    atomic_write_parquet(uni, P.SP500_UNIVERSE)
    append_manifest({"stage": "links", "artifact": P.SP500_UNIVERSE.name, "n_rows": len(uni),
                     "unlinked_permnos": len(lost),
                     "n_permnos": int(uni.permno.nunique()),
                     "n_companyids": int(uni.companyid.nunique()),
                     "compustat_additions": added, "possibly_removed": removed,
                     "pull_ts": utc_now(), "git_sha": git_sha()})
    print(f"wrote {P.SP500_UNIVERSE}  rows={len(uni):,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
