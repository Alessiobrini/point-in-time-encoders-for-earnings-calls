"""Pinned identifiers, rule versions, and tuning constants.

Every value measured against WRDS carries the date it was measured. A value that can drift is
computed at run time and asserted here, never hardcoded. The clearest example is the CRSP coverage
horizon, which is `max(mbrenddt)` at pull time and must never become a literal in this file.
"""
import datetime as dt

# --- WRDS object names ----------------------------------------------------------------------
SCHEMA_TRANSCRIPTS = "ciq_transcripts"
SCHEMA_CIQ_COMMON = "ciq_common"

TBL_DETAIL = f"{SCHEMA_TRANSCRIPTS}.wrds_transcript_detail"
TBL_COMPONENT = f"{SCHEMA_TRANSCRIPTS}.ciqtranscriptcomponent"
TBL_COMPONENT_TYPE = f"{SCHEMA_TRANSCRIPTS}.ciqtranscriptcomponenttype"
TBL_SPEAKER_TYPE = f"{SCHEMA_TRANSCRIPTS}.ciqtranscriptspeakertype"
TBL_PERSON = f"{SCHEMA_TRANSCRIPTS}.ciqtranscriptperson"
TBL_WRDS_PERSON = f"{SCHEMA_TRANSCRIPTS}.wrds_transcript_person"
TBL_GVKEY_LINK = f"{SCHEMA_CIQ_COMMON}.wrds_gvkey"

TBL_SP500_CRSP = "crsp.dsp500list_v2"          # point-in-time membership. See D1.
TBL_SP500_COMP = "comp.idxcst_his"             # CURRENT MEMBERS ONLY. Used only for D3.
TBL_CCM_LINK = "crsp.ccmxpf_lnkhist"
GVKEYX_SP500 = "000003"

# CCM link filters. LU and LC are the usual research pair. LS is added because without it Eaton
# Corporation plc (gvkey 004199, permno 11762) is dropped entirely: its only CCM rows are LS/C and
# LS/P, and it has been an S&P 500 member continuously since 1957-03-01 with 75 calls in the corpus.
# Measured 2026-09-26 over the 996 index permnos: LU/LC + P/C loses 9 permnos, adding LS loses 7,
# and every permno has some P/C link (the residual 7 have only LD or LN, which carry no usable
# accounting link). The residual loss is counted and written to disk by link_chain rather than
# passing silently, which is what let this go unnoticed.
CCM_LINKTYPES = ("LU", "LC", "LS")
CCM_LINKPRIMS = ("P", "C")

# --- event type selection -------------------------------------------------------------------
# Matching `keydeveventtypename` by string against a 1.8M-row table is slower and more fragile than
# pinning the integer. Stage 1 reads these from the live lookup and asserts this tuple still holds.
EARNINGS_EVENT_TYPE_NAME = "Earnings Calls"
EARNINGS_EVENT_TYPE_IDS: tuple[int, ...] = (48,)   # read from the live lookup 2026-09-18

# --- corpus bounds --------------------------------------------------------------------------
CORPUS_START = dt.date(2004, 1, 1)
CORPUS_END = dt.date(dt.date.today().year, 12, 31)

# --- rule and cleaning versions -------------------------------------------------------------
VERSION_RULE = "longest-total-chars-v1"   # D4
CLEANING_VERSION = "ciq-raw-v1"           # D6
CHUNKER_VERSION = "qa-pair-v2"            # v2 keeps operator turns over 1,000 chars

# --- pull tuning, from measurement ----------------------------------------------------------
# 250-id batches returned in 0.7s on a 2026-09-18 probe, and 1,000-id batches hit 25.7 MB/s and
# 625 transcripts/s on 4,000 transcripts. The SUSTAINED rate over the real 412,255-call pull was
# 342 transcripts/s, so treat the probe figure as a ceiling. 250 keeps a failed batch cheap to redo.
COMPONENT_BATCH = 250
# Stage 5a returns one row per transcript, so the result set is tiny even though the scan is not.
LENGTH_BATCH = 2000

# --- refresh (D9) ---------------------------------------------------------------------------
# Refresh is a watermark on transcriptcreationdate_utc, not a window on the call date.
# The margin absorbs WRDS load lag: the creation date is Capital IQ's clock, and the snapshot was
# measured 5 days behind it on 2026-09-18. 30 days is 6x that and costs ~7,500 rows to recheck.
WATERMARK_MARGIN_DAYS = 30

# --- suspect flags, not filters. Nothing is ever dropped on these. ---------------------------
SUSPECT_SHORT_CHARS = 5_000
SUSPECT_LONG_CHARS = 300_000

# --- measured facts, asserted by ecalls.ingest.verify_measurements ---------------------------
# (name, expected, tolerance). None tolerance means an exact, hard-fail check.
MEASURED_2026_09_18 = {
    "dsp500list_v2_rows": (2084, "grows_only"),
    "members_2005_06_30": (500, 5),
    "members_2015_06_30": (502, 5),
    "members_2024_06_30": (503, 5),
    "idxcst_his_sp500_rows_with_thru": (0, None),   # the survivorship trap. See D1.
    "ever_member_permnos_since_2004": (996, 10),
    "permnos_linked_to_gvkey": (987, 10),
    "permnos_linked_to_companyid": (987, 10),
    "distinct_companyids": (1025, 15),
    "sp500_earnings_keydevids": (53689, 500),
    "all_earnings_keydevids": (412255, 5000),
}
