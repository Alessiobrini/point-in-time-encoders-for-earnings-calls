"""The lexical text baselines of the portfolio test (D26): dictionary tone, and TF-IDF reduced by SVD.

Both are computed from the chunk layer, the same text the embeddings see, joined in chunk order.

**Tone** is negative minus positive Loughran-McDonald words over all words. The master dictionary
records the year each word entered or left a category, so the tone of a call in test year t uses the
dictionary as it stood at the end of year t-1. A flag of y means the word joined that year, and -y
means it left that year after belonging from the start, per the dictionary's own documentation. That
removes the look-ahead of scoring 2013 text with words added in 2018.

**TF-IDF** is fitted on the training calls of one refit year only, reduced to 32 components by
truncated SVD fitted on the same rows, then applied to that year's training and test calls. It is the
lexical counterpart of the 32 PCA components the embeddings get, and like them it is refit every year.

Runs on the cluster, where the text is: one task per test year for TF-IDF, and one for tone.

    python -m ecalls.portfolio.text_features tone
    python -m ecalls.portfolio.text_features tfidf --year 2017
"""
import argparse
import datetime as dt
import re

import numpy as np
import polars as pl

from ecalls.config import paths as P
from ecalls.ingest import manifest as M

LM_PATH = P.DATA_ROOT / "external" / "lm" / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
FEAT_DIR = P.PROCESSED_DIR / "pf_features"
FEAT_DIR_ALL = P.PROCESSED_DIR / "pf_features_all"      # D28, the full linked universe
WORD = re.compile(r"[A-Za-z]+")
TFIDF_DIM = 32
FIRST_TEST_YEAR, LAST_TEST_YEAR = 2013, 2025


def in_category(flag: int, year: int) -> bool:
    """Whether a word with this dictionary flag belonged to the category at the end of `year`."""
    if flag > 0:
        return flag <= year
    if flag < 0:
        return year < -flag
    return False


def word_lists(year: int, lm: pl.DataFrame | None = None) -> tuple[set, set]:
    lm = lm if lm is not None else pl.read_csv(LM_PATH, infer_schema_length=0)
    neg = {w for w, f in zip(lm["Word"], lm["Negative"]) if in_category(int(f), year)}
    pos = {w for w, f in zip(lm["Word"], lm["Positive"]) if in_category(int(f), year)}
    return neg, pos


def tone(text: str, neg: set, pos: set) -> tuple[float, int]:
    words = [w.upper() for w in WORD.findall(text)]
    if not words:
        return float("nan"), 0
    n_neg = sum(w in neg for w in words)
    n_pos = sum(w in pos for w in words)
    return (n_neg - n_pos) / len(words), len(words)


def call_texts(keydevids: set) -> pl.DataFrame:
    """One row per call, its chunks joined in chunk order. Chunk ids sort in call order within a call."""
    frames = []
    for f in sorted(P.CHUNKS_DIR.glob("chunks_*.parquet")):
        frames.append(pl.read_parquet(f, columns=["keydevid", "chunk_id", "char_start", "text"])
                      .filter(pl.col("keydevid").is_in(keydevids)))
    ch = pl.concat(frames).sort(["keydevid", "char_start"])
    return ch.group_by("keydevid", maintain_order=True).agg(pl.col("text").str.join("\n"))


def training_and_test(panel: pl.DataFrame, year: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """D26's split: training calls are those whose day +61 falls before January 1 of `year`."""
    start = dt.date(year, 1, 1)
    train = panel.filter(pl.col("exit_date") < start)
    test = panel.filter(pl.col("day0").dt.year() == year)
    return train, test


def tone_by_year(text: str, lists: dict) -> dict:
    """Tone under every year's dictionary from one pass over the text.

    Only dictionary words are counted, since membership changes by year but the words do not, so
    each year's tone is a sum over a few hundred counts rather than a second pass over the call.
    """
    vocab = set().union(*[n | p for n, p in lists.values()])
    words = [w.upper() for w in WORD.findall(text)]
    if not words:
        return {y: float("nan") for y in lists}
    counts: dict[str, int] = {}
    for w in words:
        if w in vocab:
            counts[w] = counts.get(w, 0) + 1
    out = {}
    for y, (neg, pos) in lists.items():
        n_neg = sum(c for w, c in counts.items() if w in neg)
        n_pos = sum(c for w, c in counts.items() if w in pos)
        out[y] = (n_neg - n_pos) / len(words)
    return out


def run_tone(panel: pl.DataFrame) -> pl.DataFrame:
    """Tone of every call under the dictionary of each refit year, `tone_<t>` using year t-1's."""
    lm = pl.read_csv(LM_PATH, infer_schema_length=0)
    lists = {t: word_lists(t - 1, lm) for t in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1)}
    texts = call_texts(set(panel["keydevid"].to_list()))
    rows = []
    for kd, text in texts.iter_rows():
        rows.append({"keydevid": kd, **{f"tone_{t}": v for t, v in tone_by_year(text, lists).items()}})
    return pl.DataFrame(rows)


def run_tfidf(panel: pl.DataFrame, year: int) -> pl.DataFrame:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    train, test = training_and_test(panel, year)
    # Fitted on D26's training rows, applied to every call before the end of the test year. D27's
    # targets are known sooner than D26's 61-day return, so their training sets include calls from late
    # in the previous year that D26 excludes. Transforming those calls uses no information from after
    # the fit, since every one of them precedes January 1 of the test year.
    seen = set(train["keydevid"].to_list()) | set(test["keydevid"].to_list())
    extra = panel.filter((pl.col("day0") < dt.date(year, 1, 1)) & ~pl.col("keydevid").is_in(seen))
    keep = pl.concat([train.select("keydevid"), test.select("keydevid"), extra.select("keydevid")])
    texts = call_texts(set(keep["keydevid"].to_list()))
    tr = train.select("keydevid").join(texts, on="keydevid", how="inner")
    te = pl.concat([test.select("keydevid"), extra.select("keydevid")]).join(texts, on="keydevid", how="inner")
    vec = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95, max_features=50_000,
                          stop_words="english", dtype=np.float32)
    Xtr = vec.fit_transform(tr["text"].to_list())
    svd = TruncatedSVD(n_components=TFIDF_DIM, random_state=0).fit(Xtr)
    Ztr, Zte = svd.transform(Xtr), svd.transform(vec.transform(te["text"].to_list()))
    out = pl.concat([
        pl.DataFrame({"keydevid": tr["keydevid"], "split": "train", "tfidf": Ztr.astype(np.float32).tolist()}),
        pl.DataFrame({"keydevid": te["keydevid"], "split": "applied", "tfidf": Zte.astype(np.float32).tolist()}),
    ])
    print(f"{year}: {len(tr):,} training calls, {len(te):,} test calls, vocabulary "
          f"{len(vec.vocabulary_):,}, SVD variance {svd.explained_variance_ratio_.sum():.3f}", flush=True)
    return out


def load_panel(path=None) -> pl.DataFrame:
    """The event panel with the entry and exit dates the split needs. `path` reads the D26-style panel
    of the full universe (D28 amendment) in place of the S&P 500 one."""
    import pyarrow  # noqa: F401  -- polars reads parquet through it here
    from ecalls.portfolio import panel as PN

    df = pl.read_parquet(path or PN.PANEL_PATH)
    ff = pl.read_parquet(P.RAW_DIR / "ff" / "ff_daily.parquet").sort("date")
    cal = ff["date"]
    return df.with_columns(entry_date=pl.col("entry_idx").map_elements(lambda i: cal[i], return_dtype=pl.Date),
                           exit_date=pl.col("exit_idx").map_elements(lambda i: cal[i], return_dtype=pl.Date),
                           day0=pl.col("day0").cast(pl.Date))


def load_panel_daily() -> pl.DataFrame:
    """D28's next-day panel, cast to the columns the split reads: the event date is the entry close,
    and a call trains a refit once its next-day target is known."""
    from ecalls.portfolio import panel_daily as PDL

    df = pl.read_parquet(PDL.PANEL_ALL)
    return df.with_columns(day0=pl.col("entry_date").cast(pl.Date),
                           exit_date=pl.col("known_1d").cast(pl.Date))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("tone", "tfidf"))
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--panel", choices=("sp500", "daily"), default="sp500")
    args = ap.parse_args(argv)
    daily = args.panel == "daily"
    panel = load_panel_daily() if daily else load_panel()
    out_dir = FEAT_DIR_ALL if daily else FEAT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.what == "tone":
        df, path = run_tone(panel), out_dir / "tone.parquet"
    else:
        df, path = run_tfidf(panel, args.year), out_dir / f"tfidf_{args.year}.parquet"
    M.atomic_write_table(df.to_arrow(), path)
    print(f"wrote {path} ({df.height:,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
