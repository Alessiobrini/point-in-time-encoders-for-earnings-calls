# Point-in-Time Encoders for Earnings Calls

This repository holds the code for the paper *Point-in-Time Encoders for Earnings Calls* (Alessio Brini, 2026).

The paper trains yearly vintages of text encoders on earnings calls, with cutoffs from 2010 to 2024. Each vintage reads only calls dated on or before its cutoff. It compares four pretraining recipes and fine-tunes every vintage contrastively on the calls' own question-answer pairs. It then measures how much a later vintage gains over the vintage available at the time in three settings:
- retrieval;
- a language model that answers from retrieved passages;
- forecasts of returns and fundamentals.

## What this repository contains, and what it does not

| Included | Not included |
|---|---|
| All code: ingestion, chunking, tokenizer, pretraining, contrastive fine-tuning, evaluation, forecasting, and the paper's tables and figures | Transcript text, or any file derived from it |
| The identifiers of every call (`identifiers/calls.csv.gz`) | Market data from CRSP and Compustat |
| The result files that every table and figure is built from (`reports/`), and the training records of every model (`runs/`) | Trained model weights |
| The model registry (`registry/embeddings.json`) and unit tests | |

The transcripts are licensed by S&P Global and cannot be redistributed. Anyone with the same WRDS subscriptions can rebuild the corpus exactly from the identifiers and the code. The weights are released subject to the data licensor's permission.

## Data you need, and how to get it

Every source is on WRDS (Wharton Research Data Services). You need an account with access to:

| WRDS library | Tables the code reads | Used for |
|---|---|---|
| S&P Capital IQ transcripts (`ciq_transcripts`) | `wrds_transcript_detail`, `ciqtranscriptcomponent`, `wrds_transcript_person`, `ciqtranscriptperson`, the component and speaker type tables | The calls themselves |
| Capital IQ common (`ciq_common`) | `wrds_gvkey` | Linking Capital IQ companies to Compustat |
| CRSP | `dsp500list_v2`, `dsf_v2`, `stocknames`, `ccmxpf_lnkhist` | Index membership, returns, links to Compustat |
| Compustat | `fundq` | Earnings per share and report dates |
| Fama-French | `factors_daily`, `fivefactors_daily` | The market return |

The code reads WRDS credentials from `~/.pgpass`, as the WRDS PostgreSQL server expects:

```
wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>
```

Set the environment variables below, for example by copying `.env.example` to `.env`, editing it and running `source .env`. All data files resolve under `ECALLS_DATA_ROOT`, so the corpus can live on an external disk or a cluster without code changes.

| Variable | Meaning | Default |
|---|---|---|
| `ECALLS_DATA_ROOT` | Root of every data artifact | `./data` |
| `ECALLS_ROOT` | Clone of this repository on the cluster, read by the SLURM scripts | none, required there |
| `ECALLS_PYTHON` | Python interpreter of the environment | `python` |
| `ECALLS_LLM_CACHE` | Download cache for the ChronoGPT-Instruct readers | `.llm-cache` |

The corpus takes about 16 GB of raw text. The full pull ran at about 340 transcripts per second.

### Things that are easy to get wrong in the source data

- **Compustat's index history is current-members-only.** `comp.idxcst_his` with `gvkeyx='000003'` holds today's S&P 500 members, so a 2005 query returns 216 firms instead of 500. Use `crsp.dsp500list_v2` for membership history.
- **CRSP membership ends silently.** `crsp.dsp500list_v2` stops at its last `mbrenddt` and returns no members after it. The code computes this horizon at pull time and stores it on every row.
- **A call has several transcript versions.** Capital IQ holds about 3.2 transcripts per call, and no collection type consistently carries the most text. The code keeps the version with the most text, measured on the server. Capital IQ also revises versions for weeks after a call, so the chosen version can change over time. `identifiers/calls.csv.gz` fixes the version the paper used.
- **Calls are dated in UTC.** The code converts call times to New York time for the forecasting event clock.

## Setup

```bash
conda env create -f environment.yml
conda activate pit-encoders
python -m pip install -e . --no-deps
python -m pytest                       # unit tests, no WRDS connection or data needed
```

The environment uses Python 3.11, PyTorch, transformers, polars and pyarrow. Cluster jobs used the same environment with PyTorch built for CUDA.

## Reproducing the paper

The pipeline runs in stages. Every stage writes its own output, skips work that already exists, and records a manifest row. The design decisions the code cites by number (D1, D29, ...) are summarized in `docs/decisions.md`.

### 1. Build the corpus (a machine with WRDS access)

```bash
python -m ecalls.ingest.check_wrds_access     # checks access to every table
scripts/run_all.sh --full                     # membership, links, transcript index, versions, components
python -m ecalls.chunk.build_chunks           # chunks of about 500 tokens, at most about 700
python scripts/export_identifiers.py          # optional: writes identifiers/calls.csv.gz
```

`scripts/run_all.sh --list` prints the stages. To rebuild the paper's corpus exactly, keep the transcript versions listed in `identifiers/calls.csv.gz`.

### 2. Tokenizer, packed corpus and question-answer pairs

```bash
python -m ecalls.train.tokenizer fit --cutoff 2010-12-31 --vocab-size 32768
python -m ecalls.train.pretrain_data --cutoff 2024-12-31   # one packed corpus, every vintage reads a prefix
python -m ecalls.train.pairs --cutoff 2024-12-31           # see cluster/build_pairs.sbatch
```

The held-out calls are 10% of all calls. A call is held out when the first four bytes of the SHA-1 hash of the string `ecalls-holdout-v1:<keydevid>` fall in the lowest tenth of their range (`src/ecalls/train/holdout.py`). Held-out calls enter neither pretraining nor fine-tuning, and every retrieval question comes from them. The one exception is the tokenizer, which is fitted on every call up to 2010. No test question comes from those years.

### 3. Train the vintages (SLURM cluster with GPUs)

The scripts in `cluster/` are written for SLURM. Edit the partition names to match your cluster, then submit from the repository root with `ECALLS_ROOT` and `ECALLS_PYTHON` exported. The header of each script states what it runs and the variables it reads.

| Recipe | Pretraining | Contrastive fine-tuning |
|---|---|---|
| From scratch, 15 vintages | `cluster/pretrain_vintages.sbatch` | `cluster/contrastive_vintages.sbatch` |
| Chained, 2011 to 2024 | `cluster/chain_submit.sh` (submits both stages) | |
| Replay control | `cluster/replay_submit.sh` (submits both stages) | |
| Twice the token budget, 2024 | `cluster/budget2x_submit.sh` | |
| ChronoBERT + contrastive fine-tuning | the ChronoBERT vintages, downloaded | `cluster/ablate_backbone.sbatch` |
| ChronoBERT + DAPT, 2010, 2017 and 2024 | `cluster/pack_borrowed.sbatch`, then `cluster/dapt_vintages.sbatch` | `cluster/dapt_contrastive.sbatch` |
| Contrastive seeds 1 and 2 | | `cluster/contrastive_seeds.sbatch` |

Then register the retrievers, so that every later step resolves a vintage by date:

```bash
python scripts/register_vintages.py --apply
```

Measured run times on an NVIDIA RTX 2080 Ti:
- a chained segment or a replay takes 40 to 60 minutes in float16;
- contrastive fine-tuning takes about 6 hours per vintage.

A from-scratch vintage reads 1.2B tokens in bfloat16. That takes about 2 hours on an Ampere or Ada card, and about 20 hours on a Turing card, which emulates bfloat16.

### 4. Retrieval benchmark and look-ahead premia

Each scoring task is one model on one quarter, about 20 minutes on 4 CPU cores.

```bash
sbatch cluster/compare_vintages.sbatch     # vintage Y and bge-small on quarter (Y+1)Q2; set MODELS to add others
sbatch cluster/bench_chain.sbatch          # chained vintages
sbatch cluster/bench_lookahead.sbatch      # later vintages on earlier quarters
sbatch cluster/bench_replay.sbatch         # replay controls
sbatch cluster/bench_staleness.sbatch      # every vintage on 2025Q2
sbatch cluster/bench_2x.sbatch             # twice the token budget
sbatch cluster/bench_seed_premia.sbatch    # contrastive seeds of the 2024 vintages

python -m ecalls.diagnostics.seed_summary --references     # and --decomposition, --chain, --lookahead,
                                                           # --replay, --staleness, --seed-lookahead, --training
python -m ecalls.diagnostics.restatement                   # premium by how much the firm restates each answer
```

### 5. The retrieval-augmented test

```bash
sbatch cluster/dl_llm.sbatch               # ChronoGPT-Instruct readers, 2010 to 2023
sbatch cluster/rag_lookahead.sbatch        # one task per quarter
python -m ecalls.diagnostics.rag_lookahead summary
```

### 6. Forecasts

```bash
python -m ecalls.ingest.link_all           # Capital IQ to Compustat to CRSP, valid on the call date
python -m ecalls.ingest.crsp_returns
python -m ecalls.portfolio.panel_daily     # next-day, five-day and session-after-the-call targets
python -m ecalls.portfolio.panel --universe sp500   # next-quarter return and abnormal volatility; also --universe all
python -m ecalls.portfolio.sue pull && python -m ecalls.portfolio.sue build
sbatch cluster/pf_embeddings_parts.sbatch && sbatch cluster/pf_embeddings_merge.sbatch
sbatch cluster/pf_text_features_all.sbatch
TARGET=ret1d sbatch cluster/pf_backtest.sbatch      # and ret5d, o2c, ret60, sue, av; ADJACENT=1 for the same-year test
UNIVERSE=all sbatch cluster/pf_knn.sbatch           # nearest-neighbor forecast, and UNIVERSE=sp500
python -m ecalls.portfolio.holdout_split --target sue
sbatch cluster/ridge_grid.sbatch
```

### 7. Tables and figures

The result files of steps 4 to 6 are included in `reports/`, and the models' training records in `runs/`, so this step runs without any data or model:

```bash
python -m ecalls.paper.values              # every number the text cites, as LaTeX macros
for m in table_corpus table_families table_settings table_decomposition table_forecasts \
         table_premia table_rag fig_recall fig_lookahead; do python -m ecalls.paper.$m; done
```

The outputs go to `paper/values.tex`, `paper/tables/` and `paper/figures/`.

## The point-in-time contract

`src/ecalls/chunk/registry.py` resolves a model by date or by name, never by a path. `for_date(d)` returns the newest vintage whose training cutoff is on or before `d`. It raises an error rather than fall back to an earlier one, and it never returns a model whose cutoff is unknown. A search dated `d` sees only calls dated on or before `d`. A query that names a vintage with a later cutoff fails, because every vector of that vintage could reflect later text.

```python
from ecalls.chunk.registry import load

reg = load()
reg.for_date("2018-06-30")      # the vintage available on that date
reg.by_name("ecalls-ret-20171231")
```

## Citation

```bibtex
@misc{brini2026pointintime,
  title  = {Point-in-Time Encoders for Earnings Calls},
  author = {Brini, Alessio},
  year   = {2026},
  howpublished = {\url{https://github.com/Alessiobrini/point-in-time-encoders-for-earnings-calls}}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The license covers the code and the files in this repository. It grants no rights to the WRDS data, which remain under their providers' terms.
