#!/bin/bash
#SBATCH --job-name=ecalls-bstale
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-26
#SBATCH --output=logs/bstale_%A_%a.out
#SBATCH --error=logs/bstale_%A_%a.out
set -euo pipefail

# Staleness on a fixed window, PALM's question for retrievers: every from-scratch vintage 2010 to 2023
# (tasks 0 to 13) and every chained vintage 2011 to 2023 (tasks 14 to 26) on the 2025Q2 held-out
# questions at evaluation seed 0, so each pairs question by question with the 2024 vintages already
# scored there. The chain's 2010 member is ecalls-ret-20101231. CPU, as for bench_chain.sbatch.
# Summarize with `python -m ecalls.diagnostics.seed_summary --staleness`.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$ROOT/logs" "$ROOT/reports/seeds"; cd "$ROOT"

I=${SLURM_ARRAY_TASK_ID:-0}
if [ "$I" -lt 14 ]; then MODEL="ecalls-ret-$(( 2010 + I ))1231"; else MODEL="ecalls-chain-ret-$(( 2011 + I - 14 ))1231"; fi
SHARD=2025Q2
SEED=0

out="seeds/cmp_${SHARD}_${MODEL}_holdout_s${SEED}.json"
echo "host $(hostname)  quarter $SHARD  model $MODEL"
if [ -s "$ROOT/reports/$out" ]; then echo "skip $out, already written"; exit 0; fi
srun "$PY" -m ecalls.diagnostics.bow_vs_dense \
  --shard "$SHARD" --model "$MODEL" --n-calls 400 --holdout-only --seed "$SEED" \
  --ranks-out "bench_ranks/${SHARD}_${MODEL}_s${SEED}.npz" \
  --out "$out" --device cpu
