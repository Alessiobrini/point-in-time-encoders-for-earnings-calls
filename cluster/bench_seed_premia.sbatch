#!/bin/bash
#SBATCH --job-name=ecalls-bsprem
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-55%28
#SBATCH --output=logs/bsprem_%A_%a.out
#SBATCH --error=logs/bsprem_%A_%a.out
set -euo pipefail

# D33, the training-seed check of the 2024 look-ahead premium. One draw of each 2024 vintage serves
# all 14 test quarters, so its seed noise does not average out across quarters and the call bootstrap
# cannot see it. This array scores the seed 1 and 2 retrievers of the 2024 vintage (contrastive stage
# reseeded, pretrained encoder fixed) on the same held-out questions as bench_lookahead.sbatch, so
# `seed_summary --seed-lookahead` can pair each with the point-in-time vintage question by question.
# CPU, as bench_lookahead: about 9,500 texts per quarter, 8.4 chunks/s for ours, 1.9 for the 149M
# ChronoBERT encoder on 4 cores.
# Task i: replicate i / 14 of the four below, quarter 2011Q2 + i % 14.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$ROOT/logs" "$ROOT/reports/seeds"; cd "$ROOT"

I=${SLURM_ARRAY_TASK_ID:-0}
REPLICATES=(ecalls-seed1-ret ecalls-seed2-ret ecalls-seed1-abl ecalls-seed2-abl)
MODEL="${REPLICATES[$(( I / 14 ))]}-20241231"
Y=$(( 2010 + I % 14 ))
SHARD=$(( Y + 1 ))Q2
SEED=0

out="seeds/cmp_${SHARD}_${MODEL}_holdout_s${SEED}.json"
echo "host $(hostname)  quarter $SHARD  model $MODEL"
if [ -s "$ROOT/reports/$out" ]; then echo "skip $out, already written"; exit 0; fi
srun "$PY" -m ecalls.diagnostics.bow_vs_dense \
  --shard "$SHARD" --model "$MODEL" --n-calls 400 --holdout-only --seed "$SEED" \
  --ranks-out "bench_ranks/${SHARD}_${MODEL}_s${SEED}.npz" \
  --out "$out" --device cpu
