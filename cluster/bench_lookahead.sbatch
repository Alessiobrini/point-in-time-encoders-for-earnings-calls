#!/bin/bash
#SBATCH --job-name=ecalls-blook
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-55%30
#SBATCH --output=logs/blook_%A_%a.out
#SBATCH --error=logs/blook_%A_%a.out
set -euo pipefail

# The look-ahead premium through the encoder, in retrieval. For each test quarter (Y+1)Q2, the
# point-in-time vintage is Y, already scored at evaluation seed 0 by bench_seeds.sbatch. This array
# scores the same calls and questions with two later vintages: the 2024 vintage (leaky) and the
# same-year vintage Y+1 (adjacent, the JFE paper's cleanest test). Quarters 2011Q2 to 2024Q2, since
# 2025Q2 has no later vintage. Held-out questions were never trained on by any vintage, so what a
# later vintage can know is the rest of that quarter's calls and everything after.
#
# CPU, because the scavenger GPU queue is days long and the benchmark is small: about 9,500 texts per
# quarter, measured at 8.4 chunks/s for ours and 1.9 for the 149M ablation on 4 cores.
# Task i: family i / 28 of (ecalls-ret, ecalls-abl), kind (i % 28) / 14 of (leaky, adjacent),
# quarter 2011Q2 + i % 14. The leaky and adjacent models coincide at 2024Q2, and the skip below scores
# it once.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$ROOT/logs" "$ROOT/reports/seeds"; cd "$ROOT"

I=${SLURM_ARRAY_TASK_ID:-0}
FAMILIES=(ecalls-ret ecalls-abl)
FAM=${FAMILIES[$(( I / 28 ))]}
KIND=$(( (I % 28) / 14 ))
Y=$(( 2010 + I % 14 ))                 # the point-in-time vintage for quarter (Y+1)Q2
SHARD=$(( Y + 1 ))Q2
if [ "$KIND" -eq 0 ]; then V=2024; else V=$(( Y + 1 )); fi
MODEL="${FAM}-${V}1231"
SEED=0

out="seeds/cmp_${SHARD}_${MODEL}_holdout_s${SEED}.json"
echo "host $(hostname)  quarter $SHARD  point-in-time ${FAM}-${Y}1231  later $MODEL"
if [ -s "$ROOT/reports/$out" ]; then echo "skip $out, already written"; exit 0; fi
srun "$PY" -m ecalls.diagnostics.bow_vs_dense \
  --shard "$SHARD" --model "$MODEL" --n-calls 400 --holdout-only --seed "$SEED" \
  --ranks-out "bench_ranks/${SHARD}_${MODEL}_s${SEED}.npz" \
  --out "$out" --device cpu
