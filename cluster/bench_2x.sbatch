#!/bin/bash
#SBATCH --job-name=ecalls-b2x
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/b2x_%A_%a.out
#SBATCH --error=logs/b2x_%A_%a.out
set -euo pipefail

# The twice-budget 2024 vintage (D29) on the benchmark, quarters 2011Q2 to 2025Q2, at evaluation seed 0
# so its ranks pair question by question with ecalls-ret-20241231 and the point-in-time vintages.
# Summarize with `python -m ecalls.diagnostics.seed_summary --budget`. CPU, as for bench_chain.sbatch.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$ROOT/logs" "$ROOT/reports/seeds"; cd "$ROOT"

Y=$(( 2010 + ${SLURM_ARRAY_TASK_ID:-0} ))
SHARD=$(( Y + 1 ))Q2
MODEL="ecalls-2x-ret-20241231"
SEED=0

out="seeds/cmp_${SHARD}_${MODEL}_holdout_s${SEED}.json"
echo "host $(hostname)  quarter $SHARD  model $MODEL"
if [ -s "$ROOT/reports/$out" ]; then echo "skip $out, already written"; exit 0; fi
srun "$PY" -m ecalls.diagnostics.bow_vs_dense \
  --shard "$SHARD" --model "$MODEL" --n-calls 400 --holdout-only --seed "$SEED" \
  --ranks-out "bench_ranks/${SHARD}_${MODEL}_s${SEED}.npz" \
  --out "$out" --device cpu
