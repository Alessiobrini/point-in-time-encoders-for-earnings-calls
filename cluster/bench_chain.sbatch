#!/bin/bash
#SBATCH --job-name=ecalls-bchain
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-41%30
#SBATCH --output=logs/bchain_%A_%a.out
#SBATCH --error=logs/bchain_%A_%a.out
set -euo pipefail

# The chained retrievers (D29) on the benchmark, at evaluation seed 0 so their ranks pair question by
# question with every other family. Tasks 0 to 13: chained vintage Y on quarter (Y+1)Q2, Y = 2011 to
# 2024 (the chain's 2010 member is ecalls-ret-20101231, already scored). Tasks 14 to 27: the 2024
# chained vintage on quarters 2011Q2 to 2024Q2 (leaky). Tasks 28 to 41: chained vintage Y+1 on (Y+1)Q2
# (adjacent). The leaky and adjacent models coincide at 2024Q2, and the skip scores it once. CPU, as
# for bench_lookahead.sbatch.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$ROOT/logs" "$ROOT/reports/seeds"; cd "$ROOT"

I=${SLURM_ARRAY_TASK_ID:-0}
KIND=$(( I / 14 ))
J=$(( I % 14 ))
if [ "$KIND" -eq 0 ]; then
  Y=$(( 2011 + J )); V=$Y
elif [ "$KIND" -eq 1 ]; then
  Y=$(( 2010 + J )); V=2024
else
  Y=$(( 2010 + J )); V=$(( Y + 1 ))
fi
SHARD=$(( Y + 1 ))Q2
MODEL="ecalls-chain-ret-${V}1231"
SEED=0

out="seeds/cmp_${SHARD}_${MODEL}_holdout_s${SEED}.json"
echo "host $(hostname)  quarter $SHARD  model $MODEL"
if [ -s "$ROOT/reports/$out" ]; then echo "skip $out, already written"; exit 0; fi
srun "$PY" -m ecalls.diagnostics.bow_vs_dense \
  --shard "$SHARD" --model "$MODEL" --n-calls 400 --holdout-only --seed "$SEED" \
  --ranks-out "bench_ranks/${SHARD}_${MODEL}_s${SEED}.npz" \
  --out "$out" --device cpu
