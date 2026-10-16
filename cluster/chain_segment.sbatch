#!/bin/bash
#SBATCH --job-name=ecalls-chain
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/chain_%j.out
#SBATCH --error=logs/chain_%j.out
set -euo pipefail

# One segment of the chained vintages (D29), for the year in YEAR. The segment starts from the finished
# model of the previous year end, ecalls-mlm-20101231 for 2011 and ecalls-chain-mlm-<YEAR-1>1231 after,
# and trains once over that year's new calls with its own warmup and decay. The trainer refuses a parent
# that is unfinished, has no run record, or ends anywhere but the previous year end.
#
# float16, because bfloat16 is emulated on the Turing cards that make up most of scavenger-gpu (L13,
# L17): 17.0 k tokens/s against 59.4 k in float16 on a 2080 Ti. The chain runs in sequence, so a slow
# card delays every later segment. The largest increment is 2022 at 274,035 sequences, 0.210 B tokens,
# about an hour at the float16 rate and 4.3 hours at the slowest bfloat16 rate measured, so 6 hours.
#
# Requeue-safe for the same reason as the from-scratch array. A segment whose record already exists exits
# at once, because rerunning a finished run rewrites its record with no final loss, and chain_submit.sh
# may be run again after a break in the chain.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs"; cd "$ROOT"

: "${YEAR:?set YEAR to the year of the segment, 2011 to 2024}"
WINDOW=${WINDOW:-ecalls-bpe-32768-20101231-ctx768-20241231}
# A pretraining-seed replicate of the chain sets SEED and PREFIX=pseed<S>-, so every name it reads and
# writes carries the prefix and the chain cannot continue, or overwrite, the seed-0 one.
SEED=${SEED:-0}
PREFIX=${PREFIX:-}
NAME="ecalls-${PREFIX}chain-mlm-${YEAR}1231"
if [ "$YEAR" -eq 2011 ]; then
  PARENT="ecalls-${PREFIX}mlm-20101231"
else
  PARENT="ecalls-${PREFIX}chain-mlm-$(( YEAR - 1 ))1231"
fi

if [ -f "$ECALLS_DATA_ROOT/processed/models/$NAME/_run.json" ]; then
  echo "$NAME already finished, nothing to do"
  exit 0
fi

echo "host $(hostname)  segment $YEAR  parent $PARENT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# No --token-budget: a segment's budget is one pass over its increment and the trainer refuses the flag.
# The default tokens per step, peak learning rate and warmup fraction, as the from-scratch vintages ran.
srun "$PY" -m ecalls.train.pretrain \
  --window "$WINDOW" --cutoff "${YEAR}-12-31" \
  --init-from "$PARENT" \
  --compute-dtype float16 --require-gpu --seed "$SEED" \
  --checkpoint-every 250 --log-every 100 \
  --name "$NAME"
