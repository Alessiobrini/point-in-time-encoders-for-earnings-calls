#!/bin/bash
#SBATCH --job-name=ecalls-rpl
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-13
#SBATCH --output=logs/rpl_%A_%a.out
#SBATCH --error=logs/rpl_%A_%a.out
set -euo pipefail

# The compute-matched replay control of D31, one task per year Y = 2010 to 2023. Each continues chained
# vintage Y (ecalls-mlm-20101231 for 2010) on its own last year for as many tokens as chained segment Y+1
# read, so it matches Y+1's compute and reads no text after Y. The tasks are independent, unlike the
# chain itself, so they run in parallel. float16 for the chain's reason (L13). The largest segment is
# 0.21 B tokens, about an hour at the float16 rate on a 2080 Ti, so 6 hours.
#
# A task whose record already exists exits at once, so the array can be resubmitted after a failure.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs"; cd "$ROOT"

Y=$(( 2010 + ${SLURM_ARRAY_TASK_ID:-0} ))
WINDOW=${WINDOW:-ecalls-bpe-32768-20101231-ctx768-20241231}
# SEED and PREFIX as in chain_segment.sbatch, for a pretraining-seed replicate of the chain.
SEED=${SEED:-0}
PREFIX=${PREFIX:-}
NAME="ecalls-${PREFIX}rpl-mlm-${Y}1231"
if [ "$Y" -eq 2010 ]; then PARENT="ecalls-${PREFIX}mlm-20101231"; else PARENT="ecalls-${PREFIX}chain-mlm-${Y}1231"; fi
TARGET="ecalls-${PREFIX}chain-mlm-$(( Y + 1 ))1231"

if [ -f "$ECALLS_DATA_ROOT/processed/models/$NAME/_run.json" ]; then
  echo "$NAME already finished, nothing to do"
  exit 0
fi

echo "host $(hostname)  replay $Y  parent $PARENT  matches $TARGET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" -m ecalls.train.pretrain \
  --window "$WINDOW" --cutoff "${Y}-12-31" \
  --init-from "$PARENT" --replay-of "$TARGET" \
  --compute-dtype float16 --require-gpu --seed "$SEED" \
  --checkpoint-every 250 --log-every 100 \
  --name "$NAME"
