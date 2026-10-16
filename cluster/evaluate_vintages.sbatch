#!/bin/bash
#SBATCH --job-name=ecalls-eval
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.out
set -euo pipefail

# Masked language modeling loss for all fifteen vintages on one held-out set, which is what makes the
# numbers comparable. The fifteen training losses are each measured on that vintage's own corpus, and a
# later vintage's corpus is larger and more varied, so the rising trend L14 reports is not yet evidence
# of a convergence ranking. One set inside every window settles it.
#
# All fifteen models in one job rather than an array, so the mask pattern and the device are identical
# across them and the only thing that differs is the weights.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs" "$ROOT/reports"; cd "$ROOT"

WINDOW=${WINDOW:-ecalls-bpe-32768-20101231-ctx768-20241231-holdout}
echo "host $(hostname)  window $WINDOW"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Named after the window, because two evaluation sets are in play and a fixed filename let the second
# run overwrite the first. The full-span set is out of window for the early vintages, which biases
# against them; the 2010 set is in window for all fifteen and is the clean common yardstick.
TAG=$(echo "$WINDOW" | sed 's/.*ctx[0-9]*-//')
srun "$PY" -m ecalls.train.evaluate --window "$WINDOW" --require-gpu \
  --n-sequences "${N_SEQUENCES:-20000}" --batch "${BATCH:-16}" \
  --out "$ROOT/reports/holdout_mlm_loss_${TAG}.json"
