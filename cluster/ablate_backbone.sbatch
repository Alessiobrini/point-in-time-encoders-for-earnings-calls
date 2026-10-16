#!/bin/bash
#SBATCH --job-name=ecalls-abl
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/abl_%A_%a.out
#SBATCH --error=logs/abl_%A_%a.out
set -euo pipefail

# The ablation that turns the comparison into a finding.
#
# Our vintages beat bge-small by +0.1893 within-call recall@1 across all fifteen quarters. That does
# not show finance pretraining helps, because our model also got a contrastive stage on in-domain
# pairs and trained on exactly the benchmark's task, while bge-small got neither. Three effects are
# stacked and the measurement cannot separate them.
#
# This runs the identical contrastive stage, on the identical pairs, with the identical batch, from
# ChronoBERT's checkpoint of the same year: general-domain pretraining with a known cutoff. Then
#
#   ours vs this          isolates finance-domain pretraining
#   this vs bge-small     isolates the in-domain pairs and the task format
#
# Capacity is not matched. ChronoBERT is 149 M against our 33.8 M, which is stated rather than
# corrected, because if the smaller finance-pretrained model still wins that is the stronger result
# and if it loses the capacity difference is the first thing to suspect.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs"; cd "$ROOT"

FIRST_YEAR=${FIRST_YEAR:-2010}
YEAR=$(( FIRST_YEAR + ${SLURM_ARRAY_TASK_ID:-0} ))
PAIRS=${PAIRS:-qa-pairs-20241231}
HOLDOUT_PAIRS=${HOLDOUT_PAIRS:-qa-pairs-20241231-holdout}
STEPS=${STEPS:-4000}

echo "host $(hostname)  vintage $YEAR  backbone chrono-bert-v1-${YEAR}1231  steps $STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" -m ecalls.train.contrastive \
  --backbone "chrono-bert-v1-${YEAR}1231" \
  --pairs "$PAIRS" --holdout-pairs "$HOLDOUT_PAIRS" \
  --cutoff "${YEAR}-12-31" \
  --steps "$STEPS" \
  --compute-dtype float16 --require-gpu \
  --checkpoint-every 200 --log-every 100 --eval-every 500 \
  --name "ecalls-abl-${YEAR}1231"
