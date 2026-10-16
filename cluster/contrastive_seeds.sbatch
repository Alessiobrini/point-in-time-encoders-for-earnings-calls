#!/bin/bash
#SBATCH --job-name=ecalls-tseed
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/tseed_%A_%a.out
#SBATCH --error=logs/tseed_%A_%a.out
set -euo pipefail

# Training-seed replicates of the contrastive stage, for the Antoniak and Mimno standard: does the
# 15-of-15 ordering survive retraining. Three vintages spanning the range, seeds 1 and 2, both
# families. Task i: vintage (2010, 2017, 2024)[i % 3], seed 1 + (i / 3) % 2, family ret for i < 6 and
# abl from 6.
#
# The seed changes the call sampler and the pair draws. The masked language modeling stage is not
# rerun and the backbone weights are loaded over any seeded initialization, so the replicates share
# a backbone and differ only in the contrastive stage.
#
# Everything else is identical to contrastive_vintages.sbatch and ablate_backbone.sbatch. The name
# comes from run_name, which puts the seed before the tag (ecalls-seed1-ret-20101231), so a replicate
# can never resume into the seed-0 deliverable's directory and can never tie with it in the registry.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs"; cd "$ROOT"

I=${SLURM_ARRAY_TASK_ID:-0}
YEARS=(2010 2017 2024)
YEAR=${YEARS[$(( I % 3 ))]}
SEED=$(( 1 + (I / 3) % 2 ))
if [ "$I" -lt 6 ]; then SRC="--mlm ecalls-mlm-${YEAR}1231"; else SRC="--backbone chrono-bert-v1-${YEAR}1231"; fi

echo "host $(hostname)  vintage $YEAR  seed $SEED  $SRC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" -m ecalls.train.contrastive $SRC \
  --pairs qa-pairs-20241231 --holdout-pairs qa-pairs-20241231-holdout \
  --cutoff "${YEAR}-12-31" --steps 4000 --seed "$SEED" \
  --compute-dtype float16 --require-gpu \
  --checkpoint-every 200 --log-every 100 --eval-every 500
