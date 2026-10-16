#!/bin/bash
#SBATCH --job-name=ecalls-ret
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/ret_%A_%a.out
#SBATCH --error=logs/ret_%A_%a.out
set -euo pipefail

# The contrastive stage for the 15 annual vintages, one per array task, each starting from its own
# pretrained backbone. Masked language modeling alone is a weak retriever on this corpus, measured at
# 0.2158 within-call recall@1 against 0.4804 for term counting, so this stage is what makes the vintage
# set useful rather than a later refinement.
#
# One pair set serves every task, selected by shard, for the same reason one packed corpus serves the
# masked language modeling array. `--cutoff` is what restricts a task to its own window, and the trainer
# refuses a backbone whose recorded cutoff disagrees with it.
#
# The held-out pair set supplies the progress check, so a task reports the number Phase 4 steers by as
# it runs rather than only at the end. The trainer refuses a training set in that slot.
#
# Requeue-safe: the checkpoint carries the optimizer, the gradient scaler and the call sampler.

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
# The backbone and retriever name tags. The chained family (D29) runs this script one year at a time
# with MLM_TAG=chain-mlm RET_TAG=chain-ret, and the 2x budget check with MLM_TAG=mlm2x RET_TAG=2x-ret.
MLM_TAG=${MLM_TAG:-mlm}
RET_TAG=${RET_TAG:-ret}

# The batch is left at its default and is identical on every card, because here the batch size is the
# negative count and therefore a hyperparameter. Gradient caching is what allows that: the full
# batch's gradient is assembled from microbatches, exactly rather than approximately, so only the
# microbatch varies with the allocated GPU. Without it a vintage that drew a 2080 Ti would train
# against a different loss than one that drew an A5000, and scavenger hands out whatever is free.
#
# --microbatch is therefore the only memory knob, and overriding it cannot change the result.
EXTRA=""
if [ -n "${CALLS_PER_BATCH:-}" ]; then EXTRA="$EXTRA --calls-per-batch $CALLS_PER_BATCH"; fi
if [ -n "${PAIRS_PER_CALL:-}" ]; then EXTRA="$EXTRA --pairs-per-call $PAIRS_PER_CALL"; fi
if [ -n "${MICROBATCH:-}" ]; then EXTRA="$EXTRA --microbatch $MICROBATCH"; fi

echo "host $(hostname)  vintage $YEAR  steps $STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" -m ecalls.train.contrastive \
  --mlm "ecalls-${MLM_TAG}-${YEAR}1231" \
  --pairs "$PAIRS" --holdout-pairs "$HOLDOUT_PAIRS" \
  --cutoff "${YEAR}-12-31" \
  --steps "$STEPS" \
  --compute-dtype float16 --require-gpu \
  --checkpoint-every 200 --log-every 100 --eval-every 500 \
  --name "ecalls-${RET_TAG}-${YEAR}1231" $EXTRA
