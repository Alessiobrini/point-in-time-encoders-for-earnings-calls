#!/bin/bash
#SBATCH --job-name=ecalls-packb
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/packb_%j.out
#SBATCH --error=logs/packb_%j.out
set -euo pipefail

# The corpus packed with ChronoBERT's vocabulary rather than ours, which domain-adaptive pretraining
# needs: continuing masked language modeling from those weights means every token id has to mean what
# it meant during their pretraining.
#
# One file again serves every vintage as a calendar-order prefix, so this runs once at the latest
# cutoff. ChronoBERT's tokenizer is shared across its annual checkpoints, and its base model is
# pretrained on pre-2000 text, so borrowing the 2024 checkpoint's vocabulary does not import later
# vocabulary. The vintage restriction still comes from the prefix, as it does for our own corpus.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"; cd "$ROOT"

TOKENIZER=${TOKENIZER:-registry:chrono-bert-v1-20241231}
CUTOFF=${CUTOFF:-2024-12-31}
EXTRA=""
if [ "${HOLDOUT:-0}" = "1" ]; then EXTRA="--holdout-only"; fi

echo "host $(hostname)  tokenizer $TOKENIZER  cutoff $CUTOFF  ${EXTRA:-training}"
srun "$PY" -m ecalls.train.pretrain_data --cutoff "$CUTOFF" --tokenizer "$TOKENIZER" $EXTRA
