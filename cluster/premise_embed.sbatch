#!/bin/bash
#SBATCH --job-name=ecalls-pemb
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/pemb_%j.out
#SBATCH --error=logs/pemb_%j.out
set -euo pipefail

# Call embeddings for the premise test re-run (L16). The primary design is D11's: one vintage embeds
# every panel call. 2021 is the newest cutoff before the regression's training cut of 2022-02-10, so
# no row is embedded by a model that saw the test period, which predictive_premise checks. All three
# families, ChronoBERT included, so the comparison holds the design fixed.
#
# Per-call resolution for our two families is secondary. Our vintages are pretrained independently,
# so it mixes unrelated embedding spaces in one regression. It is run so the original ChronoBERT
# result has a counterpart under the same design.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs"; cd "$ROOT"

for m in ecalls-ret-20211231 ecalls-abl-20211231 chrono-bert-v1-20211231; do
  echo "=============== one vintage: $m"
  srun "$PY" scripts/call_embeddings.py --model "$m" --encode --device cuda
done
for f in ecalls-ret ecalls-abl; do
  echo "=============== per call: $f"
  srun "$PY" scripts/call_embeddings.py --family "$f" --encode --device cuda
done
