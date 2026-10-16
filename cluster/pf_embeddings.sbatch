#!/bin/bash
#SBATCH --job-name=ecalls-pfemb
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-38
#SBATCH --output=logs/pfemb_%A_%a.out
#SBATCH --error=logs/pfemb_%A_%a.out
set -euo pipefail

# Call embeddings for the portfolio test (D26). Task i: family i / 13 of (ecalls-ret, ecalls-abl,
# chrono-bert-v1), vintage 2012 + i % 13, embedding every in-index call from 2011 through the end of
# the following year. The 2024 task covers every call and also serves the leaky variant.
#
# Time limit from a measurement: job 57904135 embedded 50,165 chunks in 2.5 min with ours and 3.5 min
# with the 149 M models on a 2080 Ti. The largest task is about 1.0 M chunks, so about 70 min.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs"; cd "$ROOT"

I=${SLURM_ARRAY_TASK_ID:-0}
# FAMILIES overrides the list, e.g. FAMILIES=ecalls-chain-ret with --array=0-12 for the chain (D29).
read -r -a FAMILIES <<< "${FAMILIES:-ecalls-ret ecalls-abl chrono-bert-v1}"
FAM=${FAMILIES[$(( I / 13 ))]}
V=$(( 2012 + I % 13 ))
THROUGH="$(( V + 1 ))-12-31"

echo "host $(hostname)  model ${FAM}-${V}1231  through $THROUGH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
srun "$PY" scripts/pf_embeddings.py --model "${FAM}-${V}1231" --through "$THROUGH" --device cuda
