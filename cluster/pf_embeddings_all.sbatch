#!/bin/bash
#SBATCH --job-name=ecalls-pfall
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-38
#SBATCH --output=logs/pfall_%A_%a.out
#SBATCH --error=logs/pfall_%A_%a.out
set -euo pipefail

# Call embeddings for the D28 next-day test on the full linked universe, 159,134 events. Task i:
# family i / 13 of (ecalls-ret, ecalls-abl, chrono-bert-v1), vintage 2012 + i % 13, every event from
# 2011 through the end of the following year, with all, Q&A and presentation call vectors from one pass.
#
# Time limit from measured rates, 330 chunks/s for ours and 240 for the 149M models on a 2080 Ti
# (job 57904135): the largest task is about 4.3M chunks, about 5 hours on the slower models.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs"; cd "$ROOT"

I=${SLURM_ARRAY_TASK_ID:-0}
FAMILIES=(ecalls-ret ecalls-abl chrono-bert-v1)
FAM=${FAMILIES[$(( I / 13 ))]}
V=$(( 2012 + I % 13 ))
THROUGH="$(( V + 1 ))-12-31"

echo "host $(hostname)  model ${FAM}-${V}1231  through $THROUGH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
srun "$PY" scripts/pf_embeddings.py --model "${FAM}-${V}1231" --through "$THROUGH" --device cuda --universe all
