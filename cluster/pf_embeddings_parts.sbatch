#!/bin/bash
#SBATCH --job-name=ecalls-pfpart
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --array=0-155
#SBATCH --output=logs/pfpart_%A_%a.out
#SBATCH --error=logs/pfpart_%A_%a.out
set -euo pipefail

# The D28 call embeddings of pf_embeddings_all.sbatch, cut into 4 shard parts per model so each task
# fits the gaps of the scavenger queue. That array asked for 16 hours and 96G and SLURM projected its
# start two and a half days out. Task i: model i / 4, part i % 4. Model m: family m / 13 of
# (ecalls-ret, ecalls-abl, chrono-bert-v1), vintage 2012 + m % 13, every event through the end of the
# following year.
#
# Sizing from the S&P run (job 58023930): at most 1h22m and 22G peak for about 28,000 calls. The full
# universe holds about 5.7 times as many, so the largest model needs about 8 hours, 2 per part.
# cluster/pf_embeddings_merge.sbatch joins the parts.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs"; cd "$ROOT"

PARTS=4
I=${SLURM_ARRAY_TASK_ID:-0}
M=$(( I / PARTS )); K=$(( I % PARTS ))
# FAMILIES overrides the list, e.g. FAMILIES=ecalls-chain-ret with --array=0-51 for the chain (D29).
read -r -a FAMILIES <<< "${FAMILIES:-ecalls-ret ecalls-abl chrono-bert-v1}"
FAM=${FAMILIES[$(( M / 13 ))]}
V=$(( 2012 + M % 13 ))
THROUGH="$(( V + 1 ))-12-31"

echo "host $(hostname)  model ${FAM}-${V}1231  part $K of $PARTS  through $THROUGH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
srun "$PY" scripts/pf_embeddings.py --model "${FAM}-${V}1231" --through "$THROUGH" --device cuda \
  --universe all --parts "$PARTS" --part "$K"
