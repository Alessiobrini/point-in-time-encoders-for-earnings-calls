#!/bin/bash
#SBATCH --job-name=ecalls-compare
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.out
set -euo pipefail

# The comparison that decides whether Phase 4 produced anything, run so that it is actually a
# comparison. Every model is scored on the same shard, the same seed and the same holdout population,
# so `bow_vs_dense` builds one candidate pool and all scorers see it.
#
# The unfiltered 0.5378 in the Phase 3 report is NOT the number to compare against, because it was
# measured on every call including the ones our models trained on. bge-small is re-run here under
# --holdout-only so both sides come from the same population.
#
# SHARDS must lie inside the vintage's window, or the vintage is being asked about language it never
# saw and the result measures staleness rather than quality.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs" "$ROOT/reports"; cd "$ROOT"

SHARDS=${SHARDS:-"2010Q2 2010Q3"}
MODELS=${MODELS:-"bge-small-en-v1.5 ecalls-ret-20101231"}
NCALLS=${NCALLS:-400}

echo "host $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for shard in $SHARDS; do
  for model in $MODELS; do
    echo "=============== $shard  $model  (held-out calls only)"
    srun "$PY" -m ecalls.diagnostics.bow_vs_dense \
      --shard "$shard" --model "$model" --n-calls "$NCALLS" --holdout-only \
      --out "reports/compare_${shard}_${model}_holdout.json"
  done
done
