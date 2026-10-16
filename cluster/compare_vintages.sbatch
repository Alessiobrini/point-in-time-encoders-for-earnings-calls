#!/bin/bash
#SBATCH --job-name=ecalls-cmp
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/cmp_%A_%a.out
#SBATCH --error=logs/cmp_%A_%a.out
set -euo pipefail

# The comparison that decides whether Phase 4 produced anything.
#
# Each vintage is scored on the quarter after its own cutoff, not on one fixed quarter shared by all
# fifteen. That is the point-in-time deployment condition, and it is also what the L14 measurement
# forces: the vintages' loss rankings on two different periods are anti-correlated at -0.913, so a
# single-period comparison across vintages measures which vintage matches that period rather than
# which is better. See reports/2026-10-01_holdout-mlm-loss-across-vintages.md.
#
# Both models in a task run on the same shard, seed and holdout population, so `bow_vs_dense` builds
# one candidate pool and every scorer sees it. Held-out calls only: our models trained on the rest,
# and the unfiltered 0.5378 from the Phase 3 report is therefore not a legitimate target.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs" "$ROOT/reports"; cd "$ROOT"

FIRST_YEAR=${FIRST_YEAR:-2010}
YEAR=$(( FIRST_YEAR + ${SLURM_ARRAY_TASK_ID:-0} ))
SHARD=${SHARD:-$(( YEAR + 1 ))Q2}          # the quarter after the cutoff: deployment, not hindsight
NCALLS=${NCALLS:-400}
MODELS=${MODELS:-"ecalls-ret-${YEAR}1231 bge-small-en-v1.5"}
# WITH_ABLATION=1 adds the ChronoBERT-backed model for the same year, so all three are scored on one
# candidate pool. Scoring it in a separate run would put it on a different pool and the two
# differences, ours against it and it against bge-small, would not be subtractable.
if [ "${WITH_ABLATION:-0}" = "1" ]; then MODELS="$MODELS ecalls-abl-${YEAR}1231"; fi

echo "host $(hostname)  vintage $YEAR  shard $SHARD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for model in $MODELS; do
  echo "=============== $SHARD  $model  (held-out calls only)"
  srun "$PY" -m ecalls.diagnostics.bow_vs_dense \
    --shard "$SHARD" --model "$model" --n-calls "$NCALLS" --holdout-only \
    --out "cmp_${SHARD}_${model}_holdout.json"
done
