#!/bin/bash
#SBATCH --job-name=ecalls-bseed
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-44
#SBATCH --output=logs/bseed_%A_%a.out
#SBATCH --error=logs/bseed_%A_%a.out
set -euo pipefail

# The benchmark again, three evaluation draws per quarter, with the BM25 weight scanned and the
# per-question ranks kept. Task i scores vintage 2010 + i % 15 on the quarter after its cutoff under
# evaluation seed i / 15, for our model, bge-small and the ablation, as compare_vintages.sbatch does.
#
# Seed 0 redraws the calls the canonical comparison used, so its tfidf, bm25, dense and hybrid numbers
# must equal reports/cmp_*_holdout.json exactly. That is the check that this run measures the same
# thing. Outputs go to reports/seeds/ and never to the canonical names, which the paper reads.
#
# The ranks are identifiers and integers only, kept under DATA_ROOT because they are large. The call,
# not the question, is the unit a bootstrap resamples, so each file carries the question's call.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs"; cd "$ROOT"
mkdir -p "$ROOT/reports/seeds"

I=${SLURM_ARRAY_TASK_ID:-0}
YEAR=$(( 2010 + I % 15 ))
SEED=$(( I / 15 ))
SHARD=$(( YEAR + 1 ))Q2
MODELS=${MODELS:-"ecalls-ret-${YEAR}1231 bge-small-en-v1.5 ecalls-abl-${YEAR}1231"}
# EXTRA_FAMILIES adds models named <family>-<YEAR>1231, for the training-seed replicates and the DAPT
# retrievers. They are scored on the same calls and questions as the others at this evaluation seed,
# so their ranks pair with the seed-0 models question by question. The skip below leaves the
# existing outputs alone, so an array run for extra families rescores nothing else.
for f in ${EXTRA_FAMILIES:-}; do MODELS="$MODELS ${f}-${YEAR}1231"; done

echo "host $(hostname)  vintage $YEAR  shard $SHARD  evaluation seed $SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for model in $MODELS; do
  out="seeds/cmp_${SHARD}_${model}_holdout_s${SEED}.json"
  if [ -s "$ROOT/reports/$out" ]; then echo "skip $out, already written"; continue; fi
  echo "=============== $SHARD  $model  seed $SEED"
  srun "$PY" -m ecalls.diagnostics.bow_vs_dense \
    --shard "$SHARD" --model "$model" --n-calls 400 --holdout-only --seed "$SEED" \
    --fusion-scan --ranks-out "bench_ranks/${SHARD}_${model}_s${SEED}.npz" \
    --out "$out" --device cuda
done
