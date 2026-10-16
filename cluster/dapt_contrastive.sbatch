#!/bin/bash
#SBATCH --job-name=ecalls-dret
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/dret_%A_%a.out
#SBATCH --error=logs/dret_%A_%a.out
set -euo pipefail

# The contrastive stage on the domain-adapted backbones, identical to ablate_backbone.sbatch in every
# setting. The ablation started from ChronoBERT's checkpoint, this starts from that checkpoint
# continued on our corpus, so the two differ only in whether the finance text was seen (L15).
# load_dapt_backbone refuses a continued run that has not finished its token budget.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs"; cd "$ROOT"

YEARS=(2010 2017 2024)
YEAR=${YEARS[${SLURM_ARRAY_TASK_ID:-0}]}

echo "host $(hostname)  vintage $YEAR  backbone ecalls-dapt-${YEAR}1231"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" -m ecalls.train.contrastive --mlm "ecalls-dapt-${YEAR}1231" \
  --pairs qa-pairs-20241231 --holdout-pairs qa-pairs-20241231-holdout \
  --cutoff "${YEAR}-12-31" --steps 4000 \
  --compute-dtype float16 --require-gpu \
  --checkpoint-every 200 --log-every 100 --eval-every 500
