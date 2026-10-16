#!/bin/bash
#SBATCH --job-name=ecalls-dapt
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/dapt_%A_%a.out
#SBATCH --error=logs/dapt_%A_%a.out
set -euo pipefail

# Domain-adaptive pretraining: continue masked language modeling from ChronoBERT's checkpoint of the
# same year, on our corpus up to that year, then run the same contrastive stage. Gururangan et al.
# (2020), doi 10.18653/v1/2020.acl-main.740.
#
# This is the experiment that separates domain from scale. The ablation showed a general backbone
# given our contrastive stage beats our own finance backbone by 0.0314, but at 4.4 times the
# parameters and roughly 380 times the pretraining tokens, so it cannot say whether the finance
# corpus is worthless or merely outweighed. Here the weights, the capacity and the contrastive stage
# are identical to the ablation's and the only difference is whether the finance corpus was seen.
#
# YEARS, not an array index, because this runs on three vintages first. Spanning the range answers
# the question at a twelfth of the cost, and the corpus a vintage sees grows ninefold across it:
# 2010 is the smallest window, 2024 the largest, 2017 the middle.
#
# float16, not bfloat16. bfloat16 is emulated on the 2080 Ti cards most scavenger nodes carry and ran at
# 5.1 to 5.4 k tokens/s, so all three tasks hit the 24 h wall. float16 with the gradient scaler measured
# 16.2 k tokens/s on the same card (job 57841613), about 20.6 h for the budget, so the limit is 48 h (L17).

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs"; cd "$ROOT"

YEARS=(${YEARS:-2010 2017 2024})
YEAR=${YEARS[${SLURM_ARRAY_TASK_ID:-0}]}
WINDOW=${WINDOW:-borrowed-chrono-bert-v1-20241231-ctx768-20241231}
BUDGET=${BUDGET:-1200000000}

echo "host $(hostname)  vintage $YEAR  backbone chrono-bert-v1-${YEAR}1231  budget $BUDGET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" -m ecalls.train.pretrain \
  --window "$WINDOW" --cutoff "${YEAR}-12-31" \
  --tokenizer "registry:chrono-bert-v1-20241231" \
  --backbone "chrono-bert-v1-${YEAR}1231" \
  --token-budget "$BUDGET" \
  --peak-lr "${PEAK_LR:-5e-5}" \
  --compute-dtype float16 --require-gpu \
  --checkpoint-every 250 --log-every 100 \
  --name "ecalls-dapt-${YEAR}1231"
