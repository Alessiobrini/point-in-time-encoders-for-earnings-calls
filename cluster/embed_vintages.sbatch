#!/bin/bash
#SBATCH --job-name=ecalls-vintage
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# 24h, not 8h. **Measured:** the 2023 vintage embeds 77 shards and needs about 11 hours on one
# GPU, and two tasks were on course to be killed by an 8h limit with work unfinished. Resume is
# per shard so nothing is lost when that happens, but the queue wait is paid twice. scavenger
# allows 7 days, and a longer limit costs nothing on a preemptible partition.
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/vintage_%A_%a.out
#SBATCH --error=logs/vintage_%A_%a.out

# The vintage set: one annual ChronoBERT checkpoint per array task, each embedding the corpus up to its
# own training cutoff. That expanding window is the roadmap's Phase 5 design, and `--up-to-cutoff` makes
# it a property of the model rather than a shard list someone has to remember.
#
# **One task per vintage rather than the usual shards-across-tasks split.** Each task loads a different
# model, so splitting a vintage across tasks would load the same weights sixteen times. The work is
# uneven by construction, since the 2024 vintage embeds the whole history and the 2010 vintage embeds a
# twelfth of it, and that is fine: the long tasks start first and the short ones fill in behind them.
#
# Requeue-safe for the same reason as the other arrays: resume compares coverage per shard, so a
# preempted vintage re-runs only the shard it was interrupted on.

set -euo pipefail

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}

export ECALLS_DATA_ROOT=$ROOT/data
export PYTHONPATH=$ROOT/src
export HF_HOME=$ROOT/.hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$ROOT/logs"
cd "$ROOT"

FIRST_YEAR=${FIRST_YEAR:-2010}
YEAR=$(( FIRST_YEAR + ${SLURM_ARRAY_TASK_ID:-0} ))
MODEL="chrono-bert-v1-${YEAR}1231"

echo "host $(hostname)  task ${SLURM_ARRAY_TASK_ID:-0}  vintage $YEAR  model $MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" -m ecalls.embed.encode --model "$MODEL" --compute-dtype float16 \
     --require-gpu --up-to-cutoff
