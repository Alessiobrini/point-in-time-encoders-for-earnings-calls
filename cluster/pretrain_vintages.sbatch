#!/bin/bash
#SBATCH --job-name=ecalls-mlm
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/mlm_%A_%a.out
#SBATCH --error=logs/mlm_%A_%a.out
set -euo pipefail

# Masked language modeling for the 15 annual vintages, one per array task. Every task reads a prefix of
# one packed corpus, since the array is in calendar order and a window is a prefix, and every task trains
# for the same token budget so only the data varies (section 5 item 2).
#
# Requeue-safe: the checkpoint carries the optimizer and the sampler state as well as the weights, so a
# preempted task resumes where it stopped rather than restarting and never finishing. Rerunning a finished
# task is a no-op past its step count.
#
# 24h rather than 8h, because the earlier vintage embedding array had two tasks killed by a limit sized
# from a differently shaped run.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/logs"; cd "$ROOT"

FIRST_YEAR=${FIRST_YEAR:-2010}
YEAR=$(( FIRST_YEAR + ${SLURM_ARRAY_TASK_ID:-0} ))
WINDOW=${WINDOW:-ecalls-bpe-32768-20101231-ctx768-20241231}
BUDGET=${BUDGET:-1200000000}
# The name tag. The 2x budget check (D29) reuses this script with BUDGET=2400000000 and TAG=mlm2x, so
# its directory can never be the finished 1x vintage's and resume into it.
TAG=${TAG:-mlm}
# The pretraining seed. A seed other than 0 must come with its own TAG, pseed<S>-mlm, so the run can
# never resume into the seed-0 deliverable of the same name.
SEED=${SEED:-0}

# No --batch-sequences: it is scaled from the allocated GPU's memory and accumulation keeps tokens per
# step fixed, so the gradient is identical on a 10 GB card and a 49 GB one. A fixed 32 killed twelve of
# fifteen tasks on 10.58 GiB cards that the throughput probe never saw.
echo "host $(hostname)  vintage $YEAR  budget $BUDGET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" -m ecalls.train.pretrain \
  --window "$WINDOW" --cutoff "${YEAR}-12-31" \
  --token-budget "$BUDGET" \
  --compute-dtype bfloat16 --require-gpu --seed "$SEED" \
  --checkpoint-every 250 --log-every 100 \
  --name "ecalls-${TAG}-${YEAR}1231"
