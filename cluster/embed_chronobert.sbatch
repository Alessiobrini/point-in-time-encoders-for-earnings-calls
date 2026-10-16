#!/bin/bash
#SBATCH --job-name=ecalls-embed
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
#SBATCH --array=0-15
#SBATCH --output=logs/embed_%A_%a.out
#SBATCH --error=logs/embed_%A_%a.out
# Nodes whose GPU reports "Failed to get device handle for GPU 0" while still accepting jobs. They are
# not transiently busy, they are broken and they keep taking work: h36-24 swallowed 9 of 16 tasks in one
# array and had already killed a task in the previous one. --require-gpu makes each landing cost 20
# seconds instead of hours, and excluding them makes it cost nothing. Re-check this list rather than
# trusting it, since it is a snapshot of a shared cluster: the recipe is in docs/decisions.md, L9.

# Embed the chunk layer with one registered vintage. Every path is absolute, because a compute
# node's working directory is not the project root.
#
# Requeue-safe by construction: `encode.py` treats a part file as done only when it covers every
# chunk_id in its shard, so a preempted task re-runs the shard it was interrupted on and skips the
# ones it finished. That is the whole reason this can sit on `scavenger-gpu`.
#
# The array splits the 88 quarterly shards round-robin rather than in contiguous blocks, so the
# corpus's sixfold growth in annual volume does not load the last task with every recent quarter.

set -euo pipefail

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
MODEL=${MODEL:-chrono-bert-v1-20241231}
BATCH=${BATCH:-256}   # only used with --fixed-batch; the default budget comes from the GPU

export ECALLS_DATA_ROOT=$ROOT/data
export PYTHONPATH=$ROOT/src
export HF_HOME=$ROOT/.hf                 # the model cache stays on the group share, not in $HOME
export HF_HUB_OFFLINE=1                  # weights are pre-cached on the login node, because a
                                         # compute node's outbound network is not guaranteed and a
                                         # blocked download would hang the task rather than fail it
export TOKENIZERS_PARALLELISM=false
# The OOM reports showed 2.5 to 3.2 GB reserved but unallocated, so fragmentation was a real part of
# the failure rather than only the batch size. This is the allocator setting those reports asked for.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$ROOT/logs" "$HF_HOME"
cd "$ROOT"

# ARRAY_WIDTH is fixed rather than read from SLURM_ARRAY_TASK_COUNT, and it must match the
# --array range above. Resubmitting a subset such as --array=1,5,15 sets SLURM_ARRAY_TASK_COUNT=3,
# so `i % ntasks == task` became `i % 3 == 5`, which is never true: tasks 5 and 15 were handed no
# shards at all and then fell through to embedding all 88 concurrently. The width has to be a
# property of the plan, not of whichever subset is being retried.
ARRAY_WIDTH=${ARRAY_WIDTH:-16}
TASK=${SLURM_ARRAY_TASK_ID:-0}

if [ "$TASK" -ge "$ARRAY_WIDTH" ]; then
  echo "task id $TASK is outside ARRAY_WIDTH=$ARRAY_WIDTH, so its shard assignment is undefined" >&2
  exit 2
fi

# SHARDS can be set in the environment to run a named subset, which is how the throughput smoke test
# runs a few quarters before the full array is committed to a GPU allocation.
if [ -z "${SHARDS:-}" ]; then
  SHARDS=$("$PY" - "$TASK" "$ARRAY_WIDTH" <<'PYEOF'
import sys
from ecalls.embed.encode import shard_keys
task, ntasks = int(sys.argv[1]), int(sys.argv[2])
print(" ".join(s for i, s in enumerate(shard_keys()) if i % ntasks == task))
PYEOF
  )
fi

# An empty list must not reach the encoder: `--shards` with no values makes argparse return [],
# which is falsy, and the CLI then treats it as "every shard". That is how three retried tasks ended
# up writing the same part files at the same time.
if [ -z "${SHARDS// /}" ]; then
  echo "no shards assigned to task $TASK of $ARRAY_WIDTH; refusing to fall through to all shards" >&2
  exit 2
fi

echo "host $(hostname)  task $TASK/$ARRAY_WIDTH  model $MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "shards: $SHARDS"

# shellcheck disable=SC2086
# No --batch-size: `encode.py` scales a padded-character budget from the device it was allocated,
# because scavenger handed this array both 24 GB A5000s and 10.58 GB 2080s and a fixed 256 OOMed the
# latter. It also halves and retries any batch that still OOMs, so a shard cannot be lost to it.
# --require-gpu: this job asked for --gres=gpu:1, so a node where CUDA is invisible has a broken
# device and must fail fast. One task that fell back to CPU ran at 4 chunk/s against 265, and burned
# 1h44 of its allocation producing one shard out of six.
srun "$PY" -m ecalls.embed.encode --model "$MODEL" --compute-dtype float16 --require-gpu \
     --shards $SHARDS
