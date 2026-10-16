#!/bin/bash
#SBATCH --job-name=ecalls-chsmoke
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/chsmoke_%j.out
#SBATCH --error=logs/chsmoke_%j.out
set -euo pipefail

# Smoke test and throughput check for the chained segments (D29): the card, the stack, a write and read
# in the models directory, then 40 steps of segment 2011 in float16 under a throwaway name, whose log
# gives tokens per second. The chain and the twice-budget run are submitted with afterok on this job.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

hostname
nvidia-smi -L
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'capability', torch.cuda.get_device_capability(0))"
probe=$ECALLS_DATA_ROOT/processed/models/.smoke_chain_$SLURM_JOB_ID
echo ok > "$probe" && cat "$probe" && rm "$probe"

NAME=ecalls-chainprobe-20111231
rm -rf "$ECALLS_DATA_ROOT/processed/models/$NAME"
srun "$PY" -m ecalls.train.pretrain \
  --window ecalls-bpe-32768-20101231-ctx768-20241231 --cutoff 2011-12-31 \
  --init-from ecalls-mlm-20101231 --compute-dtype float16 --require-gpu \
  --max-steps 40 --log-every 10 --checkpoint-every 1000 --name "$NAME"
rm -rf "$ECALLS_DATA_ROOT/processed/models/$NAME"
echo "smoke and probe passed"
