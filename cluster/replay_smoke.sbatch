#!/bin/bash
#SBATCH --job-name=ecalls-rplsmoke
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rplsmoke_%j.out
#SBATCH --error=logs/rplsmoke_%j.out
set -euo pipefail

# Smoke test and throughput check for the replay control (D31): the card, the stack, a write and read in
# the models directory, then 40 steps of the 2010 replay in float16 under a throwaway name, whose log
# gives tokens per second. replay_submit.sh makes the array wait for this job to succeed.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

hostname
nvidia-smi -L
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'capability', torch.cuda.get_device_capability(0))"
probe=$ECALLS_DATA_ROOT/processed/models/.smoke_rpl_$SLURM_JOB_ID
echo ok > "$probe" && cat "$probe" && rm "$probe"

NAME=ecalls-rplprobe-20101231
rm -rf "$ECALLS_DATA_ROOT/processed/models/$NAME"
srun "$PY" -m ecalls.train.pretrain \
  --window ecalls-bpe-32768-20101231-ctx768-20241231 --cutoff 2010-12-31 \
  --init-from ecalls-mlm-20101231 --replay-of ecalls-chain-mlm-20111231 \
  --compute-dtype float16 --require-gpu \
  --max-steps 40 --log-every 10 --checkpoint-every 1000 --name "$NAME"
rm -rf "$ECALLS_DATA_ROOT/processed/models/$NAME"
echo "smoke and probe passed"
