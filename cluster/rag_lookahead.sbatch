#!/bin/bash
#SBATCH --job-name=ecalls-rag
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-13
#SBATCH --output=logs/rag_%A_%a.out
#SBATCH --error=logs/rag_%A_%a.out
set -euo pipefail

# D32: look-ahead through the retriever inside an LLM pipeline. Task J scores quarter (Y+1)Q2 with
# Y = 2010 + J: top-3 cross-call excerpts from the point-in-time and the 2024 vintage of the chained and
# from-scratch families, read by ChronoGPT-Instruct of year Y, scored by the log-probability of the real
# answer. MODE=smoke prints the environment, checks the output directory, and scores 40 questions of
# 2011Q2 to measure throughput on the card it drew.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
export ECALLS_LLM_CACHE=${ECALLS_LLM_CACHE:-$ROOT/.llm-cache}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$ROOT/logs"; cd "$ROOT"

echo "host $(hostname)  task ${SLURM_ARRAY_TASK_ID:-none}  mode ${MODE:-run}"
nvidia-smi -L
if [ "${MODE:-run}" = "smoke" ]; then
  $PY -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'transformers', transformers.__version__, 'gpu ok', torch.cuda.is_available())"
  d=$ECALLS_DATA_ROOT/processed/rag_lookahead; mkdir -p "$d"; echo ok > "$d/.write_check"; cat "$d/.write_check"; rm "$d/.write_check"
  $PY -m ecalls.diagnostics.rag_lookahead score --year 2010 --limit 40
  exit 0
fi
Y=$(( 2010 + ${SLURM_ARRAY_TASK_ID:-0} ))
$PY -m ecalls.diagnostics.rag_lookahead score --year "$Y" --skip-existing
