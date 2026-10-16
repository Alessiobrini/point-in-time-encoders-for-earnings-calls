#!/bin/bash
#SBATCH --job-name=dl-chronogpt
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/dl_chronogpt_%j.out
# Downloads ChronoGPT-Instruct 2010 to 2023 and the GPT-2 tokenizer into $ECALLS_LLM_CACHE for D32.
set -euo pipefail
ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
"$PY" "$ROOT/scripts/dl_llm.py"
