#!/bin/bash
#SBATCH --job-name=ecalls-pftext
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-13
#SBATCH --output=logs/pftext_%A_%a.out
#SBATCH --error=logs/pftext_%A_%a.out
set -euo pipefail

# The lexical baselines of the portfolio test (D26). Task 0 scores dictionary tone for every call
# under each refit year's point-in-time dictionary. Tasks 1 to 13 fit TF-IDF and its 32-component SVD
# on the training calls of test year 2012 + i and apply it to that year's training and test calls.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src
mkdir -p "$ROOT/logs"; cd "$ROOT"

I=${SLURM_ARRAY_TASK_ID:-0}
if [ "$I" -eq 0 ]; then
  "$PY" -m ecalls.portfolio.text_features tone
else
  "$PY" -m ecalls.portfolio.text_features tfidf --year $(( 2012 + I ))
fi
