#!/bin/bash
#SBATCH --job-name=ecalls-knn
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/knn_%A_%a.out
#SBATCH --error=logs/knn_%A_%a.out
set -euo pipefail

# Nearest-neighbor forecasting (D30). Task i: target i of (ret1d, sue, av) on UNIVERSE, sp500 by
# default with the D26 call vectors. The full universe takes ret1d only, once its vectors exist.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$ROOT/logs"; cd "$ROOT"
TARGETS=(ret1d sue av)
"$PY" -m ecalls.portfolio.knn --target "${TARGETS[${SLURM_ARRAY_TASK_ID:-0}]}" --universe "${UNIVERSE:-sp500}"
