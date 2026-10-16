#!/bin/bash
#SBATCH --job-name=ecalls-ridge
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ridge_grid_%j.out
#SBATCH --error=logs/ridge_grid_%j.out
set -euo pipefail
# The ridge-penalty check of the code audit (ecalls.diagnostics.ridge_grid). CPU only.
ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
"$PY" -m ecalls.diagnostics.ridge_grid ${TARGETS:+--targets $TARGETS}
