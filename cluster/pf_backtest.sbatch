#!/bin/bash
#SBATCH --job-name=ecalls-pfbt
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/pfbt_%j.out
#SBATCH --error=logs/pfbt_%j.out
set -euo pipefail

# The portfolio backtest (D26), after the embedding array and the text-feature array have finished.
# CPU only: ridge fits and daily portfolio sums over 28,365 calls.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src
mkdir -p "$ROOT/logs"; cd "$ROOT"
"$PY" -m ecalls.portfolio.backtest --target "${TARGET:-ret}" ${ADJACENT:+--adjacent} ${LEXICAL:+--lexical-only}
