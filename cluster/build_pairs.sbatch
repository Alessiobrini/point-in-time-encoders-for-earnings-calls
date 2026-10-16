#!/bin/bash
#SBATCH --job-name=ecalls-pairs
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/pairs_%j.out
#SBATCH --error=logs/pairs_%j.out
set -euo pipefail

# One pair set at the latest vintage cutoff, for the same reason the packed corpus is one file: the set
# is written shard by shard in calendar order and carries the shard on every row, so a vintage's pairs
# are the rows whose shard ends on or before its cutoff. Building fifteen separately would re-extract
# the same question-answer structure 7.7 times over.
#
# Held-out calls are dropped here, so nothing downstream has to remember to do it. A CPU job: this is
# parquet scanning and string work, no model involved.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"; cd "$ROOT"

echo "host $(hostname)  cutoff ${CUTOFF:-2024-12-31}"
# HOLDOUT=1 builds the evaluation set instead of the training set. Both are needed: the training
# set excludes the held-out calls and the evaluation set is exactly those calls.
EXTRA=""
if [ "${HOLDOUT:-0}" = "1" ]; then EXTRA="--holdout-only"; fi
srun "$PY" -m ecalls.train.pairs --cutoff "${CUTOFF:-2024-12-31}" $EXTRA
