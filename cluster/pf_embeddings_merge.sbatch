#!/bin/bash
#SBATCH --job-name=ecalls-pfmerge
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pfmerge_%j.out
#SBATCH --error=logs/pfmerge_%j.out
set -uo pipefail

# Join the 4 parts of each of the 39 D28 embedding models (pf_embeddings_parts.sbatch). A merge
# refuses unless every part is complete and the parts together hold exactly the model's universe.
# Every model is tried, and the job fails if any one could not be merged.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1
cd "$ROOT"

FAIL=0
for FAM in ${FAMILIES:-ecalls-ret ecalls-abl chrono-bert-v1}; do
  for V in $(seq 2012 2024); do
    "$PY" scripts/pf_embeddings.py --model "${FAM}-${V}1231" --through "$(( V + 1 ))-12-31" \
      --universe all --merge 4 || { echo "MERGE FAILED ${FAM}-${V}1231"; FAIL=1; }
  done
done
exit $FAIL
