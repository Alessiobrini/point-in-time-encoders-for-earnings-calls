#!/bin/bash
#SBATCH --job-name=ecalls-pack
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pack_%j.out
#SBATCH --error=logs/pack_%j.out
set -euo pipefail

# One packed corpus at the latest vintage cutoff. Every earlier vintage reads a prefix of it, because the
# array is written shard by shard in calendar order and no sequence crosses a call. Packing the fifteen
# windows separately would store the same text 7.7 times over, 46 GB against 6 GB.
#
# A CPU job rather than a GPU one: this is tokenization and IO, measured at about 7,000 chunks per second,
# so roughly half an hour for the 12.1 M chunks up to 2024.

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
PY=${ECALLS_PYTHON:-python}
export ECALLS_DATA_ROOT=$ROOT/data PYTHONPATH=$ROOT/src TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"; cd "$ROOT"

echo "host $(hostname)  cutoff ${CUTOFF:-2024-12-31}"
srun "$PY" -m ecalls.train.pretrain_data --cutoff "${CUTOFF:-2024-12-31}"
