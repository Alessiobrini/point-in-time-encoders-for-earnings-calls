#!/bin/bash
# Submit the budget check (D29): the 2024 vintage pretrained from scratch at 2.4 B tokens, twice the
# standard budget, as ecalls-mlm2x-20241231, then its contrastive stage as ecalls-2x-ret-20241231. Both
# reuse the from-scratch scripts with one array task and the name tags set. Run on the cluster login node, from the repository root:
#
#     bash cluster/budget2x_submit.sh
#
# 3 days rather than the script's 24 hours. The slowest measured pretraining rate is 13.7 k tokens/s on a
# Turing card in bfloat16, so 2.4 B tokens can take 48.7 hours, and scavenger-gpu allows up to 7 days.
# The trainer resumes from its checkpoint, so a run killed at the limit continues when submitted again.
set -euo pipefail

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}

# AFTER makes the run wait for that job to succeed, for example a smoke test.
dep=""
if [ -n "${AFTER:-}" ]; then dep="--dependency=afterok:$AFTER"; fi
mlm=$(sbatch --parsable $dep --array=0 --time=3-00:00:00 --job-name=ecalls-mlm2x \
      --export=ALL,FIRST_YEAR=2024,BUDGET=2400000000,TAG=mlm2x \
      "$ROOT/cluster/pretrain_vintages.sbatch")
ret=$(sbatch --parsable --dependency="afterok:$mlm" --array=0 --job-name=ecalls-2x-ret \
      --export=ALL,FIRST_YEAR=2024,MLM_TAG=mlm2x,RET_TAG=2x-ret \
      "$ROOT/cluster/contrastive_vintages.sbatch")
echo "pretrain   $mlm  ecalls-mlm2x-20241231"
echo "retriever  $ret  ecalls-2x-ret-20241231"
