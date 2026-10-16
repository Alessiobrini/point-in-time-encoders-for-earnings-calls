#!/bin/bash
# Submit the replay control (D31): a smoke test, the 14 pretraining tasks once it succeeds, and the
# contrastive stage for each year once that year's pretraining task succeeds (aftercorr pairs task i
# with task i). Run on the cluster login node, from the repository root:
#
#     bash cluster/replay_submit.sh
set -euo pipefail

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
smoke=$(sbatch --parsable "$ROOT/cluster/replay_smoke.sbatch")
mlm=$(sbatch --parsable --dependency="afterok:$smoke" "$ROOT/cluster/replay_segment.sbatch")
ret=$(sbatch --parsable --dependency="aftercorr:$mlm" --partition=common --array=0-13 \
      --job-name=ecalls-rpl-ret \
      --export=ALL,FIRST_YEAR=2010,MLM_TAG=rpl-mlm,RET_TAG=rpl-ret \
      "$ROOT/cluster/contrastive_vintages.sbatch")
echo "smoke      $smoke"
echo "pretrain   $mlm  ecalls-rpl-mlm-<Y>1231, Y = 2010 to 2023"
echo "retriever  $ret  ecalls-rpl-ret-<Y>1231"
