#!/bin/bash
# Submit the chained vintages (D29): fourteen segments, 2011 to 2024, each depending on the previous one,
# and each segment's contrastive stage depending on that segment. Run on the cluster login node, from the repository root:
#
#     bash cluster/chain_submit.sh
#
# FIRST and LAST restrict the years, for example FIRST=2011 LAST=2011 for the first segment alone.
# AFTER makes the first submitted segment wait for that job to succeed, for example a smoke test.
# The first segment waits on nothing, so ecalls-mlm-20101231 must already be finished, which the trainer
# checks. After a break in the chain, run this again: a segment or retriever whose run record exists is
# not submitted, so only the missing work goes in, and the first missing segment waits on nothing.
#
# The 2010 member of the chained family is ecalls-ret-20101231, the retriever on the shared base, so no
# contrastive job is submitted for 2010.
set -euo pipefail

ROOT=${ECALLS_ROOT:?set ECALLS_ROOT to your clone of this repository}
MODELS=$ROOT/data/processed/models
FIRST=${FIRST:-2011}
LAST=${LAST:-2024}

prev=${AFTER:-}
for YEAR in $(seq "$FIRST" "$LAST"); do
  seg_name="ecalls-chain-mlm-${YEAR}1231"
  ret_name="ecalls-chain-ret-${YEAR}1231"
  if [ -f "$MODELS/$seg_name/_run.json" ]; then
    seg="done"
    prev=""
  else
    dep=""
    if [ -n "$prev" ]; then dep="--dependency=afterok:$prev"; fi
    seg=$(sbatch --parsable $dep --export=ALL,YEAR="$YEAR" "$ROOT/cluster/chain_segment.sbatch")
    prev=$seg
  fi
  if [ -f "$MODELS/$ret_name/_run.json" ]; then
    ret="done"
  else
    dep=""
    if [ "$seg" != "done" ]; then dep="--dependency=afterok:$seg"; fi
    # The contrastive script is an array over years. One task, with the year in FIRST_YEAR.
    ret=$(sbatch --parsable $dep --array=0 --job-name=ecalls-chain-ret \
          --export=ALL,FIRST_YEAR="$YEAR",MLM_TAG=chain-mlm,RET_TAG=chain-ret \
          "$ROOT/cluster/contrastive_vintages.sbatch")
  fi
  echo "$YEAR  segment $seg  $seg_name  retriever $ret  $ret_name"
done
