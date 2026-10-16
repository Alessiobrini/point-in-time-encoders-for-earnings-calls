#!/usr/bin/env bash
# Staged, idempotent runner for the whole corpus pipeline.
#
#   scripts/run_all.sh --list                  # the stage list
#   scripts/run_all.sh --stage universe        # one stage
#   scripts/run_all.sh --full                  # everything, skipping what exists
#   FORCE=1 scripts/run_all.sh --stage links   # rerun a stage whose output exists
#
# Every stage is idempotent on its own output, so a rerun costs nothing and a crash resumes.
# The WRDS stages run on a machine with WRDS access. The cluster is for training and embedding.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${ECALLS_PYTHON:-python}
cd "$ROOT"

ONLY=""; FULL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)  grep -oE '^if stage [a-z0-9_]+' "$0" | awk '{print "  "$3}'; exit 0 ;;
    --stage) ONLY="${2:-}"; shift ;;
    --full)  FULL=1 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -z "$ONLY" && "$FULL" -eq 0 ]] && { echo "pass --full, --stage <name>, or --list" >&2; exit 2; }

say()  { printf '\n=== %s ===\n' "$1"; }
skip() { printf '    skip (exists): %s\n' "$1"; }

# stage <name> <output-path> -- returns 0 when the stage should run
stage() {
  local name="$1" out="${2:-}"
  [[ -n "$ONLY" && "$ONLY" != "$name" ]] && return 1
  say "$name"
  if [[ -n "$out" && -e "$out" && "${FORCE:-0}" != "1" ]]; then skip "$out"; return 1; fi
  return 0
}

if stage probe        data/raw/wrds_access_probe.json;             then $PY -m ecalls.ingest.check_wrds_access; fi
if stage verify       "";                                         then $PY -m ecalls.ingest.verify_measurements; fi
if stage universe     data/raw/universe/sp500_membership.parquet;  then $PY -m ecalls.ingest.universe_sp500; fi
if stage links        data/processed/sp500_universe.parquet;       then $PY -m ecalls.ingest.link_chain; fi
if stage index        "";                                          then $PY -m ecalls.ingest.transcript_index; fi
if stage lengths      "";                                          then $PY -m ecalls.ingest.transcript_lengths; fi
if stage versions     data/processed/version_choice.parquet;       then $PY -m ecalls.ingest.select_versions; fi
if stage components   "";                                          then $PY -m ecalls.ingest.pull_components; fi
if stage consolidate  data/processed/components.parquet;           then $PY -m ecalls.ingest.consolidate; fi
if stage audit        "";                                          then $PY -m ecalls.audit.completeness; fi
if stage coverage     "";                                          then $PY -m ecalls.audit.coverage_report; fi

printf '\ndone.\n'
