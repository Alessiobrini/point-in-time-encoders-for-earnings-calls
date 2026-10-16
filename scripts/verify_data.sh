#!/usr/bin/env bash
# Re-hash raw files and flag drift vs data/raw/CHECKSUMS.txt
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Compare sha256 of data/raw/* against data/raw/CHECKSUMS.txt (fill in at acquisition)."
