#!/usr/bin/env bash
set -euo pipefail
toolroot=/root/autodl-tmp/rosetta/workspaces/20260923T110402Z-2687d7a884d4-8684002da746/tools
output=/root/autodl-tmp/rosetta/runs/prepost-tool-overlay-20260923-002
py=/root/autodl-tmp/rosetta/envs/smolvla-cuda-001/bin/python
test ! -e "$output"
mkdir "$output"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
timeout 90 "$py" -m pip install --no-index --no-deps --require-hashes \
  --target "$output/overlay" --find-links "$toolroot/wheels" \
  -r "$toolroot/torchlens-diagnostics.txt" > "$output/install.log" 2>&1
export PYTHONPATH="$output/overlay:$toolroot/basin"
timeout 20 "$py" "$toolroot/check_tools.py" > "$output/check.json"
timeout 20 "$py" -m basin tools > "$output/basin-tools.json"
cat "$output/check.json"
sha256sum "$output/check.json" "$output/basin-tools.json"
