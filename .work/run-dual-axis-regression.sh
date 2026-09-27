#!/usr/bin/env bash
set -euo pipefail
mkdir /tmp/dual-axis-checkout
for name in src scripts tests configs docs pyproject.toml; do
  cp -a "/source/$name" /tmp/dual-axis-checkout/
done
mkdir -p /tmp/dual-axis-checkout/runs/canonical-posttrain-received-20260916-005-ssh/verified/artifact-metadata
cp /source/.work/dual-axis-fixtures/config.json /tmp/dual-axis-checkout/runs/canonical-posttrain-received-20260916-005-ssh/verified/artifact-metadata/config.json
cd /tmp/dual-axis-checkout
export PYTHONPATH=/tmp/dual-axis-checkout/src:/tmp/dual-axis-checkout:/tmp/dual-axis-checkout/scripts:/tmp/dual-axis-checkout/tests:/basin
python scripts/check_env.py
python -m pytest -q -p no:cacheprovider \
  tests/test_dual_axis.py tests/test_dual_axis_runtime.py \
  tests/test_smolvla_training_features.py tests/test_smolvla_training_plan_schema.py \
  tests/test_trajectory_native_runtime.py tests/test_closed_loop_reproducibility.py \
  /basin/tests
