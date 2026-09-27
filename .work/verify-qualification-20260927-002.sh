#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/c/Users/Logan/Documents/GitHub/rosetta_reality
image=$(tr -d '\r\n' < .cache/qualification-stdlib-image-20260927-001/image-id.txt)
exec bash scripts/run_qualification_local_tests.sh .cache/qualification-tests-20260927-002 "$image"
