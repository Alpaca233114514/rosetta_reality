#!/usr/bin/env bash
set -Eeuo pipefail
# Must run inside scripts/run_autodl.sh shell; the Python supervisor owns stages,
# independent watchdog, source seals, output caps, negative results and shutdown.
exec python -m scripts.run_canonical_fullframes_posttrain "$@"
