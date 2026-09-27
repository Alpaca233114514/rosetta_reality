#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python - <<'PY'
import json, os, time
from pathlib import Path
start=time.time()
with Path('registration.json').open('x') as f:
    json.dump({'run':'execution-readiness-20260927-001','started_unix':start,
               'work_deadline_unix':start+180,'shutdown_deadline_unix':start+900,
               'purpose':'metadata admission audit, then stop on unmet design prerequisites',
               'optimizer_updates':0,'model_forward_limit':0,'dataset_row_read_limit':0,
               'automatic_training':False,'no_retry':True},f,indent=2)
    f.flush(); os.fsync(f.fileno())
PY
nohup python -u guard.py >guard.log 2>&1 </dev/null &
for attempt in {1..20}; do
    [[ -f guard-armed.json ]] && break
    sleep 0.1
done
test -f guard-armed.json
python - <<'PY'
import json, os, subprocess
from pathlib import Path
with Path('worker.log').open('x') as log:
    child=subprocess.Popen(['python','-u','audit.py'],stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
with Path('worker-pid.json').open('x') as f:
    json.dump({'pid':child.pid,'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().split()[21]},f)
    f.flush();os.fsync(f.fileno())
PY
cat registration.json guard-armed.json
