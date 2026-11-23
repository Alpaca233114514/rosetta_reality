import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def load(name, relative):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shorter_deadline_stops_only_owned_worker(tmp_path, monkeypatch):
    helper = load('tight_deadline_test', 'scripts/tighten_round_deadline.py')
    core = load('original_deadline_test', 'scripts/run_round_experiment_job.py')
    owned = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        owner = core.bind(owned)
        job = tmp_path / 'job'; job.mkdir()
        output = tmp_path / 'guard'; output.mkdir()
        core.save(job / 'worker-pid.json', owner)
        start = time.time() - 7199.8
        reg = {'started_unix': start, 'deadline_unix': start + 23000}
        core.save(output / 'deadline-registration.json', {'original_registration_sha256': 'test',
            'worker': owner, 'deadline_unix': start + 7200})
        def cleanup_stage(_job, _actor):
            assert not core.matches(owner)
            return {'status': 'owned_group_empty'}
        facade = SimpleNamespace(read=core.read, save=core.save, process=core.process,
            matches=core.matches, stop_group=core.stop_group, cleanup_stage=cleanup_stage)
        monkeypatch.setattr(helper, 'load_core', lambda _job: (facade, reg, 'test'))
        helper.run(job, output)
        owned.wait(timeout=2)
        assert owned.returncode != 0
        assert unrelated.poll() is None
        receipt = json.loads((output / 'exited.json').read_text())
        assert receipt['status'] == 'work_deadline_reached'
        assert receipt['shutdown_requested'] is False
    finally:
        for child in (owned, unrelated):
            if child.poll() is None: child.terminate()
            child.wait(timeout=3)
