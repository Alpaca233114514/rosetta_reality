"""Linux subprocess regression: TERM leader exit must not preserve its child."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import q0_bounded_guard as guard


@unittest.skipUnless(sys.platform == "linux", "Linux process-group identity required")
class OwnedGroupCleanup(unittest.TestCase):
    def test_term_ignoring_child_dies_after_leader_exits(self):
        code = """import os, signal, sys, time
child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open(sys.argv[1], 'x') as f:
        f.write(str(os.getpid()))
    while True:
        time.sleep(1)
while True:
    time.sleep(1)
"""
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            ready = job / "child-pid"
            leader = subprocess.Popen([sys.executable, "-c", code, str(ready)], start_new_session=True,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 5
                while (not ready.exists() or not ready.read_text().strip()) and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.exists())
                child_pid = int(ready.read_text())
                identity = guard.original.process_identity(leader.pid)
                (job / "worker-pid.json").write_text(json.dumps({
                    "pid": leader.pid, "start_ticks": identity[0]}))
                result = guard.stop_worker(job, grace_seconds=.1)
                leader.wait(timeout=3)
                self.assertEqual(result, "owned_group_terminated")
                identity = guard.original.process_identity(child_pid)
                if identity is not None:
                    state = Path(f"/proc/{child_pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                    self.assertEqual(state, "Z")
            finally:
                try:
                    os.killpg(leader.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                leader.wait(timeout=3)

    def test_unverifiable_live_group_blocks_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            leader = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                      start_new_session=True)
            try:
                (job / "worker-pid.json").write_text(json.dumps({"pid": leader.pid, "start_ticks": "invalid"}))
                with self.assertRaises(RuntimeError):
                    guard.stop_worker(job, grace_seconds=0)
                self.assertTrue((job / "q0-worker-cleanup-blocked.json").exists())
                with self.assertRaises(RuntimeError):
                    guard.request_shutdown(job)
            finally:
                os.killpg(leader.pid, signal.SIGKILL)
                leader.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
