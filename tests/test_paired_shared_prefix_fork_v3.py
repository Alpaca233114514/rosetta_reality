"""Pure comparison regressions, not proof of live restore or teacher replay."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("fork_engineering", Path(__file__).resolve().parents[1]/"scripts/check_paired_shared_prefix_fork_v3.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row():
    return {"action":[.1,.2],"phase":"grasp","before":{"q":[1.]},"after":{"contact":True},
            "integration":[1.,2.],"reward":0.,"done":False,"info":{"is_success":False},"safe":True,
            "native_images":{"top":{"sha256":"different-native-render-is-disclosed-separately"}}}


def test_native_render_diff_does_not_claim_physical_diff():
    a,b = row(),row()
    b["native_images"]["top"]["sha256"]="other-render"
    assert module.compare_branches([a],[b])["physical_action_event_exact"] is True


@pytest.mark.parametrize("key", ["action","phase","before","after","integration","reward","done","info","safe"])
def test_every_physical_action_event_difference_is_rejected(key):
    a,b=row(),row()
    b[key]="corrupt"
    assert module.compare_branches([a],[b])["physical_action_event_exact"] is False


def test_length_mismatch_rejected():
    assert not module.compare_branches([row()],[]) ["physical_action_event_exact"]
