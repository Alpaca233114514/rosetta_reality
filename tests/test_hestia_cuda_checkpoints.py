import pytest

from scripts.diagnose_hestia_cuda_checkpoints import validate_permit


def valid():
    return {
        "model_execution_authorized": True,
        "training_authorized": False,
        "shutdown_authorized": True,
        "template_sha256": "fixture",
        "watchdog_active": True,
        "started_unix": 100,
        "deadline_unix": 700,
        "allowed_steps": [1280, 320, 640, 960],
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_execution_authorized", False),
        ("training_authorized", True),
        ("shutdown_authorized", False),
        ("watchdog_active", False),
        ("template_sha256", "wrong"),
    ],
)
def test_unsealed_or_widened_permit_stops(key, value):
    permit = valid()
    permit[key] = value
    with pytest.raises(ValueError):
        validate_permit("fixture", permit, 1280, 110)


@pytest.mark.parametrize("now", [99, 700, 701])
def test_outside_shared_deadline_stops(now):
    with pytest.raises(ValueError):
        validate_permit("fixture", valid(), 1280, now)


def test_unregistered_step_and_extended_budget_stop():
    with pytest.raises(ValueError):
        validate_permit("fixture", valid(), 256, 110)
    permit = valid()
    permit["deadline_unix"] = 701
    with pytest.raises(ValueError):
        validate_permit("fixture", permit, 1280, 110)
