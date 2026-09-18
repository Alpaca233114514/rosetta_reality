import json
from types import SimpleNamespace

import pytest

from scripts import run_hestia_parameter_crossover as control
from scripts.diagnose_hestia_parameter_crossover import endpoint_controls, validate_permit
from scripts.hestia_parameter_crossover import CONDITIONS


def test_missing_authorization_creates_nothing(tmp_path):
    args = SimpleNamespace(execute_authorized=False, shutdown_authorized=True)
    with pytest.raises(ValueError, match="authorization"):
        control.supervise(args, {"output": str(tmp_path / "job")})
    assert not (tmp_path / "job").exists()


def test_stale_pid_never_signals(tmp_path, monkeypatch):
    (tmp_path / "active-child.json").write_text(json.dumps({"pid": 123, "ticks": "old"}))
    monkeypatch.setattr(control, "alive", lambda *_: False)

    def forbidden(*_):
        raise AssertionError("Unrelated PID signaled")

    monkeypatch.setattr(control.os, "killpg", forbidden)
    control.stop_child(tmp_path)


def test_escaping_output_rejected(monkeypatch):
    monkeypatch.setenv("ROSETTA_AUTODL_RUNTIME_PROFILE", "fixture")
    monkeypatch.setenv("ROSETTA_TORCH_DEVICE", "cuda")
    args = SimpleNamespace(execute_authorized=True, shutdown_authorized=True)
    with pytest.raises(ValueError, match="within the fresh workspace"):
        control.supervise(args, {"output": "../escape"})


def permit():
    return {
        "model_execution_authorized": True,
        "training_authorized": False,
        "shutdown_authorized": True,
        "template_sha256": "bound",
        "watchdog_active": True,
        "allowed_conditions": list(CONDITIONS),
        "started_unix": 100,
        "deadline_unix": 1000,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"training_authorized": True},
        {"watchdog_active": False},
        {"allowed_conditions": list(reversed(CONDITIONS))},
        {"deadline_unix": 1001},
        {"template_sha256": "wrong"},
        {"shutdown_authorized": False},
    ],
)
def test_permit_drift_fails(change):
    p = permit()
    validate_permit("bound", p, "base1280", 200)
    p.update(change)
    with pytest.raises(ValueError):
        validate_permit("bound", p, "base1280", 200)


def test_expiry_and_unknown_arm_fail():
    for at in (99, 1000):
        with pytest.raises(ValueError):
            validate_permit("bound", permit(), "base1280", at)
    with pytest.raises(ValueError):
        validate_permit("bound", permit(), "unregistered", 200)


def test_both_exact_endpoints_required(tmp_path):
    endpoint_controls(tmp_path, "base1280")
    with pytest.raises(FileNotFoundError):
        endpoint_controls(tmp_path, "base1280_kv640")
    (tmp_path / "base1280").mkdir()
    (tmp_path / "base1280/result.json").write_text(
        json.dumps(
            {
                "condition": "base1280",
                "intervention_applied": False,
                "same_device_control": {"passed": False},
            }
        )
    )
    with pytest.raises(ValueError, match="prerequisite"):
        endpoint_controls(tmp_path, "base640")


def test_complete_controls_pass_and_later_tampering_fails(tmp_path):
    import hashlib

    for name in ("base1280", "base640"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "arrays.npz").write_bytes(b"fixture")
        result = {
            "condition": name,
            "intervention_applied": False,
            "same_device_control": {
                "passed": True,
                "normalized_predictions": {"max_abs": 0},
                "standard_predictions": {"max_abs": 0},
            },
            "array_sha256": hashlib.sha256(b"fixture").hexdigest(),
        }
        (folder / "result.json").write_text(json.dumps(result))
    endpoint_controls(tmp_path, "base1280_kv640")
    endpoint_controls(tmp_path, "base640_kv1280")
    (tmp_path / "base640/arrays.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        endpoint_controls(tmp_path, "base1280_kv640")


def test_file_backed_patch_proves_untouched_parameters(tmp_path):
    import torch
    from safetensors.torch import save_file

    from scripts.hestia_parameter_crossover import kv_keys, patch_policy

    params = {k: torch.zeros(320, 320) for k in kv_keys()}
    params["untouched"] = torch.arange(3, dtype=torch.float32)
    policy = SimpleNamespace(named_parameters=lambda: params.items())
    path = tmp_path / "donor.safetensors"
    save_file({k: torch.ones(320, 320) for k in kv_keys()}, path)
    r = patch_policy(policy, path)
    assert set(r["changed_keys"]) == set(kv_keys())
    assert r["before"]["untouched"] == r["after"]["untouched"]
    assert r["other_parameters_exact"] and r["donor_copy_exact"]
