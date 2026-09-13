"""Exercise the actual admission main with synthetic files and no CUDA execution."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import diagnose_hestia_k_direction_head as fixed
from scripts.hestia_k_direction_head import CONDITIONS, component_mode, condition_spec


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch, module, condition="base640_native", *, fake_gpu=True):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "profile.yaml").write_text("synthetic profile")
    (tmp_path / "input.txt").write_text("synthetic input")
    (tmp_path / "upstream.py").write_text("# synthetic upstream")
    (tmp_path / "audit.json").write_text(
        json.dumps({"frozen_vlm_exact": True, "tensor_count": 500})
    )
    job = tmp_path / "job"
    job.mkdir()
    pid = os.getpid()
    ticks = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    (job / "active-child.json").write_text(
        json.dumps({"pid": pid, "ticks": ticks, "condition": condition})
    )
    roots = {}
    for step in (640, 1280):
        root = tmp_path / "checkpoints" / "fixture" / f"{step:06d}" / "pretrained_model"
        root.mkdir(parents=True)
        # Only byte identities are exercised; this is not a real safetensors file.
        (root / "model.safetensors").write_bytes(f"synthetic endpoint {step}".encode())
        roots[str(step)] = {"model.safetensors": sha(root / "model.safetensors")}
    for name in CONDITIONS[: CONDITIONS.index(condition)]:
        if name not in CONDITIONS[:3]:
            continue
        folder = job / name
        folder.mkdir()
        (folder / "arrays.npz").write_bytes(b"synthetic endpoint output")
        result = {
            "condition": name,
            "intervention_applied": component_mode(name) != "native",
            "array_sha256": sha(folder / "arrays.npz"),
            "same_device_control": {
                "passed": True,
                "normalized_predictions": {"max_abs": 0},
                "standard_predictions": {"max_abs": 0},
            },
        }
        (folder / "result.json").write_text(json.dumps(result))
    plan = {
        "id": "hestia-k-direction-head-20260913-001",
        "selected_layer": 1,
        "allowed_conditions": list(CONDITIONS),
        "output": "job",
        "profile_sha256": sha(tmp_path / "profile.yaml"),
        "sha256": {"input.txt": sha(tmp_path / "input.txt")},
        "upstream_files": {"upstream.py": sha(tmp_path / "upstream.py")},
        "checkpoint_prefix": "fixture",
        "checkpoints": roots,
        "parameter_audit": "audit.json",
        "reference_endpoints": {"640": "reference640.npz", "1280": "reference1280.npz"},
        "reference_direction": "direction640.npz",
        "reference_v_full": {"640": "full640.npz", "1280": "full1280.npz"},
        "historical_runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "lerobot": "synthetic",
            "device_name": "synthetic-card",
        },
    }
    path = tmp_path / "template.json"
    path.write_text(json.dumps(plan))
    now = time.time()
    permit = {
        "model_execution_authorized": True,
        "training_authorized": False,
        "shutdown_authorized": True,
        "template_sha256": sha(path),
        "watchdog_active": True,
        "watchdog_pid": pid,
        "watchdog_ticks": ticks,
        "allowed_conditions": list(CONDITIONS),
        "started_unix": now - 1,
        "deadline_unix": now + 100,
    }
    (tmp_path / "permit.json").write_text(json.dumps(permit))
    monkeypatch.setenv("ROSETTA_AUTODL_RUNTIME_PROFILE", str(tmp_path / "profile.yaml"))
    monkeypatch.setenv("ROSETTA_TORCH_DEVICE", "cuda")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("ROSETTA_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setattr(
        module.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(locate_file=lambda _: tmp_path),
    )
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "synthetic")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: fake_gpu)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1 if fake_gpu else 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic-card")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector",
            "--template",
            str(path),
            "--permit",
            str(tmp_path / "permit.json"),
            "--condition",
            condition,
        ],
    )
    calls = []
    monkeypatch.setattr(module, "collect", lambda p, out: calls.append((p, out)))
    return calls, plan


@pytest.mark.parametrize("condition", CONDITIONS)
def test_fixed_cli_checks_real_files_and_selects_registered_arm(tmp_path, monkeypatch, condition):
    calls, _ = fixture(tmp_path, monkeypatch, fixed, condition)
    fixed.main()
    assert len(calls) == 1
    plan, out = calls[0]
    base, donor = condition_spec(condition)
    assert (plan["base_step"], plan["donor_step"]) == (base, donor)
    assert Path(plan["checkpoint"]).name == "pretrained_model"
    assert str(base).zfill(6) in plan["checkpoint"]
    assert (plan["donor_checkpoint"] is None) == (donor is None)
    if donor is not None:
        assert str(donor).zfill(6) in plan["donor_checkpoint"]
    assert out.name == condition and out.is_dir()
    expected_reference = (
        (
            "direction640.npz"
            if component_mode(condition) == "kdirection"
            else f"reference{base}.npz"
        )
        if component_mode(condition) in {"native", "kdirection"} or condition.endswith("restore")
        else None
    )
    assert plan["reference_arrays"] == expected_reference


@pytest.mark.parametrize(
    "which,expected",
    [
        ("profile.yaml", "Registered CUDA profile"),
        ("input.txt", "Input/source drift"),
        ("upstream.py", "Upstream changed"),
    ],
)
def test_tampering_still_blocks_before_collector(tmp_path, monkeypatch, which, expected):
    calls, _ = fixture(tmp_path, monkeypatch, fixed)
    (tmp_path / which).write_text("changed")
    with pytest.raises(ValueError, match=expected):
        fixed.main()
    assert not calls


def test_cpu_mode_does_not_silently_run_collector(tmp_path, monkeypatch):
    calls, _ = fixture(tmp_path, monkeypatch, fixed, fake_gpu=False)
    with pytest.raises(ValueError, match="Original GPU model required"):
        fixed.main()
    assert not calls


def test_unregistered_parent_layer_stops_before_model(tmp_path, monkeypatch):
    calls, plan = fixture(tmp_path, monkeypatch, fixed)
    plan["selected_layer"] = 3
    (tmp_path / "template.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="parent layer"):
        fixed.main()
    assert not calls


def test_restore_positive_control_failure_blocks_means(tmp_path, monkeypatch):
    calls, _ = fixture(tmp_path, monkeypatch, fixed, "base640_kh0")
    path = tmp_path / "job/base640_krestore/result.json"
    result = json.loads(path.read_text())
    result["same_device_control"]["normalized_predictions"]["max_abs"] = 1e-8
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="bitwise exact"):
        fixed.main()
    assert not calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_authorized", True),
        ("model_execution_authorized", False),
        ("shutdown_authorized", False),
        ("watchdog_active", False),
        ("deadline_unix", 0),
        ("allowed_conditions", []),
    ],
)
def test_invalid_permit_cannot_reach_model(tmp_path, monkeypatch, field, value):
    calls, _ = fixture(tmp_path, monkeypatch, fixed)
    path = tmp_path / "permit.json"
    permit = json.loads(path.read_text())
    permit[field] = value
    path.write_text(json.dumps(permit))
    with pytest.raises(ValueError):
        fixed.main()
    assert not calls


def test_modified_control_arrays_stop_mean_before_model(tmp_path, monkeypatch):
    calls, _ = fixture(tmp_path, monkeypatch, fixed, "base640_kh0")
    (tmp_path / "job/base640_kdirection/arrays.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Control arrays changed"):
        fixed.main()
    assert not calls
