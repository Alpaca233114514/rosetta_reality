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

from scripts import diagnose_hestia_kv_split as fixed
from scripts import diagnose_hestia_parameter_crossover as old
from scripts.hestia_kv_split import CONDITIONS, condition_spec


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch, module, condition="base1280", *, fake_gpu=True):
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
        if name not in ("base1280", "base640"):
            continue
        folder = job / name
        folder.mkdir()
        (folder / "arrays.npz").write_bytes(b"synthetic endpoint output")
        result = {
            "condition": name,
            "intervention_applied": False,
            "array_sha256": sha(folder / "arrays.npz"),
            "same_device_control": {
                "passed": True,
                "normalized_predictions": {"max_abs": 0},
                "standard_predictions": {"max_abs": 0},
            },
        }
        (folder / "result.json").write_text(json.dumps(result))
    plan = {
        "id": "hestia-parameter-crossover-20260912-001"
        if module is old
        else "hestia-kv-split-20260912-001",
        "output": "job",
        "profile_sha256": sha(tmp_path / "profile.yaml"),
        "sha256": {"input.txt": sha(tmp_path / "input.txt")},
        "upstream_files": {"upstream.py": sha(tmp_path / "upstream.py")},
        "checkpoint_prefix": "fixture",
        "checkpoints": roots,
        "parameter_audit": "audit.json",
        "reference_endpoints": {"640": "reference640.npz", "1280": "reference1280.npz"},
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
