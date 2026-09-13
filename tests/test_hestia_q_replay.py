"""Meaningful replay arithmetic, ordering and admission counterexamples."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.hestia_q_replay import CONDITIONS, MODES, Replay, attention_parts, validate_protocol


def original_eager():
    path = Path("runs/smolvla-visual-repair-20260909-002/upstream/smolvlm_with_expert.py")
    if not path.exists():
        import importlib.metadata

        path = Path(
            importlib.metadata.distribution("lerobot").locate_file(
                "lerobot/policies/smolvla/smolvlm_with_expert.py"
            )
        )
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "eager_attention_forward"
    )
    scope = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope["eager_attention_forward"]


@pytest.mark.parametrize("heads,kv", [(4, 2), (2, 2), (6, 1)])
@pytest.mark.parametrize("masked", [False, True])
def test_eager_reproduction_and_probability_replay(heads, kv, masked):
    core = SimpleNamespace(num_attention_heads=heads, num_key_value_heads=kv)
    generator = torch.Generator().manual_seed(93)
    q = torch.randn(1, 7, heads, 8, generator=generator)
    k = torch.randn(1, 11, kv, 8, generator=generator)
    v = torch.randn(1, 11, kv, 8, generator=generator)
    mask = torch.ones(1, 7, 11, dtype=torch.bool)
    if masked:
        mask[:, :, 3:6] = False
    expected = original_eager()(core, mask, 1, 8, q, k, v)
    output, p = attention_parts(core, mask, 1, 8, q, k, v)
    assert torch.equal(output, expected)
    if masked:
        assert (p[:, :, :, 3:6] == 0).all()
    blocked, _ = attention_parts(core, mask, 1, 8, q * 5, k * -3, v, p)
    assert torch.equal(blocked, expected)
    changed, _ = attention_parts(core, mask, 1, 8, q, k * -3, v)
    assert not torch.equal(changed, expected)
    with pytest.raises(ValueError, match="layout"):
        attention_parts(core, mask, 1, 8, q, k, v, p.double())


def minimal_replay():
    obj = Replay.__new__(Replay)
    obj.refs = {}
    obj.completed = {mode: 0 for mode in MODES}
    return obj


def test_replay_requires_same_sample_and_complete_previous_control():
    obj = minimal_replay()
    obj.begin(0, 0, "native")
    with pytest.raises(ValueError, match="incomplete"):
        obj.begin(0, 0, "qself")
    obj.refs = {i: None for i in range(80)}
    with pytest.raises(ValueError, match="control"):
        obj.begin(0, 0, "qself")
    obj.completed["native"] = 1
    obj.begin(0, 0, "qself")
    with pytest.raises(ValueError, match="sample"):
        obj.begin(1, 0, "qself")
    with pytest.raises(ValueError, match="sample"):
        obj.begin(0, 1, "qself")
    with pytest.raises(ValueError, match="Unregistered"):
        obj.begin(45, 0, "native")
    with pytest.raises(ValueError, match="count"):
        obj.end()


def test_cross_context_restores_after_failure():
    obj = minimal_replay()
    obj.layer = None

    def fail(*args):
        assert obj.layer == 3
        raise RuntimeError("injected")

    obj.original_cross = fail
    with pytest.raises(RuntimeError):
        obj.cross(None, None, 3)
    assert obj.layer is None


def test_protocol_admission_rejects_scope_drift():
    plan = {
        "id": "hestia-q-replay-20260913-001",
        "launchable": True,
        "conditions": list(CONDITIONS),
        "modes": list(MODES),
        "maximum_policy_forwards": 2160,
        "optimizer_steps": 0,
        "full_prefix_shape": [1, 241, 320],
        "replace_token_range": [0, 64],
        "work_deadline_seconds": 4200,
        "shutdown_deadline_seconds": 4800,
        "dev_calibration": "train40_only",
        "train_calibration": "train39_excluding_self",
    }
    validate_protocol(plan)
    for key, value in (
        ("launchable", False),
        ("optimizer_steps", 1),
        ("maximum_policy_forwards", 2161),
        ("replace_token_range", [0, 192]),
        ("work_deadline_seconds", 4201),
    ):
        bad = copy.deepcopy(plan)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_protocol(bad)


def test_expired_or_changed_permit():
    from scripts.diagnose_hestia_q_replay import validate_permit

    value = {
        "model_execution_authorized": True,
        "training_authorized": False,
        "shutdown_authorized": True,
        "template_sha256": "a",
        "watchdog_active": True,
        "allowed_conditions": list(CONDITIONS),
        "started_unix": 100,
        "deadline_unix": 4300,
    }
    validate_permit("a", value, "base640", 200)
    for changed, now in (
        ({"training_authorized": True}, 200),
        ({}, 4300),
        ({"allowed_conditions": ["base640"]}, 200),
        ({"deadline_unix": 4301}, 200),
    ):
        with pytest.raises(ValueError):
            validate_permit("a", {**value, **changed}, "base640", now)


def test_transfer_rejects_unregistered_files_and_wrong_size():
    from scripts.verify_hestia_q_replay_results import validate_manifest

    files = {
        name: {"bytes": 1, "sha256": "a" * 64}
        for name in ("registration.json", "permit.json", "worker-exited.json")
    }
    validate_manifest({"files": files, "total_bytes": 3})
    for name in (
        "../secret",
        "base640/model.safetensors",
        "base640/sentinel-41-0-1.npz",
        "base640/sentinel-0-10-1.npz",
        "base640/sentinel-0-0-2.npz",
    ):
        with pytest.raises(ValueError):
            validate_manifest(
                {"files": {**files, name: {"bytes": 1, "sha256": "a" * 64}}, "total_bytes": 4}
            )
    with pytest.raises(ValueError):
        validate_manifest({"files": files, "total_bytes": 4})


def test_independent_sentinel_arithmetic(tmp_path):
    import numpy as np

    from scripts.verify_hestia_q_replay_analysis import verify_sentinels

    core = SimpleNamespace(num_attention_heads=2, num_key_value_heads=1)
    rng = torch.Generator().manual_seed(42)
    q = torch.randn(1, 3, 2, 4, generator=rng).bfloat16()
    k = torch.randn(1, 5, 1, 4, generator=rng).bfloat16()
    v = torch.randn(1, 5, 1, 4, generator=rng).bfloat16()
    mask = torch.ones(1, 3, 5, dtype=torch.bool)
    mask[:, :, -1] = False
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, p = attention_parts(core, mask, 1, 4, q, k, v)
    values = {
        n: t.float().numpy()
        for n, t in {"q": q, "k": k, "v": v, "output": output, "probs": p}.items()
    }
    values["mask"] = mask.numpy()
    original = tmp_path / "sentinel.npz"
    np.savez(original, **values)
    for base in (640, 1280):
        folder = tmp_path / f"base{base}"
        folder.mkdir()
        for row in (0, 40):
            for step in range(10):
                for layer in range(1, 16, 2):
                    (folder / f"sentinel-{row}-{step}-{layer}.npz").write_bytes(
                        original.read_bytes()
                    )
    assert verify_sentinels(tmp_path)["sentinel_files"] == 320
    values["output"][0, 0, 0] += 2
    np.savez(tmp_path / "base640/sentinel-0-0-1.npz", **values)
    with pytest.raises(ValueError, match="ULP"):
        verify_sentinels(tmp_path)


def test_known_action_effects_and_independent_recomputation(tmp_path):
    import json

    import numpy as np

    from scripts.analyze_hestia_q_replay import analyze
    from scripts.hestia_scene_kv import file_hash
    from scripts.verify_hestia_q_replay_analysis import verify

    historical = tmp_path / "historical"
    historical.mkdir()
    dims = [
        {"name": f"{side}_{kind}", "unit": unit}
        for side in ("left", "right")
        for kind, unit in (("joint", "radian"), ("gripper", "normalized"))
    ]
    (historical / "manifest.json").write_text(json.dumps({"metadata": {"dimensions": dims}}))
    rng = np.random.default_rng(7)
    y = rng.normal(size=(45, 50, 4))
    for base in (640, 1280):
        folder = tmp_path / f"base{base}"
        folder.mkdir()
        values = {f"{s}_targets": y for s in ("normalized", "standard")}
        for mode in MODES:
            offset = 0.1 if mode == "kmean" else 0.05 if mode == "kmean_qnative" else 0.2
            for space in ("normalized", "standard"):
                values[f"{mode}_{space}_predictions"] = np.broadcast_to(y + offset, (4, *y.shape))
        np.savez(folder / "arrays.npz", **values)
        (folder / "result.json").write_text(
            json.dumps(
                {
                    "model_forwards": 1080,
                    "exact_control_forwards": 900,
                    "all_parameters_unchanged": True,
                    "optimizer_steps": 0,
                    "array_sha256": file_hash(folder / "arrays.npz"),
                }
            )
        )
        with (folder / "attention.jsonl").open("w") as stream:
            for noise in range(4):
                for row in range(45):
                    for mode in MODES:
                        for step in range(10):
                            for layer in range(1, 16, 2):
                                e = {
                                    "row": row,
                                    "noise": noise,
                                    "mode": mode,
                                    "step": step,
                                    "layer": layer,
                                    "q": "q",
                                    "v": "v",
                                    "mask": "mask",
                                    "k": "km" if mode.startswith("kmean") else "k",
                                    "output": "native"
                                    if mode in ("native", "qself", "aself", "kmean_ablock")
                                    else mode,
                                    "probs": "native"
                                    if mode in ("native", "qself", "aself", "kmean_ablock")
                                    else mode,
                                }
                                stream.write(json.dumps(e) + "\n")
    result = analyze(tmp_path, historical)
    assert result["primary"]["640"]["fixed_q_consistent_improvement"]
    assert result["primary"]["640"]["feedback_consistently_offsets"]
    report = verify(tmp_path, result)
    assert report["attention_event_checks"] == 172800
    broken = copy.deepcopy(result)
    key = "640/standard/dev5/full/left_joint/mae"
    broken["metrics"][key]["by_noise"]["kmean_qnative"][0] += 0.1
    with pytest.raises(ValueError, match="numeric"):
        verify(tmp_path, broken)
