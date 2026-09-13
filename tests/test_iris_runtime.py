"""Reject identity drift and false reload claims before real CUDA use."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.iris_runtime import observations, sha, verify_checkpoint, verify_reload


def checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    stage = {"run_name": "iris-test", "steps": 2, "batch_size": 4, "episodes": [49, 4]}
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_7_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        (root / name).write_text("stub")
    (root / "train_config.json").write_text(
        json.dumps(
            {
                "job_name": stage["run_name"],
                "steps": 2,
                "batch_size": 4,
                "dataset": {"episodes": [49, 4]},
            }
        )
    )
    identities = {p.name: sha(p) for p in root.iterdir()}
    return root, identities, {"optimizer_smoke": stage}


def test_checkpoint_complete_seal(tmp_path):
    root, identities, plan = checkpoint(tmp_path)
    assert verify_checkpoint(root, identities, plan) == root


@pytest.mark.parametrize("mode", ["weight", "extra", "missing_processor", "run", "path"])
def test_checkpoint_rejects_drift(tmp_path, mode):
    root, identities, plan = checkpoint(tmp_path)
    if mode == "weight":
        (root / "model.safetensors").write_text("changed")
    elif mode == "extra":
        (root / "extra.txt").write_text("unsealed")
    elif mode == "missing_processor":
        del identities["policy_preprocessor.json"]
    elif mode == "run":
        plan["optimizer_smoke"]["run_name"] = "different"
    else:
        identities["../escaped"] = "0" * 64
    with pytest.raises(ValueError):
        verify_checkpoint(root, identities, plan)


def pair(tmp_path):
    keys = (
        "normalized_predictions",
        "standard_predictions",
        "normalized_targets",
        "standard_targets",
        "internal_grippers",
        "noise",
        "valid_mask",
    )
    paths = [tmp_path / "first.npz", tmp_path / "second.npz"]
    for pid, path in enumerate(paths, 101):
        np.savez(path, **{key: np.ones((2, 3)) for key in keys})
        meta = {
            "array_sha256": sha(path),
            "parameters_unchanged": True,
            "hidden_test_loaded": False,
            "processor_verification": "saved_checkpoint_state_no_statistics_override",
            "collector_pid": pid,
            "source_sha256": "1" * 64,
            "episodes": [49, 4],
            "image_sha256": ["2" * 64, "3" * 64],
            "noise_seeds": [None, 20260905, 20260906, 20260907],
        }
        Path(str(path) + ".json").write_text(json.dumps(meta))
    return paths


def test_reload_compares_all_arrays(tmp_path):
    report = verify_reload(*pair(tmp_path))
    assert report["scalar_checks"] == 42 and len(report["exact_arrays"]) == 7


@pytest.mark.parametrize(
    "key,value",
    [
        ("collector_pid", 101),
        ("source_sha256", "9" * 64),
        ("processor_verification", "registered_base_statistics"),
        ("array_sha256", "0" * 64),
    ],
)
def test_reload_rejects_false_identity(tmp_path, key, value):
    paths = pair(tmp_path)
    meta_path = Path(str(paths[1]) + ".json")
    meta = json.loads(meta_path.read_text())
    meta[key] = value
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        verify_reload(*paths)


def test_reload_rejects_late_chunk_change_even_with_correct_file_hash(tmp_path):
    paths = pair(tmp_path)
    with np.load(paths[1]) as data:
        values = {key: data[key] for key in data.files}
    values["normalized_predictions"][-1, -1] += 1
    np.savez(paths[1], **values)
    meta_path = Path(str(paths[1]) + ".json")
    meta = json.loads(meta_path.read_text())
    meta["array_sha256"] = sha(paths[1])
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="array differs"):
        verify_reload(*paths)


def test_observations_exclude_actions_and_keep_camera_padding():
    batch = dict.fromkeys(
        [
            "observation.state",
            "observation.language.tokens",
            "observation.language.attention_mask",
            "camera_padding_mask",
            "action",
        ],
        1,
    )
    actual = observations(batch)
    assert "action" not in actual and "camera_padding_mask" in actual
    del batch["observation.state"]
    with pytest.raises(ValueError):
        observations(batch)
