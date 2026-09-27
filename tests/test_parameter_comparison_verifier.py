import json
import os

import pytest
import torch
from safetensors.torch import save_file

from scripts.compare_smolvla_parameters import run, sha
from scripts.verify_parameter_comparison import verify


def test_independent_verifier_rejects_resealed_wrong_number(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    save_file({"x": torch.tensor([0., 2., 4.])}, a)
    save_file({"x": torch.tensor([1., 2., 6.])}, b)
    plan = {"id": "test", "scope": "saved_weight_algebra_only", "optimizer_steps": 0,
            "image": os.environ["ROSETTA_CONTAINER_IMAGE_ID"],
            "resources": {"memory_mib": 4096, "cpu_threads": 2, "wall_seconds": 600},
            "models": {"a": str(a), "b": str(b)}, "pairs": [["a", "b"]],
            "files": {str(a): sha(a), str(b): sha(b)}}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    out = tmp_path / "result"
    run(p, out)
    verify(p, out, tmp_path / "verified.json")
    row_path = out / "a-to-b.jsonl"
    row = json.loads(row_path.read_text())
    row["delta_l2"] *= 2
    row_path.write_text(json.dumps(row) + "\n")
    result_path = out / "result.json"
    result = json.loads(result_path.read_text())
    result["pairs"]["a-to-b"]["rows_sha256"] = sha(row_path)
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Independent arithmetic mismatch"):
        verify(p, out, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
    with pytest.raises(FileExistsError):
        run(p, out)


def test_hash_drift_preserves_failure(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({
        "scope": "saved_weight_algebra_only", "optimizer_steps": 0,
        "image": os.environ["ROSETTA_CONTAINER_IMAGE_ID"],
        "resources": {"memory_mib": 4096, "cpu_threads": 2, "wall_seconds": 600},
        "files": {str(p): "0" * 64}}))
    out = tmp_path / "failure"
    with pytest.raises(ValueError, match="hash mismatch"):
        run(p, out)
    assert json.loads((out / "failure.json").read_text())["status"] == "incomplete"
    assert not (out / "result.json").exists()
