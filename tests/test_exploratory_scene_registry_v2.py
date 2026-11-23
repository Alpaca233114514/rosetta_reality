"""Synthetic registry identity regressions; no simulation or model calls."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_test_" + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prepare = module("prepare_exploratory_scene_registry_v2")
auditor = module("audit_exploratory_scene_registry_v2")


def write(path, value):
    path.write_bytes(prepare.encoded(value))


def fixture(root):
    paths = []
    for group, seed in (("Q1", 2410), ("Q2", 2420), ("bank", 2440)):
        path = root / (group + "-plan.json")
        write(path, {"phase": group, "id": group + "-synthetic", "pairs": [{"seed": seed, "pair_id": group + "-parent"}],
                     "contract": "synthetic", "sampling_rule": "symmetric_common_legal_chunk_grid_v1", "minimum_common_chunk_starts": 50})
        paths.append(path)
    groups, parents, plans = prepare.registration(paths)
    entries = []
    sources = {"synthetic": "0" * 64}
    for group, seeds in groups.items():
        for seed in seeds:
            state = np.array([float(seed), 0.0], dtype=np.float64)
            physics_name, python_name = f"{seed}.npz", f"{seed}.json.gz"
            np.savez_compressed(root / physics_name, integration=state)
            payload = prepare.encoded({k: [] for k in ("owners", "rngs", "global_python", "global_numpy", "global_torch")})
            (root / python_name).write_bytes(gzip.compress(payload, mtime=0))
            geometry = {"synthetic_seed": seed}
            entries.append({"seed": seed, "group": group, "parent_id": parents.get(seed, "synthetic"),
                "scene_fingerprint": auditor.digest(geometry), "geometry": geometry,
                "reset_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                "integration_dtype": str(state.dtype), "integration_shape": list(state.shape),
                "model_sha256": "1" * 64, "complete_state_captured": True,
                "physics_path": physics_name, "python_state_path": python_name,
                "python_state_sha256": hashlib.sha256(payload).hexdigest(), "source_sha256": sources})
    write(root / "registration.json", {"groups": groups, "plans": plans, "source_sha256": sources})
    write(root / "registry.json", {"status": "completed", "groups": groups, "plans": plans, "entries": entries,
          "legacy_object_scene_nonoverlap": "not_established", "legacy_observations_read": False})
    seal(root)


def seal(root):
    write(root / "manifest.json", {"files": [{"path": p.name, "bytes": p.stat().st_size,
          "sha256": prepare.sha(p)} for p in sorted(root.iterdir()) if p.is_file() and p.name != "manifest.json"]})


def test_complete_synthetic_registry_checks_all_families(tmp_path):
    fixture(tmp_path)
    result = auditor.audit(tmp_path)
    assert result["passed"] and result["groups_complete"]
    assert result["registered_states"] == 15
    assert result["legacy_object_scene_nonoverlap"] == "not_established"
    assert result["state_restore_dynamics_tested"] is False


def test_registry_missing_state_is_rejected_even_when_resealed(tmp_path):
    fixture(tmp_path)
    value = json.loads((tmp_path / "registry.json").read_bytes())
    value["entries"].pop()
    write(tmp_path / "registry.json", value)
    seal(tmp_path)
    with pytest.raises(ValueError, match="missing, repeated or extra"):
        auditor.audit(tmp_path)


def test_state_changed_after_manifest_is_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path / "2410.npz").write_bytes(b"not-the-captured-state")
    with pytest.raises(ValueError, match="identity changed"):
        auditor.audit(tmp_path)


def test_duplicate_geometries_fail_without_dropping_denominator(tmp_path):
    fixture(tmp_path)
    value = json.loads((tmp_path / "registry.json").read_bytes())
    value["entries"][1]["geometry"] = value["entries"][0]["geometry"]
    value["entries"][1]["scene_fingerprint"] = value["entries"][0]["scene_fingerprint"]
    write(tmp_path / "registry.json", value)
    seal(tmp_path)
    result = auditor.audit(tmp_path)
    assert result["passed"] is False and result["registered_states"] == 15
    assert len(result["duplicate_scene_fingerprints"]) == 1


def test_parent_lineage_change_is_rejected(tmp_path):
    fixture(tmp_path)
    value = json.loads((tmp_path / "registry.json").read_bytes())
    value["entries"][0]["parent_id"] = "different-parent"
    write(tmp_path / "registry.json", value)
    seal(tmp_path)
    with pytest.raises(ValueError, match="parent lineage"):
        auditor.audit(tmp_path)
