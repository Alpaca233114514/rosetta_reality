"""Recompute fresh reset registry identities; no simulator or policy invocation."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def audit(root):
    import numpy as np
    root = root.resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    files = {}
    for row in manifest["files"]:
        name = row["path"]
        part = PurePosixPath(name)
        path = root.joinpath(*part.parts).resolve()
        if part.is_absolute() or ".." in part.parts or "\\" in name or ":" in name or not path.is_relative_to(root) or name in files:
            raise ValueError("Unsafe or repeated registry payload")
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Registry payload identity changed")
        files[name] = row
    if not {"registration.json", "registry.json"} <= files.keys():
        raise ValueError("Registry root records are not manifest bound")
    registry_bytes = (root / "registry.json").read_bytes()
    registry = json.loads(registry_bytes)
    registration = json.loads((root / "registration.json").read_bytes())
    groups = registry["groups"]
    expected = {(group, seed) for group, seeds in groups.items() for seed in seeds}
    total = sum(len(seeds) for seeds in groups.values())
    if (registry["status"] != "completed" or groups != registration["groups"] or total != len(expected)
            or len({seed for _, seed in expected}) != total
            or set(groups) != {"Q1", "Q2", "bank", "teacher_tuning", "policy_gate", "engineering"}):
        raise ValueError("Incomplete or overlapping registered source families")
    if (groups["teacher_tuning"] != [10, *range(1900, 1905)] or groups["policy_gate"] != list(range(1000, 1005))
            or groups["engineering"] != [2400] or registry["plans"] != registration["plans"]):
        raise ValueError("Registered reference families changed")
    parents = {}
    for group, maximum in (("Q1", 4), ("Q2", 20), ("bank", 10)):
        declared = registry["plans"][group]
        family = declared["family_definition"]
        if (family["phase"] != group or [p["seed"] for p in family["pairs"]] != groups[group]
                or not 1 <= len(groups[group]) <= maximum
                or digest(family) != declared["family_definition_sha256"]):
            raise ValueError("Captured source family definition differs")
        parents.update({p["seed"]: p["pair_id"] for p in family["pairs"]})
    entries = registry["entries"]
    if len(entries) != total or {(e["group"], e["seed"]) for e in entries} != expected:
        raise ValueError("Registry has missing, repeated or extra states")
    fingerprints, models = {}, set()
    for entry in entries:
        if entry["complete_state_captured"] is not True or entry["source_sha256"] != registration["source_sha256"]:
            raise ValueError("Incomplete reset source identity")
        if entry["seed"] in parents and entry["parent_id"] != parents[entry["seed"]]:
            raise ValueError("Registered parent lineage changed")
        if entry["physics_path"] not in files or entry["python_state_path"] not in files:
            raise ValueError("State payload not bound to registry manifest")
        with np.load(root / entry["physics_path"], allow_pickle=False) as values:
            state = values["integration"]
            if (not np.isfinite(state).all() or list(state.shape) != entry["integration_shape"]
                    or str(state.dtype) != entry["integration_dtype"]
                    or hashlib.sha256(state.tobytes()).hexdigest() != entry["reset_state_sha256"]):
                raise ValueError("Complete integration-state identity mismatch")
        python_bytes = gzip.decompress((root / entry["python_state_path"]).read_bytes())
        python = json.loads(python_bytes)
        if hashlib.sha256(python_bytes).hexdigest() != entry["python_state_sha256"] or not {"owners", "rngs", "global_python", "global_numpy", "global_torch"} <= python.keys():
            raise ValueError("Complete wrapper/RNG identity mismatch")
        fingerprint = digest(entry["geometry"])
        if fingerprint != entry["scene_fingerprint"]:
            raise ValueError("Reset geometry fingerprint mismatch")
        fingerprints.setdefault(fingerprint, []).append({"group": entry["group"], "seed": entry["seed"]})
        models.add(entry["model_sha256"])
    duplicates = [{"fingerprint": key, "states": values} for key, values in fingerprints.items() if len(values) > 1]
    if len(models) != 1 or registry["legacy_object_scene_nonoverlap"] != "not_established" or registry["legacy_observations_read"] is not False:
        raise ValueError("Physics or legacy-data boundary changed")
    return {"schema": "rosetta.exploratory_scene_registry_audit.v1", "passed": not duplicates,
            "registry_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(), "groups_complete": True,
            "registered_states": total, "duplicate_scene_fingerprints": duplicates,
            "legacy_object_scene_nonoverlap": "not_established", "legacy_observations_read": False,
            "full_state_files_hash_verified": True, "state_restore_dynamics_tested": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    raise SystemExit(0 if result["passed"] else 2)
