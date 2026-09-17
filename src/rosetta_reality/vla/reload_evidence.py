"""Full-array equality evidence; scalar metric agreement is never tensor proof."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from rosetta_reality.vla.training.plan import is_sha256

PROCESS_ID = uuid.uuid4().hex
ARRAYS = {
    "normalized_actions",
    "standard_actions",
    "noise",
    "sample_identities",
    "valid_mask",
}
HASH_IDENTITIES = {
    "plan_sha256",
    "model_sha256",
    "processor_sha256",
    "action_contract_sha256",
    "dataset_manifest_sha256",
    "input_sha256",
    "inference_recipe_sha256",
}
SHAPE_FIELDS = {
    "sample_count",
    "chunk_size",
    "normalized_action_dim",
    "standard_action_dim",
}
IDENTITIES = HASH_IDENTITIES | SHAPE_FIELDS


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate(arrays, identity):
    import numpy as np

    if set(identity) not in (IDENTITIES, IDENTITIES | {"noise_action_dim"}) or any(
        not is_sha256(identity[k]) for k in HASH_IDENTITIES
    ):
        raise ValueError(
            "Full-chunk evidence requires complete model/processor/input identities"
        )
    if any(type(identity[k]) is not int or identity[k] <= 0 for k in SHAPE_FIELDS):
        raise ValueError(
            "Registered sample, chunk and dimension counts must be positive integers"
        )
    if set(arrays) != ARRAYS:
        raise ValueError("Complete action chunks, noise, samples and mask are required")
    norm, standard, noise = (
        arrays[k] for k in ("normalized_actions", "standard_actions", "noise")
    )
    if any(
        a.ndim != 3 or not all(a.shape) or a.dtype.kind != "f"
        for a in (norm, standard, noise)
    ):
        raise ValueError(
            "Actions and noise must be nonempty floating [sample,horizon,dimension]"
        )
    noise_dim = identity.get("noise_action_dim", identity["normalized_action_dim"])
    if type(noise_dim) is not int or noise_dim <= 0:
        raise ValueError("Registered noise dimension must be a positive integer")
    if norm.shape[:2] != noise.shape[:2] or norm.shape[:2] != standard.shape[:2]:
        raise ValueError("Full-chunk array shapes disagree")
    prefix = (identity["sample_count"], identity["chunk_size"])
    if (
        noise.shape != (*prefix, noise_dim)
        or norm.shape != (*prefix, identity["normalized_action_dim"])
        or standard.shape
        != (
            *prefix,
            identity["standard_action_dim"],
        )
    ):
        raise ValueError(
            "Captured arrays do not cover the registered full sample/chunk dimensions"
        )
    if any(not np.isfinite(a).all() for a in (norm, standard, noise)):
        raise ValueError("Full-chunk evidence contains nonfinite values")
    samples, mask = arrays["sample_identities"], arrays["valid_mask"]
    if (
        samples.shape != (norm.shape[0], 2)
        or samples.dtype.kind not in "iu"
        or (samples < 0).any()
    ):
        raise ValueError(
            "Sample identities require nonnegative episode/frame integer pairs"
        )
    if (
        mask.shape != norm.shape[:2]
        or mask.dtype != np.bool_
        or not mask.any(axis=1).all()
    ):
        raise ValueError("Invalid action mask")
    if ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError("Padding cannot become valid again inside a chunk")


def write_bundle(output, arrays, identity):
    """Call from each actual collection process after capturing all chunk arrays."""
    import numpy as np

    _validate(arrays, identity)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "arrays.npz", **arrays)
    metadata = {
        "schema_version": 1,
        "kind": "full_action_chunk_evidence",
        "identity": identity,
        "process": {"id": PROCESS_ID, "pid": os.getpid()},
        "arrays_sha256": _sha(output / "arrays.npz"),
    }
    with (output / "manifest.json").open("x") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)


def compare_bundles(first, second, *, expected_identity):
    """Verify saved arrays from distinct collection processes, not model execution itself."""
    import numpy as np

    first, second = Path(first).resolve(), Path(second).resolve()
    if first == second:
        raise ValueError("Two independent collection paths are required")
    bundles = []
    for path in (first, second):
        meta = json.loads((path / "manifest.json").read_text())
        if (
            meta.get("schema_version") != 1
            or meta.get("kind") != "full_action_chunk_evidence"
        ):
            raise ValueError("Metrics-only reports are not full tensor evidence")
        if meta.get("identity") != expected_identity:
            raise ValueError(
                "Collection identity differs from the expected checkpoint and inputs"
            )
        if _sha(path / "arrays.npz") != meta.get("arrays_sha256"):
            raise ValueError("Full-chunk array checksum drift")
        with np.load(path / "arrays.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        _validate(arrays, meta["identity"])
        process = meta.get("process", {})
        if (
            not isinstance(process.get("id"), str)
            or len(process["id"]) != 32
            or type(process.get("pid")) is not int
            or process["pid"] <= 0
        ):
            raise ValueError("Collection process identity is absent")
        bundles.append((meta, arrays))
    if bundles[0][0]["process"]["id"] == bundles[1][0]["process"]["id"]:
        raise ValueError("Repeated collection in one process is not independent reload")
    equality = {
        key: (
            bundles[0][1][key].dtype == bundles[1][1][key].dtype
            and np.array_equal(bundles[0][1][key], bundles[1][1][key])
        )
        for key in sorted(ARRAYS)
    }
    return {
        "status": "passed" if all(equality.values()) else "tensor_mismatch",
        "exact_tensor_equality": all(equality.values()),
        "arrays_equal": equality,
        "identity": expected_identity,
        "manifest_sha256": [_sha(p / "manifest.json") for p in (first, second)],
        "proof_scope": "saved_full_arrays_from_distinct_collection_processes",
        "model_execution_proven_by_this_comparison": False,
        "formal_resume_verified": False,
        "m2_complete": False,
    }
