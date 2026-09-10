"""Verify the vfunfreeze two-step smoke actually moved the visual front end.

The registered optimizer-smoke evidence for the vision-front-end axis must
prove, on top of the generic finite-loss guarantees, that (1) the declared
front-end parameters received non-zero optimizer updates, and (2) every
language-model parameter stayed bit-identical.  This script compares the
two-step smoke checkpoint's ``model.safetensors`` tensor-by-tensor against the
revision-pinned base snapshot the run initialized from, classifies every
parameter by name, and writes a create-only JSON verdict.  It never loads
datasets or accelerators and runs on CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_phase as phase_runner  # noqa: E402
import smolvla_vfunfreeze_protocol as protocol  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla.action_space import load_smolvla_experiment  # noqa: E402

# Serialized policy tensors carry the ``model.`` prefix of the SmolVLAPolicy
# wrapper around VLAFlowMatching.
SCOPE_PREFIXES = (
    "model.vlm_with_expert.vlm.model.vision_model.",
    "model.vlm_with_expert.vlm.model.connector.",
)
LANGUAGE_PREFIXES = (
    "model.vlm_with_expert.vlm.model.language_model.",
    "model.vlm_with_expert.vlm.model.text_model.",
    "model.vlm_with_expert.vlm.model.embed_tokens.",
    "model.vlm_with_expert.vlm.lm_head.",
    "model.vlm_with_expert.vlm.model.lm_head.",
)
MIN_SCOPE_CHANGED_FRACTION = 0.5


def _classify(name: str) -> str:
    if name.startswith(SCOPE_PREFIXES):
        return "vision_front_end"
    if name.startswith(LANGUAGE_PREFIXES):
        return "language_model"
    return "baseline_trainable_or_buffer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    experiment_path = REPOSITORY_ROOT / protocol.PARENT_CONFIG
    experiment = load_smolvla_experiment(experiment_path, REPOSITORY_ROOT)
    base_root = phase_runner._model_root(experiment)
    revision = str(experiment["model"]["revision"])
    base_file = base_root / "model.safetensors"
    if not base_file.is_file():
        raise FileNotFoundError(f"Revision-pinned base snapshot is missing: {base_file.name}")
    checkpoint_file = args.checkpoint.resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Smoke checkpoint is missing: {checkpoint_file.name}")
    if args.output.resolve().exists():
        raise FileExistsError("The smoke-update verdict is create-only.")

    with safe_open(base_file, framework="pt") as base_handle:
        base_keys = set(base_handle.keys())
        base_tensors = {key: base_handle.get_tensor(key) for key in base_keys}
    with safe_open(checkpoint_file, framework="pt") as handle:
        checkpoint_keys = set(handle.keys())
        if checkpoint_keys != base_keys:
            missing = sorted(base_keys - checkpoint_keys)
            extra = sorted(checkpoint_keys - base_keys)
            raise ValueError(
                "Smoke checkpoint tensor inventory differs from the base "
                f"snapshot (missing: {missing[:8]}, extra: {extra[:8]})."
            )

    stats: dict[str, dict[str, object]] = {
        "vision_front_end": {"total": 0, "changed": 0, "max_abs_difference": 0.0},
        "language_model": {"total": 0, "changed": 0, "max_abs_difference": 0.0},
        "baseline_trainable_or_buffer": {"total": 0, "changed": 0, "max_abs_difference": 0.0},
    }
    with safe_open(checkpoint_file, framework="pt") as handle:
        for key in sorted(checkpoint_keys):
            klass = _classify(key)
            updated = handle.get_tensor(key)
            reference = base_tensors[key]
            difference = float(
                torch.abs(updated.double() - reference.double()).max()
            )
            entry = stats[klass]
            entry["total"] = int(entry["total"]) + 1
            entry["max_abs_difference"] = max(float(entry["max_abs_difference"]), difference)
            if difference != 0.0:
                entry["changed"] = int(entry["changed"]) + 1
    del base_tensors

    vision = stats["vision_front_end"]
    language = stats["language_model"]
    scope_changed_fraction = (
        float(vision["changed"]) / float(vision["total"]) if int(vision["total"]) else 0.0
    )
    criteria = [
        {
            "name": "vision_front_end_parameters_present",
            "passed": int(vision["total"]) > 0,
            "measured": int(vision["total"]),
        },
        {
            "name": "language_model_parameters_present",
            "passed": int(language["total"]) > 0,
            "measured": int(language["total"]),
        },
        {
            "name": "vision_front_end_nonzero_update_fraction",
            "passed": scope_changed_fraction >= MIN_SCOPE_CHANGED_FRACTION,
            "measured": scope_changed_fraction,
            "threshold_min": MIN_SCOPE_CHANGED_FRACTION,
        },
        {
            "name": "language_model_bit_identical",
            "passed": int(language["changed"]) == 0,
            "measured": int(language["changed"]),
            "threshold_max": 0,
        },
    ]
    passed = all(criterion["passed"] for criterion in criteria)
    verdict = {
        "schema_version": 1,
        "stage": "smolvla_vfunfreeze_smoke_update_verification",
        "status": "passed" if passed else "failed",
        "checkpoint": checkpoint_file.name,
        "checkpoint_sha256": file_sha256(checkpoint_file),
        "base_snapshot_sha256": file_sha256(base_file),
        "base_revision": revision,
        "classification_prefixes": {
            "vision_front_end": list(SCOPE_PREFIXES),
            "language_model": list(LANGUAGE_PREFIXES),
        },
        "stats": stats,
        "criteria": criteria,
        "hidden_test_loaded": False,
        "vfunfreeze_protocol": {
            "wrapper_sha256": file_sha256(Path(__file__)),
            "protocol_module_sha256": file_sha256(
                REPOSITORY_ROOT / "scripts/smolvla_vfunfreeze_protocol.py"
            ),
            "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        },
    }
    create_json(args.output.resolve(), verdict)
    print(json.dumps({"status": verdict["status"], "stats": stats}))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
