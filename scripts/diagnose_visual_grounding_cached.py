"""Repeat a registered diagnostic with bounded image reuse and exact parity checks.

Pass the ordinary diagnose_visual_grounding.py arguments plus --reference-report
and --cache-report. Both outputs are create-only; any prediction difference fails
the acceleration check while preserving both reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    args, baseline_arguments = parser.parse_known_args()
    if args.cache_report.exists():
        raise FileExistsError("Cache evidence is create-only.")
    reference = json.loads(args.reference_report.read_text())
    if "--output" not in baseline_arguments:
        raise ValueError("The cached diagnostic requires an explicit output path.")
    output = Path(baseline_arguments[baseline_arguments.index("--output") + 1])
    if len({path.resolve() for path in (output, args.cache_report, args.reference_report)}) != 3:
        raise ValueError("Reference, predictions and cache evidence need distinct paths.")

    import diagnose_frame0_vision_probe as loader
    import diagnose_visual_grounding as baseline
    import numpy as np

    from rosetta_reality.experiment import file_sha256
    from rosetta_reality.vla.inference_image_cache import FrozenImageEmbeddingCache

    original_loader, original_arguments = loader._load_artifact, sys.argv
    caches = []
    try:
        with ExitStack() as stack:

            def load_with_cache(artifact_id, device):
                if artifact_id != reference["artifact_id"]:
                    raise ValueError("The acceleration reference names another artifact.")
                policy, processors = original_loader(artifact_id, device)
                if policy.config.compile_model:
                    raise ValueError("Image reuse is registered only for eager inference.")
                cache = stack.enter_context(FrozenImageEmbeddingCache(policy.model.vlm_with_expert))
                caches.append(cache)
                return policy, processors

            loader._load_artifact = load_with_cache
            sys.argv = [sys.argv[0], *baseline_arguments]
            result = baseline.main()
    finally:
        loader._load_artifact, sys.argv = original_loader, original_arguments
    if result:
        return result
    accelerated = json.loads(output.read_text())
    identity_fields = (
        "artifact_id",
        "artifact_manifest_sha256",
        "dataset_revision",
        "dataset_manifest_sha256",
        "action_contract_sha256",
        "container_image_id",
        "device",
        "torch_version",
        "episodes",
        "frame_offsets",
        "noise_seeds",
        "mismatched_image_episodes",
        "image_identity",
        "implementation_sha256",
    )
    if any(reference.get(field) != accelerated.get(field) for field in identity_fields):
        raise ValueError("Cached inference and reference diagnostic identities differ.")
    exact, maximum_difference = True, 0.0
    for before, after in zip(reference["results"], accelerated["results"], strict=True):
        if (before["frame_offset"], before["noise_seed"]) != (
            after["frame_offset"],
            after["noise_seed"],
        ):
            raise ValueError("Cached comparison sample/noise ordering changed.")
        for key in ("correct_predictions", "mismatched_predictions"):
            left, right = np.asarray(before[key]), np.asarray(after[key])
            if left.shape != right.shape:
                raise ValueError("Cached prediction shape changed.")
            exact = exact and np.array_equal(left, right)
            maximum_difference = max(maximum_difference, float(np.abs(left - right).max()))
    evidence = {
        "schema_version": 1,
        "stage": "frozen_visual_cache_parity",
        "status": "passed" if exact else "failed",
        "exact_action_equality": bool(exact),
        "compared_output": "executed first action, all 14 dimensions, every paired forward",
        "maximum_absolute_action_difference": maximum_difference,
        "reference_report_sha256": file_sha256(args.reference_report),
        "cached_report_sha256": file_sha256(output),
        "reference_seconds": reference["elapsed_seconds"],
        "cached_seconds": accelerated["elapsed_seconds"],
        "speedup": reference["elapsed_seconds"] / accelerated["elapsed_seconds"],
        "cache": [cache.report() for cache in caches],
        "implementation_sha256": {
            name: file_sha256(REPOSITORY_ROOT / name)
            for name in (
                "scripts/diagnose_visual_grounding_cached.py",
                "src/rosetta_reality/vla/inference_image_cache.py",
            )
        },
        "limitation": (
            "One sequential timing comparison includes model loading and host variation; "
            "no training/Gate claim. Equality covers recorded first actions, not entire chunks."
        ),
    }
    with args.cache_report.open("x") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
    print(json.dumps(evidence), flush=True)
    return 0 if exact else 4


if __name__ == "__main__":
    raise SystemExit(main())
