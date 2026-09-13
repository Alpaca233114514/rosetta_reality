"""Bounded saved-weight algebra; never construct a policy or perform inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def spectrum(matrix):
    w = np.asarray(matrix, dtype=np.float64)
    if w.ndim != 2 or min(w.shape) == 0 or not np.isfinite(w).all():
        raise ValueError("Finite nonempty matrix required")
    u, s, vh = np.linalg.svd(w, full_matrices=False)
    tolerance = max(w.shape) * np.finfo(np.float64).eps * s[0]
    rank = int(np.count_nonzero(s > tolerance))
    error = np.linalg.norm((u * s) @ vh - w)
    norm = np.linalg.norm(w)
    if error > 1e-12 * max(norm, 1):
        raise ValueError("SVD reconstruction failed")
    if not np.isclose(np.square(s).sum(), np.square(w).sum(), atol=1e-12, rtol=1e-12):
        raise ValueError("Frobenius identity failed")
    return {
        "shape": list(w.shape),
        "rank": rank,
        "rank_threshold": float(tolerance),
        "maximum_singular": float(s[0]),
        "minimum_singular": float(s[-1]),
        "condition": float(s[0] / s[-1]) if rank == min(w.shape) else None,
        "stable_rank": float(np.square(s).sum() / s[0] ** 2) if s[0] > 0 else 0.0,
        "dimensions_below_one_percent_max": int(np.count_nonzero(s < 0.01 * s[0])),
        "frobenius_norm": float(norm),
        "reconstruction_l2_error": float(error),
    }, s


def run(plan, plan_path):
    from safetensors import safe_open

    if (
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["image"]
        or os.environ.get("HF_HUB_OFFLINE") != "1"
    ):
        raise ValueError("Pinned offline container required")
    started = time.monotonic()
    for path, sha in plan["sha256"].items():
        if digest(path) != sha:
            raise ValueError(f"Input/source drift: {path}")
    output = Path(plan["output"])
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("Fresh empty output required")
    names = [
        f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.{kind}_proj.weight"
        for layer in (1, 15)
        for kind in ("k", "v")
    ]
    metrics, spectra, previous, changes = {}, {}, {}, {}
    for step in (640, 1280):
        metrics[str(step)] = {}
        with safe_open(plan["checkpoints"][str(step)], framework="pt", device="cpu") as f:
            for name in names:
                w = f.get_tensor(name).double().numpy()
                if w.shape != (320, 320):
                    raise ValueError("Registered matrix shape differs")
                metric, singular = spectrum(w)
                metrics[str(step)][name] = metric
                spectra[f"{step}_{name}"] = singular
                if step == 640:
                    previous[name] = w.copy()
                else:
                    norm = np.linalg.norm(previous[name])
                    if norm == 0:
                        raise ValueError("Zero reference matrix; cannot form relative change")
                    a = metrics["640"][name]["condition"]
                    b = metric["condition"]
                    ratio = b / a if a is not None and b is not None else None
                    changes[name] = {
                        "relative_frobenius_change": float(
                            np.linalg.norm(w - previous[name]) / norm
                        ),
                        "condition_ratio": ratio,
                        "tenfold_condition_worsening": ratio >= 10 if ratio is not None else None,
                    }
    with (output / "spectra.npz").open("xb") as f:
        np.savez_compressed(f, **spectra)
    with np.load(output / "spectra.npz", allow_pickle=False) as f:
        if set(f.files) != set(spectra) or any(
            not np.array_equal(f[k], v) for k, v in spectra.items()
        ):
            raise ValueError("Spectrum reload drift")
    for path, sha in plan["sha256"].items():
        if digest(path) != sha:
            raise ValueError(f"Post-run drift: {path}")
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if elapsed >= 180 or rss >= 2 * 1024**3:
        raise ValueError("Resource budget exceeded")
    result = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": digest(plan_path),
        "metrics": metrics,
        "changes": changes,
        "array_sha256": digest(output / "spectra.npz"),
        "array_reload_exact": True,
        "seconds": elapsed,
        "peak_rss_bytes": rss,
        "weight_matrices_materialized": 8,
        "weight_values_materialized": 8 * 320 * 320,
        "policy_constructed": False,
        "policy_forwards": 0,
        "optimizer_steps": 0,
        "raw_sample_rows": 0,
        "hidden_loaded": False,
        "ssh_used": False,
        "unique_root_cause_established": False,
        "m2_complete": False,
    }
    with (output / "result.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.plan)
