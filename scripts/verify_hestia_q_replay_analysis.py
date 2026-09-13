"""Independent explicit-index action arithmetic and replay identity verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.hestia_q_replay import MODES


def verify_sentinels(root):
    """Float64 reconstruction with explicit BF16 rounding; separate from exact CUDA gates."""
    import torch

    checked = 0
    maximum = 0.0

    def bf16(a):
        return torch.from_numpy(np.asarray(a).copy()).bfloat16().float().numpy().astype(np.float64)

    def within_one_ulp(actual, expected):
        nonlocal checked, maximum
        delta = np.abs(actual - expected)
        # One BF16 spacing at the expected magnitude, with a normal/subnormal floor.
        scale = np.maximum(np.abs(expected), np.finfo(np.float32).tiny)
        tolerance = np.exp2(np.floor(np.log2(scale)) - 7)
        if not np.all(np.isfinite(actual)) or np.any(delta > tolerance):
            raise ValueError("Sentinel independent reconstruction exceeds one BF16 ULP")
        checked += delta.size
        maximum = max(maximum, float(delta.max()))

    count = 0
    for base in (640, 1280):
        for row in (0, 40):
            for step in range(10):
                for layer in range(1, 16, 2):
                    path = root / f"base{base}/sentinel-{row}-{step}-{layer}.npz"
                    with np.load(path, allow_pickle=False) as a:
                        q, k, v = (a[x].astype(np.float64) for x in ("q", "k", "v"))
                        heads = q.shape[2]
                        k = np.repeat(k, heads // k.shape[2], axis=2)
                        v = np.repeat(v, heads // v.shape[2], axis=2)
                        logits = bf16(
                            np.matmul(bf16(q).transpose(0, 2, 1, 3), bf16(k).transpose(0, 2, 3, 1))
                        )
                        logits = bf16(logits * q.shape[-1] ** -0.5)
                        logits = np.where(a["mask"][:, None], logits, np.finfo(np.float32).min)
                        exp = np.exp(logits - logits.max(-1, keepdims=True))
                        probs = bf16((exp / exp.sum(-1, keepdims=True)).astype(np.float32))
                        within_one_ulp(a["probs"], probs)
                        # Use the actual saved probability matrix for independent P*V.
                        output = bf16(
                            np.matmul(a["probs"].astype(np.float64), v.transpose(0, 2, 1, 3))
                        )
                        output = output.transpose(0, 2, 1, 3).reshape(a["output"].shape)
                        within_one_ulp(a["output"], output)
                    count += 1
    return {
        "sentinel_files": count,
        "scalar_checks": checked,
        "maximum_absolute_difference": maximum,
        "tolerance": "one_BF16_ULP_float64_reconstruction_only",
        "cuda_exact_control_tolerance": 0,
    }


def verify(root, analysis):
    checks = 0

    def equal(actual, expected):
        nonlocal checks
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-10, atol=1e-12):
            raise ValueError("Independent numeric recomputation differs")
        checks += a.size

    groups = analysis["groups"]
    for base in (640, 1280):
        with np.load(root / f"base{base}/arrays.npz", allow_pickle=False) as data:
            for key, record in analysis["metrics"].items():
                b, space, split, window, group, metric = key.split("/")
                if int(b) != base:
                    continue
                rows = list(range(40)) if split == "train40" else list(range(40, 45))
                slots = list(range(50)) if window == "full" else [0]
                dims = groups[group]
                y = data[f"{space}_targets"]
                errors = {}
                for mode in MODES:
                    p = data[f"{mode}_{space}_predictions"]
                    vals = []
                    for noise in range(4):
                        scene = []
                        for row in rows:
                            d = p[noise, row][np.ix_(slots, dims)] - y[row][np.ix_(slots, dims)]
                            scene.append(
                                float(np.sum(np.abs(d) if metric == "mae" else d * d) / d.size)
                            )
                        vals.append(scene)
                    errors[mode] = np.asarray(vals)
                    equal(vals, record["by_noise_scene"][mode])
                    equal([sum(v) / len(v) for v in vals], record["by_noise"][mode])
                for mode in MODES:
                    delta = errors["native"] - errors[mode]
                    equal(np.sum(delta, axis=1) / len(rows), record["gain"][mode])
                    for omitted in range(len(rows)):
                        kept = [r for r in range(len(rows)) if r != omitted]
                        equal(
                            delta[:, kept].mean(1),
                            record["leave_one_scene_out_gain"][mode][omitted],
                        )
                equal((errors["kmean_qnative"] - errors["kmean"]).mean(1), record["feedback_gain"])
                for method in ("mean", "median"):
                    values = []
                    for row in rows:
                        reference_rows = [r for r in range(40) if split != "train40" or r != row]
                        train = y[reference_rows][:, slots][:, :, dims]
                        c = np.mean(train, axis=0) if method == "mean" else np.median(train, axis=0)
                        d = c - y[row][np.ix_(slots, dims)]
                        values.append((np.abs(d) if metric == "mae" else d * d).mean())
                    equal(np.mean(values), record["constants"][method])
                if "decomposition" in record:
                    target = y[rows][:, slots][:, :, dims]
                    ey = target.mean(0)
                    for mode in MODES:
                        pred = data[f"{mode}_{space}_predictions"][:, rows][:, :, slots][
                            :, :, :, dims
                        ]
                        ep = pred.mean(1)
                        variance = (pred * pred).mean((1, 2, 3)) - (ep * ep).mean((1, 2))
                        covariance = (pred * target).mean((1, 2, 3)) - (ep * ey).mean((1, 2))
                        terms = record["decomposition"][mode]
                        equal(variance, terms["variance"])
                        equal(-2 * covariance, terms["minus_twice_covariance"])
                        equal(((ep - ey) ** 2).mean((1, 2)), terms["bias"])
                        # Include target scene variance to reconstruct absolute MSE.
                        total = (
                            np.array(terms["bias"])
                            + variance
                            - 2 * covariance
                            + ((target - ey) ** 2).mean()
                        )
                        equal(total, record["by_noise"][mode])
    events = 0
    for base in (640, 1280):
        with (root / f"base{base}/attention.jsonl").open() as stream:
            for noise in range(4):
                for row in range(45):
                    native, kmean = {}, {}
                    for mode in MODES:
                        for step in range(10):
                            for layer in range(1, 16, 2):
                                e = json.loads(next(stream))
                                if any(
                                    e[k] != v
                                    for k, v in {
                                        "row": row,
                                        "noise": noise,
                                        "mode": mode,
                                        "step": step,
                                        "layer": layer,
                                    }.items()
                                ):
                                    raise ValueError("Replay event ordering changed")
                                key = (step, layer)
                                if mode == "native":
                                    native[key] = e
                                else:
                                    n = native[key]
                                    for field in ("v", "mask"):
                                        if e[field] != n[field]:
                                            raise ValueError("Mask or V changed")
                                    if mode in ("qself", "kmean_qnative") and e["q"] != n["q"]:
                                        raise ValueError("Q replay used a different reference")
                                    if mode in ("qself", "aself", "kmean_ablock"):
                                        if e["output"] != n["output"] or e["probs"] != n["probs"]:
                                            raise ValueError(
                                                "Exact replay attention control failed"
                                            )
                                    if mode in ("qself", "aself") and e["k"] != n["k"]:
                                        raise ValueError("Native K changed")
                                    if mode == "kmean":
                                        kmean[key] = e
                                    elif mode.startswith("kmean") and e["k"] != kmean[key]["k"]:
                                        raise ValueError("K intervention differs across modes")
                                events += 1
            if stream.read().strip():
                raise ValueError("Unexpected extra attention events")
    return {
        "status": "passed",
        "independent_scalar_checks": checks,
        "attention_event_checks": events,
        "new_model_forwards": 0,
        "optimizer_steps": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.root, json.loads(args.analysis.read_text()))
    report["sentinel_verification"] = verify_sentinels(args.root)
    report["analysis_sha256"] = hashlib.sha256(args.analysis.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
