"""Independent NumPy train-calibration checks and per-head descriptive geometry."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def bf16_nearest(x):
    """Round finite float64 to the nearest BF16 value without torch or double rounding."""
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite BF16 input")
    a = np.abs(x)
    bits = a.astype(np.float32).view(np.uint32) & np.uint32(0xFFFF0000)
    lower = bits.view(np.float32).astype(np.float64)
    bits = np.where(lower > a, bits - np.uint32(65536), bits).astype(np.uint32)
    lower = bits.view(np.float32).astype(np.float64)
    upper = (bits + np.uint32(65536)).view(np.float32).astype(np.float64)
    dl, du = a - lower, upper - a
    higher = (du < dl) | ((du == dl) & ((bits & np.uint32(65536)) != 0))
    return np.copysign(np.where(higher, upper, lower), x).astype(np.float32)


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite descriptive geometry")
    return dict(
        mean=float(a.mean()),
        minimum=float(a.min()),
        maximum=float(a.max()),
        quantiles=np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
    )


def main():
    recovered, metrics_path, output = map(Path, sys.argv[1:])
    native = recovered / "base640_native"
    calibration = json.loads((native / "calibration.json").read_text())
    raw_path = native / "native-keys.npz"
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != calibration["native_keys_sha256"]:
        raise ValueError("Native key file identity differs")
    mean_path = native / "calibration.npz"
    if hashlib.sha256(mean_path.read_bytes()).hexdigest() != calibration["arrays_sha256"]:
        raise ValueError("Calibration array identity differs")
    train = calibration["training_rows"]
    dev = [row for row in range(45) if row not in train]
    heads, dim = calibration["kv_heads"], calibration["head_dim"]
    if len(train) != 40 or len(set(train)) != 40 or len(dev) != 5 or heads * dim != 320:
        raise ValueError("Calibration row/head schema differs")
    layers = list(range(1, 16, 2))
    geometry = []
    comparisons, max_norm_error = 0, 0.0
    with (
        np.load(raw_path, allow_pickle=False) as raw,
        np.load(mean_path, allow_pickle=False) as means,
    ):
        if set(raw.files) != set(map(str, layers)):
            raise ValueError("Raw layer set differs")
        for layer in layers:
            k = raw[str(layer)].astype(np.float64)
            if k.shape != (45, 1, 64, 320) or not np.array_equal(bf16_nearest(k), k):
                raise ValueError("Raw key BF16 layout differs")
            m = means[f"{layer}_mean"]
            if not np.array_equal(bf16_nearest(k[train].mean(0)), m):
                raise ValueError("Independent train-only mean mismatch")
            k = k.reshape(45, 64, heads, dim)
            r = np.sqrt(np.square(k).sum(-1))
            n = means[f"{layer}_norm"].reshape(64, heads)
            norm_error = float(np.max(np.abs(r[train].mean(0) - n)))
            if norm_error > 1e-12:
                raise ValueError("Independent train-only mean-norm mismatch")
            max_norm_error = max(norm_error, max_norm_error)
            comparisons += m.size + n.size
            m = m.astype(np.float64).reshape(64, heads, dim)
            s = np.linalg.norm(m, axis=-1)
            if min(r.min(), s.min(), n.min()) <= 1e-12:
                raise ValueError("Undefined direction")
            cosine = (k * m).sum(-1) / (r * s)
            for head in range(heads):
                geometry.append(
                    {
                        "layer": layer,
                        "head": head,
                        "norm_mean_over_mean_norm": distribution(s[:, head] / n[:, head]),
                        "train_native_norm": distribution(r[train, :, head]),
                        "dev_native_norm": distribution(r[dev, :, head]),
                        "train_cosine_to_mean": distribution(cosine[train, :, head]),
                        "dev_cosine_to_mean": distribution(cosine[dev, :, head]),
                        "dev_to_train_norm_ratio": distribution(r[dev, :, head] / n[:, head]),
                    }
                )
    metrics = json.loads(metrics_path.read_text())
    factorial = []
    for space in ("standard", "normalized"):
        for view in ("train40", "dev5"):
            for window in ("first", "full"):
                for group in (
                    "joint",
                    "gripper",
                    "left_joint",
                    "right_joint",
                    "left_gripper",
                    "right_gripper",
                ):
                    for metric in ("mae", "mse"):
                        values = {
                            mode: np.asarray(
                                metrics["metrics"]["base640_" + mode][space][view][window][group][
                                    "errors"
                                ]["model"][metric]["by_noise"]
                            )
                            for mode in (
                                "native",
                                "kmean",
                                "kdirection",
                                "kmagnitude",
                                "knormavg",
                                "kdiravg",
                            )
                        }
                        interaction = (
                            values["kmean"]
                            - values["kdirection"]
                            - values["kmagnitude"]
                            + values["native"]
                        )
                        factorial.append(
                            {
                                "space": space,
                                "view": view,
                                "window": window,
                                "group": group,
                                "metric": metric,
                                "direction_delta": (
                                    values["kdirection"] - values["native"]
                                ).tolist(),
                                "magnitude_delta": (
                                    values["kmagnitude"] - values["native"]
                                ).tolist(),
                                "combined_delta": (values["kmean"] - values["native"]).tolist(),
                                "interaction": interaction.tolist(),
                                "normavg_delta": (values["knormavg"] - values["native"]).tolist(),
                                "diravg_delta": (values["kdiravg"] - values["native"]).tolist(),
                            }
                        )
    result = {
        "calibration_verified": True,
        "calibration_scalar_comparisons": comparisons,
        "maximum_norm_error": max_norm_error,
        "head_dim": dim,
        "kv_heads": heads,
        "query_heads": calibration["query_heads"],
        "geometry": geometry,
        "factorial": factorial,
        "local_model_forwards": 0,
        "attention_logits_observed": False,
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in {"geometry", "factorial"}}))


if __name__ == "__main__":
    main()
