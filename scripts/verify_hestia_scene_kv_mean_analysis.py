"""Independent raw-array checks of all postprocessed errors and fixed templates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main():
    result_path, recovered, source, output = map(Path, sys.argv[1:])
    result = json.loads(result_path.read_text())
    meta = json.loads(
        (source / "runs/hestia-recovered-20260911-001/C-first/manifest.json").read_text()
    )["metadata"]
    rows = {
        name: [meta["episodes"].index(ep) for ep in meta["views"][name]]
        for name in ("train40", "dev5")
    }
    groups = {}
    for unit, kind in (("radian", "joint"), ("normalized", "gripper")):
        groups[kind] = [i for i, dim in enumerate(meta["dimensions"]) if dim["unit"] == unit]
        for side in ("left", "right"):
            groups[f"{side}_{kind}"] = [
                i for i in groups[kind] if meta["dimensions"][i]["name"].startswith(side + "_")
            ]
    arrays = {}
    for condition in result["conditions"]:
        with np.load(recovered / condition / "arrays.npz", allow_pickle=False) as saved:
            arrays[condition] = {k: saved[k].astype(np.float64) for k in saved.files}
    count, maximum = 0, 0.0

    def check(expected, actual):
        nonlocal count, maximum
        a, b = np.asarray(expected), np.asarray(actual)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Invalid independent comparison")
        if not np.allclose(a, b, atol=1e-12, rtol=1e-12):
            raise ValueError("Independent metric disagreement")
        count += a.size
        maximum = max(maximum, float(np.max(np.abs(a - b))))

    for condition, a in arrays.items():
        native = arrays[condition.split("_")[0] + "_native"]
        for space in ("standard", "normalized"):
            for view, indices in rows.items():
                for window, (start, stop) in {"full": (0, 50), "first": (0, 1)}.items():
                    for group, dims in groups.items():
                        p = a[space + "_predictions"][:, indices, start:stop][..., dims]
                        y = a[space + "_targets"][indices, start:stop][..., dims]
                        train_y = a[space + "_targets"][rows["train40"], start:stop][..., dims]
                        original_train = native[space + "_predictions"][
                            :, rows["train40"], start:stop
                        ][..., dims]
                        original_p = native[space + "_predictions"][:, indices, start:stop][
                            ..., dims
                        ]
                        label_mean = train_y.mean(0)
                        target_mean = y.mean(0)
                        prediction_mean = p.mean(1)
                        values = result["metrics"][condition][space][view][window][group]
                        for name, pred in (
                            ("model", p),
                            ("train_prediction_template", original_train.mean(1, keepdims=True)),
                            ("train_label_mean", label_mean),
                            ("train_label_median", np.median(train_y, axis=0)),
                        ):
                            error = np.broadcast_to(pred, p.shape) - y
                            for metric, per_noise in (
                                ("mae", np.abs(error).mean((1, 2, 3))),
                                ("mse", np.square(error).mean((1, 2, 3))),
                            ):
                                check(per_noise, values["errors"][name][metric]["by_noise"])
                                check(per_noise.mean(), values["errors"][name][metric]["mean"])
                            check(
                                np.abs(error).mean((2, 3)),
                                values["errors"][name]["mae_by_noise_episode"],
                            )
                        variance = (np.square(p).mean(1) - np.square(prediction_mean)).mean((1, 2))
                        covariance = ((p * y).mean(1) - prediction_mean * target_mean).mean((1, 2))
                        bias = (
                            np.square(prediction_mean - target_mean)
                            - np.square(label_mean - target_mean)
                        ).mean((1, 2))
                        gap = (np.square(p - y) - np.square(label_mean - y)).mean((1, 2, 3))
                        for key, expected in (
                            ("prediction_scene_variance", variance),
                            ("minus_twice_scene_covariance", -2 * covariance),
                            ("squared_scene_mean_bias_difference", bias),
                            ("model_minus_train_mean_mse", gap),
                        ):
                            check(expected, values[key]["by_noise"])
                            check(expected.mean(), values[key]["mean"])
                        check(gap, bias + variance - 2 * covariance)
                        pair = result["effects"][condition][space][view][window][group]
                        check(
                            (np.square(p - y) - np.square(original_p - y)).mean((1, 2, 3)),
                            pair["mse_delta_by_noise"],
                        )
                        check(
                            (np.abs(p - y) - np.abs(original_p - y)).mean((1, 2, 3)),
                            pair["mae_delta_by_noise"],
                        )
    receipt = {
        "status": "passed",
        "scalar_comparisons": int(count),
        "maximum_absolute_difference": maximum,
        "fixed_native_train_templates_verified": True,
        "local_model_forwards": 0,
        "optimizer_steps": 0,
    }
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
