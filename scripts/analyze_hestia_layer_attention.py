"""Check layer-local intervention observations and summarize all causal/attention effects."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def probability_change(native, changed):
    """Image mass and within-image redistribution, averaged over rows and heads by caller."""
    if (
        native.shape != changed.shape
        or not np.isfinite(native).all()
        or not np.isfinite(changed).all()
    ):
        raise ValueError("Probability pair schema differs")
    n, p = native.astype(np.float64), changed.astype(np.float64)
    nm, pm = n.sum(-1, keepdims=True), p.sum(-1, keepdims=True)
    n, p = n / np.maximum(nm, 1e-30), p / np.maximum(pm, 1e-30)
    middle = 0.5 * (n + p)
    js = 0.5 * (
        (n * np.log(np.maximum(n, 1e-30) / np.maximum(middle, 1e-30))).sum(-1)
        + (p * np.log(np.maximum(p, 1e-30) / np.maximum(middle, 1e-30))).sum(-1)
    )
    return dict(
        image_mass_delta=(pm - nm)[..., 0],
        within_image_total_variation=0.5 * np.abs(p - n).sum(-1),
        within_image_js=js,
    )


def main():
    recovered, metric_path, output = map(Path, sys.argv[1:])
    result = json.loads(metric_path.read_text())
    conditions = result["conditions"]
    layers = tuple(range(1, 16, 2))
    calibration = json.loads((recovered / "base640_native/calibration.json").read_text())
    train = calibration["training_rows"]
    views = {"train40": train, "dev5": [r for r in range(45) if r not in train]}
    data, records = {}, {}
    for condition in conditions:
        path = recovered / condition / "attention.npz"
        record = json.loads((recovered / condition / "attention.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["attention_sha256"]:
            raise ValueError("Attention array SHA differs")
        if (
            record["condition"] != condition
            or record["captures"] != 360
            or record["noise_index"] != 0
            or record["denoise_index"] != 0
            or record["attention_output_returned_unchanged"] is not True
        ):
            raise ValueError("Attention observation identity differs")
        selected = (
            []
            if condition.endswith("_native")
            else list(layers)
            if condition.endswith(("_krestore", "_kdirection"))
            else [int(condition[-2:])]
        )
        if record["selected_layers"] != selected:
            raise ValueError("Selected layer identity differs")
        with np.load(path, allow_pickle=False) as a:
            if set(a.files) != {"image_probability", "summary"}:
                raise ValueError("Unexpected attention array")
            image, summary = a["image_probability"], a["summary"]
        if (
            image.shape != (45, 8, 15, 64)
            or summary.shape != (45, 8, 15, 8)
            or not np.isfinite(image).all()
            or not np.isfinite(summary).all()
        ):
            raise ValueError("Attention shape/finite contract differs")
        if (
            not np.allclose(image.sum(-1), summary[..., 0], atol=1e-6, rtol=0)
            or np.any(summary[..., 3] != 0)
            or np.any(np.abs(summary[..., :4].sum(-1) - 1) > 0.004)
        ):
            raise ValueError("Probability mass contract differs")
        data[condition] = image, summary
        records[condition] = record
    native_image, native_summary = data["base640_native"]
    if not all(
        np.array_equal(a, b)
        for a, b in zip(data["base640_krestore"], data["base640_native"], strict=True)
    ):
        raise ValueError("Restoration attention is not exact")
    prior_exact = {}
    for condition in conditions[3:]:
        selected = int(condition[-2:])
        stop = layers.index(selected)
        image, summary = data[condition]
        if not np.array_equal(image[:, :stop], native_image[:, :stop]) or not np.array_equal(
            summary[:, :stop], native_summary[:, :stop]
        ):
            raise ValueError("Attention changed before selected intervention layer")
        prior_exact[condition] = list(layers[:stop])
    effects = []
    for condition in conditions[2:]:
        image, summary = data[condition]
        pair = probability_change(native_image, image)
        for view, rows in views.items():
            for index, layer in enumerate(layers):
                for head in range(15):
                    effects.append(
                        dict(
                            condition=condition,
                            view=view,
                            layer=layer,
                            query_head=head,
                            kv_head=head // 3,
                            native_summary=native_summary[rows, index, head]
                            .mean(0)
                            .astype(float)
                            .tolist(),
                            changed_summary=summary[rows, index, head]
                            .mean(0)
                            .astype(float)
                            .tolist(),
                            summary_delta=(
                                summary[rows, index, head] - native_summary[rows, index, head]
                            )
                            .mean(0)
                            .astype(float)
                            .tolist(),
                            **{
                                key: float(value[rows, index, head].mean())
                                for key, value in pair.items()
                            },
                        )
                    )
    metrics = result["metrics"]

    def errors(condition, view="dev5", window="full", group="left_joint", space="standard"):
        return metrics[condition][space][view][window][group]["errors"]["model"]

    baseline = errors("base640_native")
    full = errors("base640_kdirection")
    full_benefit = baseline["mse"]["mean"] - full["mse"]["mean"]
    if full_benefit <= 0:
        raise ValueError("Registered all-layer direction benefit did not reproduce")
    layer_effects = []
    for condition in conditions[3:]:
        own = errors(condition)
        dm = np.array(own["mae"]["by_noise"]) - baseline["mae"]["by_noise"]
        ds = np.array(own["mse"]["by_noise"]) - baseline["mse"]["by_noise"]
        fraction = -float(ds.mean()) / full_benefit
        layer_effects.append(
            dict(
                condition=condition,
                layer=int(condition[-2:]),
                mae=own["mae"]["mean"],
                mse=own["mse"]["mean"],
                mae_delta_by_noise=dm.tolist(),
                mse_delta_by_noise=ds.tolist(),
                fraction_of_all_layer_mse_benefit=fraction,
                qualifies_for_head_isolation=bool(
                    fraction >= 0.60 and np.all(dm < 0) and np.all(ds < 0)
                ),
            )
        )
    eligible = [r for r in layer_effects if r["qualifies_for_head_isolation"]]
    selected = min(eligible, key=lambda r: (r["mse"], r["layer"]))["layer"] if eligible else None
    value = dict(
        layer_effects=layer_effects,
        next_registered_head_layer=selected,
        selection_is_development_diagnostic_only=True,
        prior_layer_attention_exact=prior_exact,
        restore_attention_exact=True,
        noise_index=0,
        denoise_index=0,
        head_effects=effects,
        summary_fields=records["base640_native"]["summary_fields"],
        attention_changes_are_observational=True,
        local_model_forwards=0,
    )
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in value.items() if k != "head_effects"}))


if __name__ == "__main__":
    main()
