"""Validate head isolation and apply the preregistered routing selection rule."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from scripts.analyze_hestia_layer_attention import probability_change


def unchanged_query_heads(native, changed, kv_head):
    """Only the selected GQA triplet may change at the intervention layer."""
    if kv_head not in range(5):
        raise ValueError("Unregistered KV head")
    others = [h for h in range(15) if h // 3 != kv_head]
    for before, after in zip(native, changed, strict=True):
        if not np.array_equal(before[:, 0, others], after[:, 0, others]):
            raise ValueError("Unselected first-layer query head changed")
    return others


def head_effects(metrics):
    def errors(condition):
        return metrics[condition]["standard"]["dev5"]["full"]["left_joint"]["errors"]["model"]

    baseline = errors("base640_native")
    full = errors("base640_kdirection")
    denominator = baseline["mse"]["mean"] - full["mse"]["mean"]
    if denominator <= 0:
        raise ValueError("Parent layer1 benefit did not reproduce")
    rows = []
    for head in range(5):
        own = errors(f"base640_kh{head}")
        dm = np.asarray(own["mae"]["by_noise"]) - baseline["mae"]["by_noise"]
        ds = np.asarray(own["mse"]["by_noise"]) - baseline["mse"]["by_noise"]
        fraction = -float(ds.mean()) / denominator
        rows.append(dict(
            condition=f"base640_kh{head}", layer=1, kv_head=head,
            mae=own["mae"]["mean"], mse=own["mse"]["mean"],
            mae_delta_by_noise=dm.tolist(), mse_delta_by_noise=ds.tolist(),
            fraction_of_layer1_mse_benefit=fraction,
            qualifies_for_routing_isolation=bool(
                fraction >= 0.60 and np.all(dm < 0) and np.all(ds < 0)
            ),
        ))
    return rows


def main():
    recovered, metric_path, output = map(Path, sys.argv[1:])
    result = json.loads(metric_path.read_text())
    conditions = ["base640_native", "base640_krestore", "base640_kdirection"] + [
        f"base640_kh{h}" for h in range(5)
    ]
    if result["conditions"] != conditions:
        raise ValueError("Registered head conditions differ")
    calibration = json.loads((recovered / "base640_native/calibration.json").read_text())
    train = calibration["training_rows"]
    views = {"train40": train, "dev5": [r for r in range(45) if r not in train]}
    data, records = {}, {}
    for condition in conditions:
        folder = recovered / condition
        path = folder / "attention.npz"
        record = json.loads((folder / "attention.json").read_text())
        heads = [] if condition.endswith("native") else list(range(5)) if condition.endswith(
            ("krestore", "kdirection")
        ) else [int(condition[-1])]
        selection = json.loads((folder / "head-selection.json").read_text())
        if selection != dict(layer=1, heads=heads, condition=condition, head_dim=64,
                             query_heads_per_kv_head=3):
            raise ValueError("Head identity differs")
        if (record["attention_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
            or record["condition"] != condition or record["captures"] != 360
            or record["noise_index"] != 0 or record["denoise_index"] != 0
            or record["attention_output_returned_unchanged"] is not True
            or record["selected_layers"] != ([] if not heads else [1])):
            raise ValueError("Attention capture identity differs")
        with np.load(path, allow_pickle=False) as a:
            if set(a.files) != {"image_probability", "summary"}:
                raise ValueError("Unexpected attention arrays")
            image, summary = a["image_probability"], a["summary"]
        if (image.shape != (45, 8, 15, 64) or summary.shape != (45, 8, 15, 8)
            or not np.isfinite(image).all() or not np.isfinite(summary).all()
            or np.any(image < 0)
            or not np.allclose(image.sum(-1), summary[..., 0], atol=1e-6, rtol=0)
            or np.any(summary[..., 3] != 0)
            or np.any(np.abs(summary[..., :4].sum(-1) - 1) > 0.004)):
            raise ValueError("Attention probability contract differs")
        data[condition], records[condition] = (image, summary), record
    native = data["base640_native"]
    if not all(np.array_equal(a, b) for a, b in zip(native, data["base640_krestore"], strict=True)):
        raise ValueError("Restore attention differs")
    exact = {f"base640_kh{h}": unchanged_query_heads(native, data[f"base640_kh{h}"], h)
             for h in range(5)}
    effects = []
    for condition in conditions[2:]:
        image, summary = data[condition]
        pair = probability_change(native[0], image)
        for view, rows in views.items():
            for index, layer in enumerate(range(1, 16, 2)):
                for head in range(15):
                    effects.append(dict(
                        condition=condition, view=view, layer=layer, query_head=head, kv_head=head//3,
                        native_summary=native[1][rows, index, head].mean(0).astype(float).tolist(),
                        changed_summary=summary[rows, index, head].mean(0).astype(float).tolist(),
                        summary_delta=(summary[rows, index, head]-native[1][rows, index, head])
                                      .mean(0).astype(float).tolist(),
                        **{k: float(v[rows, index, head].mean()) for k, v in pair.items()},
                    ))
    causal = head_effects(result["metrics"])
    eligible = [r for r in causal if r["qualifies_for_routing_isolation"]]
    selected = min(eligible, key=lambda r: (r["mse"], r["kv_head"]))["kv_head"] if eligible else None
    value = dict(
        head_effects=causal, next_registered_routing_head=selected,
        selection_is_development_diagnostic_only=True, unselected_query_heads_exact=exact,
        restore_attention_exact=True, noise_index=0, denoise_index=0,
        attention_effects=effects, summary_fields=records["base640_native"]["summary_fields"],
        attention_changes_are_observational=True, local_model_forwards=0,
    )
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in value.items() if k != "attention_effects"}))


if __name__ == "__main__":
    main()
