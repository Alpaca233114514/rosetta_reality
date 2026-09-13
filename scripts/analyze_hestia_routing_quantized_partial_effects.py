"""Validate routing factor isolation and apply the preregistered dominance rule."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from scripts.analyze_hestia_joint_attention import qualify, subset_exact
from scripts.analyze_hestia_layer_attention import probability_change

CONDITIONS = ["base640_" + m for m in ("native", "rrestore", "rfull", "rintra")]


def classify(metrics):
    def error(condition):
        return metrics[condition]["standard"]["dev5"]["full"]["left_joint"]["errors"]["model"]

    result = qualify(error("base640_native"), error("base640_rfull"), error("base640_rintra"))
    return dict(
        fraction_of_full_pair_mse_benefit=result["fraction_of_all_head_mse_benefit"],
        mae_delta_by_noise=result["mae_delta_by_noise"],
        mse_delta_by_noise=result["mse_delta_by_noise"],
        intra_sufficiency_qualified=result["pair_dominance_qualified"],
        mass_component="not measured: isolation gate failed",
        complete_factorial_classification="not measured",
        mass_contribution_excluded=False,
    )


def main():
    recovered, metric_path, output = map(Path, sys.argv[1:])
    result = json.loads(metric_path.read_text())
    if result["conditions"] != CONDITIONS:
        raise ValueError("Condition coverage differs")
    calibration = json.loads((recovered / "base640_native/calibration.json").read_text())
    train = calibration["training_rows"]
    views = {"train40": train, "dev5": [r for r in range(45) if r not in train]}
    data, records, routing, fields = {}, {}, {}, None
    bounds = dict(
        pre_mass_error=1e-12,
        post_sum_error=0.004,
        post_intra_mass_leak=0.004,
        post_mass_conditional_tv_leak=0.004,
        balanced_mass_error=1e-4,
        balanced_conditional_tv=0.004,
        rounding_correction_l1=2.0,
    )
    for condition in CONDITIONS:
        mode = condition.split("_")[1]
        folder = recovered / condition
        record = json.loads((folder / "attention.json").read_text())
        route = json.loads((folder / "routing.json").read_text())
        selection = json.loads((folder / "head-selection.json").read_text())
        if selection != dict(
            layer=1,
            heads=[] if mode == "native" else [1, 2],
            condition=condition,
            head_dim=64,
            query_heads_per_kv_head=3,
        ):
            raise ValueError("Head selection differs")
        if (
            route["mode"] != mode
            or route["layer"] != 1
            or route["kv_heads"] != [1, 2]
            or route["query_heads"] != list(range(3, 9))
            or route["calls"] != (0 if mode == "native" else 1800)
            or route["original_eager_kernel"] is not True
            or route["matched_current_q_and_v"] is not True
        ):
            raise ValueError("Routing mechanism or coverage differs")
        maxima = route["maxima"]
        if mode != "native" and (
            set(maxima) != set(bounds)
            or any(
                not np.isfinite(maxima[k]) or not 0 <= maxima[k] <= bound
                for k, bound in bounds.items()
            )
        ):
            raise ValueError("Factor isolation leakage exceeds registered bound")
        path = folder / "attention.npz"
        if (
            record["attention_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
            or record["condition"] != condition
            or record["routing_mode"] != mode
            or record["captures"] != 360
            or record["noise_index"] != 0
            or record["denoise_index"] != 0
            or record["softmax_return_overridden"] != (mode in {"rrestore", "rintra", "rmass"})
            or record["selected_layers"] != ([] if mode == "native" else [1])
        ):
            raise ValueError("Attention provenance differs")
        if fields is None:
            fields = record["summary_fields"]
        if fields != record["summary_fields"]:
            raise ValueError("Summary fields differ")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"image_probability", "summary"}:
                raise ValueError("Attention arrays differ")
            image, summary = arrays["image_probability"], arrays["summary"]
        if (
            image.shape != (45, 8, 15, 64)
            or summary.shape != (45, 8, 15, 8)
            or not np.isfinite(image).all()
            or not np.isfinite(summary).all()
            or np.any(image < 0)
            or np.any(summary[..., 3] != 0)
            or not np.allclose(image.sum(-1), summary[..., 0], atol=1e-6, rtol=0)
        ):
            raise ValueError("Attention array contract differs")
        data[condition], records[condition], routing[condition] = (image, summary), record, route
    native = data["base640_native"]
    if not all(np.array_equal(a, b) for a, b in zip(native, data["base640_rrestore"], strict=True)):
        raise ValueError("Restore attention differs")
    exact = {c: subset_exact(native, data[c], [1, 2]) for c in CONDITIONS[2:]}
    observations = []
    for condition in CONDITIONS[2:]:
        image, summary = data[condition]
        delta = probability_change(native[0], image)
        for view, rows in views.items():
            for index, layer in enumerate(range(1, 16, 2)):
                for head in range(15):
                    observations.append(
                        dict(
                            condition=condition,
                            view=view,
                            layer=layer,
                            query_head=head,
                            native_summary=native[1][rows, index, head].mean(0).tolist(),
                            changed_summary=summary[rows, index, head].mean(0).tolist(),
                            **{k: float(v[rows, index, head].mean()) for k, v in delta.items()},
                        )
                    )
    value = dict(
        primary=classify(result["metrics"]),
        routing_checks=routing,
        unselected_query_heads_exact=exact,
        restore_attention_exact=True,
        attention_effects=observations,
        summary_fields=fields,
        local_model_forwards=0,
        attention_observation_scope="first noise and denoise; mean of50 queries",
        numerical_leakage_scope="maximum over all1800 selected-layer calls per intervention",
    )
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in value.items() if k != "attention_effects"}))


if __name__ == "__main__":
    main()
