"""Check measured joint head effects, attention isolation and the fixed 80% rule."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from scripts.analyze_hestia_layer_attention import probability_change

SUBSETS = {"kh1": [1], "kh2": [2], "kh12": [1, 2], "kh034": [0, 3, 4]}
CONDITIONS = ["base640_native", "base640_krestore", "base640_kdirection"] + [
    "base640_" + name for name in SUBSETS
]


def subset_exact(native, changed, heads):
    if not heads or len(set(heads)) != len(heads) or any(h not in range(5) for h in heads):
        raise ValueError("Invalid head subset")
    unselected = [h for h in range(15) if h // 3 not in heads]
    for a, b in zip(native, changed, strict=True):
        if not np.array_equal(a[:, 0, unselected], b[:, 0, unselected]):
            raise ValueError("Unselected query attention changed")
    return unselected


def qualify(native, full, pair):
    denominator = native["mse"]["mean"] - full["mse"]["mean"]
    if denominator <= 0:
        raise ValueError("Parent all-head benefit did not reproduce")
    dm = np.asarray(pair["mae"]["by_noise"]) - native["mae"]["by_noise"]
    ds = np.asarray(pair["mse"]["by_noise"]) - native["mse"]["by_noise"]
    fraction = -float(ds.mean()) / denominator
    return dict(
        fraction_of_all_head_mse_benefit=fraction,
        mae_delta_by_noise=dm.tolist(),
        mse_delta_by_noise=ds.tolist(),
        pair_dominance_qualified=bool(fraction >= 0.80 and np.all(dm < 0) and np.all(ds < 0)),
    )


def main():
    source, result_path, output = map(Path, sys.argv[1:])
    result = json.loads(result_path.read_text())
    if result["conditions"] != CONDITIONS:
        raise ValueError("Joint conditions differ")
    data, fields = {}, None
    for condition in CONDITIONS:
        mode = condition.split("_")[1]
        heads = [] if mode == "native" else SUBSETS.get(mode, list(range(5)))
        path = source / condition / "attention.npz"
        record = json.loads((source / condition / "attention.json").read_text())
        selection = json.loads((source / condition / "head-selection.json").read_text())
        if selection != dict(
            layer=1, heads=heads, condition=condition, head_dim=64, query_heads_per_kv_head=3
        ):
            raise ValueError("Head selection differs")
        if (
            record["attention_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
            or record["condition"] != condition
            or record["captures"] != 360
            or record["noise_index"] != 0
            or record["denoise_index"] != 0
            or record["selected_layers"] != ([] if mode == "native" else [1])
            or record["attention_output_returned_unchanged"] is not True
        ):
            raise ValueError("Attention capture differs")
        if fields is None:
            fields = record["summary_fields"]
        elif fields != record["summary_fields"]:
            raise ValueError("Summary field identity differs")
        with np.load(path, allow_pickle=False) as a:
            if set(a.files) != {"image_probability", "summary"}:
                raise ValueError("Attention field set differs")
            image, summary = a["image_probability"], a["summary"]
        if (
            image.shape != (45, 8, 15, 64)
            or summary.shape != (45, 8, 15, 8)
            or not np.isfinite(image).all()
            or not np.isfinite(summary).all()
            or np.any(image < 0)
            or not np.allclose(image.sum(-1), summary[..., 0], atol=1e-6, rtol=0)
            or np.any(summary[..., 3] != 0)
            or np.any(np.abs(summary[..., :4].sum(-1) - 1) > 0.004)
        ):
            raise ValueError("Attention probability contract differs")
        data[condition] = image, summary
    native = data["base640_native"]
    if not all(np.array_equal(a, b) for a, b in zip(native, data["base640_krestore"], strict=True)):
        raise ValueError("Restore attention differs")
    exact = {
        name: subset_exact(native, data["base640_" + name], hs) for name, hs in SUBSETS.items()
    }
    calibration = json.loads((source / "base640_native/calibration.json").read_text())
    train = calibration["training_rows"]
    views = {"train40": train, "dev5": [r for r in range(45) if r not in train]}
    observations = []
    for condition in CONDITIONS[2:]:
        image, summary = data[condition]
        pair = probability_change(native[0], image)
        for view, rows in views.items():
            for index, layer in enumerate(range(1, 16, 2)):
                for head in range(15):
                    observations.append(dict(
                        condition=condition, view=view, layer=layer, query_head=head,
                        kv_head=head // 3,
                        native_summary=native[1][rows, index, head].mean(0).tolist(),
                        changed_summary=summary[rows, index, head].mean(0).tolist(),
                        **{k: float(v[rows, index, head].mean()) for k, v in pair.items()},
                    ))

    def errors(condition):
        return result["metrics"][condition]["standard"]["dev5"]["full"]["left_joint"][
            "errors"
        ]["model"]

    primary = qualify(
        errors("base640_native"), errors("base640_kdirection"), errors("base640_kh12")
    )
    value = dict(
        primary=primary, unselected_query_heads_exact=exact, restore_attention_exact=True,
        attention_effects=observations, summary_fields=fields,
        observations_only_first_noise_first_denoise=True, local_model_forwards=0,
        actual_joint_intervention_measured=True,
    )
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in value.items() if k != "attention_effects"}))


if __name__ == "__main__":
    main()
