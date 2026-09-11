"""Compare a fixed KV full-chunk readout with saved native policy outputs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from rosetta_reality.vla.contextual_kv_probe import _metrics
from scripts.diagnose_kv_regularization import compare, digest
from scripts.diagnose_socket_action_horizon import read_registered_rows


def flattened_groups(groups, horizon, width):
    return {
        name: [t * width + d for t in range(horizon) for d in dims] for name, dims in groups.items()
    }


def chunk_metrics(pred, target, reference, groups, start, stop):
    n, horizon, width = target.shape
    if pred.shape != target.shape or not 0 <= start < stop <= horizon:
        raise ValueError("Invalid chunk shape/window.")
    local_groups = flattened_groups(groups, stop - start, width)
    return _metrics(
        pred[:, start:stop].reshape(n, -1),
        target[:, start:stop].reshape(n, -1),
        reference[:, start:stop].reshape(len(reference), -1),
        local_groups,
    )


def project_targets(raw, contract, contract_sha):
    """Use the actual registered training projection, including overshoot guards."""
    import torch
    from lerobot.lerobot_types import TransitionKey

    from rosetta_reality.vla.processor import ActionContractProjectionProcessorStep

    step = ActionContractProjectionProcessorStep.from_contract(
        contract, action_contract_sha256=contract_sha
    )
    tensor = torch.as_tensor(raw, dtype=torch.float64)
    projected = step({TransitionKey.ACTION: tensor})[TransitionKey.ACTION].numpy()
    independent = np.clip(raw, step.lower_bounds, step.upper_bounds)
    if not np.array_equal(projected, independent):
        raise ValueError("Native training projection and independent bounds disagree.")
    return projected, {
        "changed_elements_by_dimension": (projected != raw).sum(axis=(0, 1)).tolist(),
        "first_action_changed_by_dimension": (projected[:, 0] != raw[:, 0]).sum(0).tolist(),
        "max_abs_change_by_dimension": np.abs(projected - raw).max(axis=(0, 1)).tolist(),
        "raw_min_by_dimension": raw.min(axis=(0, 1)).tolist(),
        "raw_max_by_dimension": raw.max(axis=(0, 1)).tolist(),
        "native_projection_parity_exact": True,
    }


def analyze(features, targets, native, groups, *, alphas, seed, train_count):
    n, horizon, width = targets.shape
    if native.shape != targets.shape or not np.isfinite(native).all():
        raise ValueError("Native output identity/finite check failed.")
    result, arrays = compare(
        features,
        targets.reshape(n, -1),
        train_count=train_count,
        groups=flattened_groups(groups, horizon, width),
        grids={"ridge": alphas},
        seed=seed,
    )
    prediction = arrays["ridge_predictions"].reshape(targets.shape)
    reference = targets[:train_count]
    windows = {
        "full": [0, horizon],
        "first": [0, 1],
        "early": [0, 10],
        "middle": [10, 25],
        "late": [25, 50],
        "last": [49, 50],
    }
    windows_report = {}
    for name, (start, stop) in windows.items():
        windows_report[name] = {
            kind: chunk_metrics(
                pred[train_count:], targets[train_count:], reference, groups, start, stop
            )
            for kind, pred in (("ridge", prediction), ("native", native))
        }
    curves = {
        kind: {group: {"mae": [], "best_constant_mae": [], "image_gain": []} for group in groups}
        for kind in ("ridge", "native")
    }
    for t in range(horizon):
        for kind, pred in (("ridge", prediction), ("native", native)):
            metrics = chunk_metrics(
                pred[train_count:], targets[train_count:], reference, groups, t, t + 1
            )
            for group, m in metrics.items():
                curves[kind][group]["mae"].append(m["mae"])
                curves[kind][group]["best_constant_mae"].append(
                    min(m["train_mean_baseline_mae"], m["train_median_baseline_mae"])
                )
                curves[kind][group]["image_gain"].append(m["paired_mae_gain"])
    criteria = {}
    for group in groups:
        r = windows_report["full"]["ridge"][group]
        p = windows_report["full"]["native"][group]
        oof = result["ridge"]["nested_train_oof"][group]
        criteria[group] = {
            "development_beats_native_and_constants": r["mae"]
            < min(p["mae"], r["train_mean_baseline_mae"], r["train_median_baseline_mae"]) - 1e-8,
            "nested_beats_fold_constants": oof["mae"]
            < min(oof["fold_train_mean_mae"], oof["fold_train_median_mae"]) - 1e-8,
            "development_positive_image_gain": r["paired_mae_gain"] > 1e-8,
        }
    return {
        "readout": result["ridge"],
        "development_windows": windows_report,
        "development_per_step": curves,
        "criteria": criteria,
        "full_chunk_readout_gap_supported": all(all(c.values()) for c in criteria.values()),
    }, arrays


def main():
    from rosetta_reality.data import resolve_prepared_cache
    from rosetta_reality.data.cache_resolver import ordered_feature_names
    from rosetta_reality.data.config import load_dataset_config
    from rosetta_reality.sim.action_contract import load_action_contract
    from rosetta_reality.vla.vision_diagnostics import validate_splits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if (
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["runtime"]["image"]
        or os.environ.get("HF_HUB_OFFLINE") != "1"
    ):
        raise ValueError("Registered offline CPU container required.")
    for path, sha in plan["input_and_code_sha256"].items():
        if digest(path) != sha:
            raise ValueError(f"Identity drift: {path}")
    train, dev, hidden = (
        plan[key] for key in ("train_episodes", "development_episodes", "hidden_episodes")
    )
    validate_splits(train, dev, hidden)
    episodes = train + dev
    contract = load_action_contract(Path(plan["action_contract"]))
    if contract.chunk_length != 50 or contract.dimension != 14 or contract.frequency_hz != 50:
        raise ValueError("Registered temporal/dimension contract differs.")
    cfg = load_dataset_config(Path("configs/data/aloha_sim_insertion_m2.yaml"))
    root, _ = resolve_prepared_cache(cfg, Path.cwd(), validate_checksums=True)
    if digest(root / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest drift.")
    if ordered_feature_names(root, cfg.fields.action) != contract.dimension_names:
        raise ValueError("Ordered action semantics differ.")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = read_registered_rows(root, cfg, episodes, hidden, list(range(50)))
    raw = np.asarray([[rows[(ep, t)][cfg.fields.action] for t in range(50)] for ep in episodes])
    states0 = np.asarray([rows[(ep, 0)][cfg.fields.state] for ep in episodes])
    if not np.array_equal(states0, np.broadcast_to(states0[0], states0.shape)):
        raise ValueError("Frame-zero state is not constant across scenes.")
    with np.load(plan["features"], allow_pickle=False) as saved:
        if (
            saved["episodes"].tolist() != episodes
            or not np.array_equal(saved["actions"], raw[:, 0])
            or saved["standard_outputs"].shape != (45, 1, 50, 14)
        ):
            raise ValueError("Saved native chunk/label identity differs.")
        features, native = saved["early"].copy(), saved["standard_outputs"][:, 0].copy()
    targets, projection = project_targets(raw, contract, digest(plan["action_contract"]))
    groups = {}
    for index, dimension in enumerate(contract.dimensions):
        groups.setdefault(dimension.unit, []).append(index)
    result, arrays = analyze(
        features,
        targets,
        native,
        groups,
        alphas=plan["alphas"],
        seed=plan["seed"],
        train_count=len(train),
    )
    arrays.update(raw_targets=raw, projected_targets=targets, native_standard=native)
    if time.monotonic() - started > plan["runtime"]["maximum_seconds"]:
        raise TimeoutError("Registered analysis time budget exceeded.")
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as saved:
        if not all(np.array_equal(v, saved[k]) for k, v in arrays.items()):
            raise ValueError("Saved arrays failed exact reload.")
    report = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": digest(args.plan),
        "analysis": result,
        "target_projection": projection,
        "rows_materialized": len(rows),
        "array_sha256": digest(output / "arrays.npz"),
        "array_reload_exact": True,
        "elapsed_seconds": time.monotonic() - started,
        "new_model_forwards": 0,
        "future_states_used": False,
        "hidden_rows_materialized": False,
        "policy_improvement": "not measured",
        "task_success": "not measured",
        "m2_complete": False,
    }
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "completed",
                "criteria": result["criteria"],
                "alpha": result["readout"]["alpha"],
                "seconds": report["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
