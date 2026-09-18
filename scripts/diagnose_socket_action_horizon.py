"""Compare first/end-of-chunk left-action readouts from observed socket position."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.diagnose_kv_regularization import compare, digest


def position_basis(positions):
    """Fixed pixel normalization and degree-two basis; no target-dependent fit."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all():
        raise ValueError("Expected finite [scene, xy] pixel positions.")
    x, y = positions[:, 0] / 640, positions[:, 1] / 480
    return np.stack((x, y, x * x, x * y, y * y), axis=1)


def read_registered_rows(root, cfg, episodes, hidden, offsets):
    import pyarrow.dataset as arrow

    if not episodes or len(episodes) != len(set(episodes)) or set(episodes) & set(hidden):
        raise ValueError("Invalid or hidden episode request.")
    f = cfg.fields
    columns = [f.episode_index, f.frame_index, f.timestamp, f.action, f.state]
    table = arrow.dataset(root / "data", format="parquet").to_table(
        columns=columns,
        filter=arrow.field(f.episode_index).isin(episodes)
        & arrow.field(f.frame_index).isin(offsets),
    )
    rows = table.to_pylist()
    mapping = {(int(row[f.episode_index]), int(row[f.frame_index])): row for row in rows}
    wanted = {(ep, t) for ep in episodes for t in offsets}
    if set(mapping) != wanted or len(rows) != len(wanted):
        raise ValueError("Expected one row per registered episode and offset.")
    for (ep, t), row in mapping.items():
        if abs(row[f.timestamp] - t / 50) > 1e-6:
            raise ValueError("Frame/timestamp alignment differs from 50 Hz.")
        if not all(np.isfinite(row[k]).all() for k in (f.action, f.state)):
            raise ValueError("Nonfinite action/state.")
    return mapping


def main():
    from rosetta_reality.vla.vision_diagnostics import load_frame_zero_context

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if (
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["runtime"]["image"]
        or os.environ.get("HF_HUB_OFFLINE") != "1"
    ):
        raise ValueError("Registered offline container required.")
    for path, sha in plan["input_and_code_sha256"].items():
        if digest(path) != sha:
            raise ValueError(f"Identity drift: {path}")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    context = load_frame_zero_context(Path.cwd(), "non_hidden")
    if context["episodes"] != plan["episodes"]:
        raise ValueError("Episode identity mismatch.")
    if digest(context["root"] / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest drift.")
    with np.load(plan["position_arrays"], allow_pickle=False) as saved:
        positions = saved["targets"].copy()
    position_record = json.loads(Path(plan["position_result"]).read_text())
    if [row["episode"] for row in position_record["extraction"]] != context["episodes"]:
        raise ValueError("Position row order mismatch.")
    features = position_basis(positions)
    cfg = context["config"]
    rows = read_registered_rows(
        context["root"], cfg, context["episodes"], plan["hidden_episodes"], plan["offsets"]
    )
    results, arrays = {}, {"positions": positions}
    for offset in plan["offsets"]:
        actions = np.asarray([rows[(ep, offset)][cfg.fields.action] for ep in context["episodes"]])
        if offset == 0 and not np.array_equal(actions, context["actions"]):
            raise ValueError("Original frame-zero action identity failed.")
        results[str(offset)], predictions = compare(
            features,
            actions[:, :7],
            train_count=40,
            groups={"left_joint_radian": list(range(6)), "left_gripper_normalized": [6]},
            grids={"position_only": plan["alphas"]},
            seed=plan["seed"],
        )
        arrays[f"actions_{offset}"] = actions
        arrays.update({f"h{offset}_{key}": value for key, value in predictions.items()})
    ratios = {}
    for offset in plan["offsets"]:
        arm = results[str(offset)]["position_only"]
        dev = arm["development"]["left_joint_radian"]
        nested = arm["nested_train_oof"]["left_joint_radian"]
        ratios[str(offset)] = {
            "development_to_best_constant": dev["mae"]
            / min(dev["train_mean_baseline_mae"], dev["train_median_baseline_mae"]),
            "nested_to_best_constant": nested["mae"]
            / min(nested["fold_train_mean_mae"], nested["fold_train_median_mae"]),
            "development_image_gain": dev["paired_mae_gain"],
        }
    late, early = ratios["49"], ratios["0"]
    criteria = {
        "late_development_ratio_below_0_8": late["development_to_best_constant"] < 0.8,
        "late_nested_ratio_below_0_8": late["nested_to_best_constant"] < 0.8,
        "late_ratio_gain_at_least_0_2": (
            late["development_to_best_constant"] < early["development_to_best_constant"] - 0.2
            and late["nested_to_best_constant"] < early["nested_to_best_constant"] - 0.2
        ),
        "late_positive_image_gain": late["development_image_gain"] > 1e-8,
    }
    elapsed = time.monotonic() - started
    if elapsed > plan["runtime"]["maximum_seconds"]:
        raise TimeoutError("Budget exceeded.")
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as saved:
        assert all(np.array_equal(v, saved[k]) for k, v in arrays.items())
    report = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": digest(args.plan),
        "arms": results,
        "joint_ratios": ratios,
        "criteria": criteria,
        "late_target_more_readable": all(criteria.values()),
        "elapsed_seconds": elapsed,
        "rows_materialized": len(rows),
        "hidden_rows_materialized": False,
        "future_states_used_as_inputs": False,
        "new_model_forwards": 0,
        "array_reload_exact": True,
        "array_sha256": digest(output / "arrays.npz"),
        "task_success": "not measured",
        "policy_improvement": "not measured",
        "m2_complete": False,
    }
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "completed", "ratios": ratios, "criteria": criteria}))


if __name__ == "__main__":
    main()
