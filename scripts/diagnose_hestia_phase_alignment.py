"""Compare clock and target-event coordinates with identical geometric donor sets."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from scripts.diagnose_hestia_checkpoint_localization import (
    file_hash,
    read_json,
    require,
    split_rows,
)
from scripts.diagnose_hestia_scene_phase import (
    first_upcross,
    geometry_neighbors,
    trajectory_scores,
)


def aligned_targets(target, anchors, offsets):
    y = np.asarray(target, dtype=np.float64)
    require(y.ndim == 3 and np.isfinite(y).all(), "Finite [scene,time,dimension] target required")
    require(
        len(anchors) == len(y) and all(a is not None and int(a) == a for a in anchors),
        "Measured integer event times required; censored events cannot be filled",
    )
    offsets = np.asarray(offsets)
    require(
        offsets.ndim == 1 and offsets.size > 0 and np.issubdtype(offsets.dtype, np.integer),
        "Nonempty integer offset window required",
    )
    indices = np.asarray(anchors, dtype=int)[:, None] + offsets[None, :]
    require(indices.min() >= 0 and indices.max() < y.shape[1], "Aligned window outside saved data")
    result = np.stack([y[i, index] for i, index in enumerate(indices)])
    require(result.shape == (len(y), len(offsets), y.shape[-1]), "Aligned axes differ")
    return result, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    plan = read_json(args.plan)
    require(
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") == plan["image"], "Registered image required"
    )
    for name, sha in plan["sha256"].items():
        require(file_hash(name) == sha, f"Source drift: {name}")
    prior = read_json(plan["source_result"])
    require(
        prior["status"] == "completed" and prior["array_reload_exact"], "Completed source required"
    )
    require(file_hash(plan["source_arrays"]) == prior["array_sha256"], "Source array drift")
    meta = read_json(plan["metadata"])["metadata"]
    rows = split_rows(meta)
    with np.load(plan["source_arrays"], allow_pickle=False) as a:
        y, x, recorded, episodes = (
            a[k] for k in ("projected_targets", "coordinates", "neighbors", "episodes")
        )
    require(episodes.tolist() == meta["episodes"], "Episode order differs")
    _, nearest, allowed = geometry_neighbors(x, meta["episodes"], rows["train40"], plan["k"])
    require(np.array_equal(nearest, recorded), "Donor choices changed")
    groups = prior["groups"]
    anchors = first_upcross(y[:, :, groups["left_gripper"][0]], plan["threshold"])
    offsets = list(range(*plan["offsets"]))
    result, arrays = {}, {}
    for name, reference in (("clock", [plan["clock_anchor"]] * len(y)), ("event_oracle", anchors)):
        target, indices = aligned_targets(y, reference, offsets)
        metrics, predictions = trajectory_scores(
            target, nearest, allowed, groups, rows, {"window": [0, len(offsets)]}
        )
        result[name] = {view: values["window"] for view, values in metrics.items()}
        for values in result[name].values():
            for scores in values.values():
                scores["neighbor_to_best_constant"] = scores["neighbor_mae"] / min(
                    scores["mean_mae"], scores["median_mae"]
                )
        arrays[name + "_targets"] = target
        arrays[name + "_predictions"] = predictions
        arrays[name + "_source_indices"] = indices
    output = Path(plan["output"])
    require(output.is_dir() and not any(output.iterdir()), "New empty output directory required")
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as restored:
        require(
            all(np.array_equal(a, restored[k]) for k, a in arrays.items()), "Derived reload differs"
        )
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(
        elapsed < plan["maximum_seconds"] and rss < plan["maximum_rss_bytes"], "Budget exceeded"
    )
    record = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": file_hash(args.plan),
        "metrics": result,
        "array_sha256": file_hash(output / "arrays.npz"),
        "array_reload_exact": True,
        "seconds": elapsed,
        "peak_rss_bytes": rss,
        "true_query_event_time_used": True,
        "deployable_predictor": False,
        "new_model_forwards": 0,
        "policy_optimizer_steps": 0,
        "new_raw_rows_materialized": 0,
        "hidden_loaded": False,
        "m2_complete": False,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
