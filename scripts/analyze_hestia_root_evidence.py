"""Saved-array K/V factorial audit; no policy, raw data, or optimizer imports."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

PLAN = Path("reports/training/m2-smolvla-hestia-root-evidence-plan-002-2026-09-13.json")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] > 1 and np.isfinite(x).all(), "Invalid metric")
    return {
        "mean": float(x.mean()),
        "by_noise": x.mean(1).tolist(),
        "by_episode": x.mean(0).tolist(),
        "by_noise_episode": x.tolist(),
        "leave_one_episode_out_by_noise": ((x.sum(1)[:, None] - x) / (x.shape[1] - 1)).tolist(),
    }


def factorial(e00, e10, e01, e11):
    """Positive values mean harm from changing old K/V to new at fixed remainder."""
    a, b, c, d = [np.asarray(x, dtype=np.float64) for x in (e00, e10, e01, e11)]
    require(all(x.shape == a.shape for x in (b, c, d)), "Unpaired factorial")
    k_old, k_new = b - a, d - c
    v_old, v_new = c - a, d - b
    interaction = d - b - c + a
    k_average, v_average = (k_old + k_new) / 2, (v_old + v_new) / 2
    require(np.allclose(k_average + v_average, d - a, atol=1e-12), "Factorial closure")
    return {
        name: summarize(x)
        for name, x in {
            "k_given_old_v": k_old,
            "k_given_new_v": k_new,
            "v_given_old_k": v_old,
            "v_given_new_k": v_new,
            "interaction": interaction,
            "k_order_average": k_average,
            "v_order_average": v_average,
            "joint_effect": d - a,
            "v_minus_k_order_average": v_average - k_average,
        }.items()
    }


def positive_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def load(plan):
    for name, expected in plan["sha256"].items():
        require(sha(name) == expected, f"Identity drift: {name}")
    arrays, checks = {}, 0
    for source in plan["sources"]:
        root = Path(source["root"])
        manifest = read(root / "handoff-manifest.json")
        for name, entry in manifest["files"].items():
            path = root / name
            require(
                path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(),
                "Path escape",
            )
            require(
                path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                "Bundle drift",
            )
            checks += 1
        worker = read(root / "worker-exited.json")
        require(not worker["error"] and worker["optimizer_steps"] == 0, "Failed source worker")
        for condition in source["conditions"]:
            require(condition in worker["completed_conditions"], "Incomplete source")
            report = read(root / condition / "result.json")
            require(
                report["condition"] == condition
                and report["all_parameters_unchanged"]
                and report["raw_images_exact"],
                "Condition integrity",
            )
            require(
                report["optimizer_steps"] == 0
                and report["model_forwards"] == 180
                and not report["hidden_loaded"],
                "Source scope",
            )
            with np.load(root / condition / "arrays.npz", allow_pickle=False) as z:
                current = {key: z[key] for key in z.files}
            require(all(np.isfinite(x).all() for x in current.values()), "Nonfinite source")
            if condition in arrays:
                require(set(current) == set(arrays[condition]), "Array key drift")
                require(
                    all(np.array_equal(x, arrays[condition][k]) for k, x in current.items()),
                    "Cross-run endpoint drift",
                )
            arrays[condition] = current
    base = arrays["base640"]
    for saved in arrays.values():
        for key in ("noise", "standard_targets", "normalized_targets"):
            require(np.array_equal(saved[key], base[key]), "Pairing drift")
    meta = read(plan["metadata"])["metadata"]
    require(meta["hidden_test_loaded"] is False, "Metadata hidden access")
    episodes = meta["episodes"]
    require(len(episodes) == 45 and len(set(episodes)) == 45, "Episode drift")
    require(not set(episodes) & set(plan["hidden_episodes"]), "Hidden episode")
    rows = {view: [episodes.index(e) for e in meta["views"][view]] for view in ("train40", "dev5")}
    require(len(rows["train40"]) == 40 and len(rows["dev5"]) == 5, "Split counts")
    require(not set(rows["train40"]) & set(rows["dev5"]), "Split overlap")
    for saved in arrays.values():
        for space in ("standard", "normalized"):
            require(
                saved[space + "_predictions"].shape == (4, 45, 50, len(meta["dimensions"])),
                "Prediction layout",
            )
            require(
                saved[space + "_targets"].shape == (45, 50, len(meta["dimensions"])),
                "Target layout",
            )
    return arrays, meta, rows, checks


def main():
    started = time.monotonic()
    plan = read(PLAN)
    require(os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") == plan["image"], "Container required")
    require(os.environ.get("HF_HUB_OFFLINE") == "1", "Offline required")
    output = Path(plan["output"]) / "result.json"
    require(not output.exists(), "Preserve result")
    arrays, meta, rows, file_checks = load(plan)
    groups = {}
    for unit, kind in (("radian", "joint"), ("normalized", "gripper")):
        groups[kind] = [i for i, d in enumerate(meta["dimensions"]) if d["unit"] == unit]
        for side in ("left", "right"):
            groups[side + "_" + kind] = [
                i for i in groups[kind] if meta["dimensions"][i]["name"].startswith(side + "_")
            ]
    previous = read(plan["early_result"])
    result = {
        "id": plan["id"],
        "plan_sha256": sha(PLAN),
        "source_files_verified": file_checks,
        "historical_metric_checks": 0,
        "groups": groups,
        "episodes": meta["views"],
        "noise_conditions": meta["noise_conditions"],
        "metrics": {},
        "model_forwards": 0,
        "optimizer_steps": 0,
        "hidden_loaded": False,
        "gate3": "not measured",
        "gate4": "not measured",
        "m2_complete": False,
        "interpretation": (
            "Order averages allocate interaction equally; they are not unique root-cause shares. "
            "Leave-one-out is descriptive, not independent confirmation."
        ),
    }
    for space in ("standard", "normalized"):
        for view, indices in rows.items():
            for window, (start, stop) in plan["windows"].items():
                for group, dims in groups.items():
                    y_all = arrays["base640"][space + "_targets"].astype(np.float64)[
                        :, start:stop, dims
                    ]
                    y = y_all[indices]
                    train = y_all[rows["train40"]]
                    predictions = {
                        k: v[space + "_predictions"].astype(np.float64)[:, indices, start:stop][
                            ..., dims
                        ]
                        for k, v in arrays.items()
                    }
                    for metric in ("mae", "mse"):

                        def error(p):
                            residual = p - y
                            return (np.abs(residual) if metric == "mae" else residual**2).mean(
                                (-2, -1)
                            )

                        values = {k: error(p) for k, p in predictions.items()}
                        constants = {
                            "mean": error(train.mean(0)),
                            "median": error(np.median(train, axis=0)),
                        }
                        for step in (640, 1280):
                            expected = previous["metrics"][str(step)][space][view][window][group][
                                "errors"
                            ]["model"][metric]["by_noise"]
                            require(
                                np.allclose(
                                    values[f"base{step}"].mean(1), expected, rtol=1e-12, atol=1e-12
                                ),
                                "Historical metric drift",
                            )
                            result["historical_metric_checks"] += 1
                        factors = {}
                        for base in (640, 1280):
                            names = (
                                ["base640", "base640_k1280", "base640_v1280", "base640_kv1280"]
                                if base == 640
                                else [
                                    "base1280_kv640",
                                    "base1280_v640",
                                    "base1280_k640",
                                    "base1280",
                                ]
                            )
                            factors[str(base)] = factorial(*(values[k] for k in names))
                        delta = values["base1280"] - values["base640"]
                        baseline = min(float(c.mean()) for c in constants.values())
                        gap640 = float(values["base640"].mean()) - baseline
                        gap1280 = float(values["base1280"].mean()) - baseline
                        key = "/".join((space, view, window, group, metric))
                        result["metrics"][key] = {
                            "conditions": {k: summarize(v) for k, v in values.items()},
                            "constants": {k: float(v.mean()) for k, v in constants.items()},
                            "factorial_fixed_remainder": factors,
                            "late_regression": summarize(delta),
                            "gap640": gap640,
                            "gap1280": gap1280,
                            "early_gap_fraction_of_final_gap": positive_ratio(gap640, gap1280),
                            "rollback_fraction_of_final_gap": {
                                kind: positive_ratio(
                                    float(
                                        (values["base1280"] - values[f"base1280_{kind}640"]).mean()
                                    ),
                                    gap1280,
                                )
                                for kind in ("k", "v", "kv")
                            },
                            "conditions_beating_both_constants_by_noise": {
                                k: (
                                    (v.mean(1) < constants["mean"].mean())
                                    & (v.mean(1) < constants["median"].mean())
                                ).tolist()
                                for k, v in values.items()
                            },
                        }
    result["seconds"] = time.monotonic() - started
    result["maximum_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(result["seconds"] < plan["maximum_seconds"], "Deadline")
    require(result["maximum_rss_bytes"] < plan["maximum_rss_bytes"], "Memory")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ("metrics", "episodes")}))


if __name__ == "__main__":
    main()
