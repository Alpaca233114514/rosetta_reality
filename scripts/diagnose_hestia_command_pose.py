"""Measure commanded ALOHA end-effector poses; never step physics or fit a policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
from scripts.diagnose_hestia_scene_phase import geometry_neighbors


def runtime_identity():
    from gym_aloha.constants import ASSETS_DIR

    assets = Path(ASSETS_DIR)
    files = {
        p.relative_to(assets).as_posix(): file_hash(p)
        for p in sorted(assets.rglob("*"))
        if p.is_file()
    }
    require("bimanual_viperx_insertion.xml" in files, "Native insertion assets missing")
    return {
        "versions": {
            name: importlib.metadata.version(name) for name in ("gym-aloha", "mujoco", "numpy")
        },
        "asset_sha256": files,
        "asset_tree_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }


def rotation_error(predicted, target):
    """SO(3) geodesic distance, in radians; no Euler-angle subtraction."""
    a, b = np.asarray(predicted), np.asarray(target)
    require(a.shape == b.shape and a.shape[-2:] == (3, 3), "Matching rotation matrices required")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite rotation")
    relative = np.swapaxes(a, -1, -2) @ b
    sine = (
        np.linalg.norm(
            np.stack(
                (
                    relative[..., 2, 1] - relative[..., 1, 2],
                    relative[..., 0, 2] - relative[..., 2, 0],
                    relative[..., 1, 0] - relative[..., 0, 1],
                ),
                axis=-1,
            ),
            axis=-1,
        )
        / 2
    )
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.arctan2(sine, cosine)


def rotation_mean(values):
    """Equal-weight chordal mean projected onto SO(3), along donor axis zero."""
    matrix = np.asarray(values).mean(axis=0)
    u, singular, vt = np.linalg.svd(matrix)
    require(np.min(singular) > 1e-8, "Degenerate rotation mean; cannot invent an orientation")
    correction = np.broadcast_to(np.eye(3), matrix.shape).copy()
    correction[..., 2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


class CommandKinematics:
    def __init__(self, dimensions):
        import mujoco
        from gym_aloha.constants import ASSETS_DIR

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(
            str(Path(ASSETS_DIR) / "bimanual_viperx_insertion.xml")
        )
        self.data = mujoco.MjData(self.model)
        self.dimensions = dimensions
        self.columns, self.addresses, self.mapping = [], [], []
        for column, dim in enumerate(dimensions):
            if dim["unit"] != "radian":
                require(
                    dim["name"] in ("left_gripper", "right_gripper"), "Unknown non-arm dimension"
                )
                continue
            arm, suffix = dim["name"].split("_", 1)
            joint = self.model.joint(f"vx300s_{arm}/{suffix}")
            joint_id = int(joint.id)
            require(
                self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE, "Expected hinge joint"
            )
            require(bool(self.model.jnt_limited[joint_id]), "Unlimited arm joint")
            require(
                np.allclose(
                    self.model.jnt_range[joint_id],
                    [dim["minimum"], dim["maximum"]],
                    atol=1e-5,
                    rtol=0,
                ),
                "Contract/native joint limits differ",
            )
            address = int(self.model.jnt_qposadr[joint_id])
            self.columns.append(column)
            self.addresses.append(address)
            self.mapping.append(
                {"action": dim["name"], "joint": joint.name, "qpos_address": address}
            )
        require(
            len(self.addresses) == 12 and len(set(self.addresses)) == 12,
            "Two distinct six-joint chains required",
        )
        self.bodies = [
            int(self.model.body(f"vx300s_{arm}/gripper_link").id) for arm in ("left", "right")
        ]
        self.sanity = self.check_sanity()

    def poses(self, actions):
        a = np.asarray(actions, dtype=np.float64)
        require(
            a.shape[-1] == len(self.dimensions) and np.isfinite(a).all(), "Invalid action array"
        )
        lower = np.asarray([self.dimensions[c]["minimum"] for c in self.columns])
        upper = np.asarray([self.dimensions[c]["maximum"] for c in self.columns])
        arm = a[..., self.columns]
        require(
            np.all(arm >= lower - 1e-6) and np.all(arm <= upper + 1e-6),
            "Arm command outside contract; no silent clipping",
        )
        shape = a.shape[:-1]
        positions = np.empty((arm.reshape(-1, len(self.columns)).shape[0], 2, 3))
        rotations = np.empty(positions.shape[:-1] + (3, 3))
        for i, row in enumerate(arm.reshape(-1, len(self.columns))):
            self.data.qpos[:] = self.model.qpos0
            self.data.qpos[self.addresses] = row
            self.mj.mj_kinematics(self.model, self.data)
            positions[i] = self.data.xpos[self.bodies]
            rotations[i] = self.data.xmat[self.bodies].reshape(2, 3, 3)
        require(np.isfinite(positions).all() and np.isfinite(rotations).all(), "Nonfinite FK")
        require(np.allclose(np.linalg.det(rotations), 1, atol=1e-12, rtol=0), "Invalid FK rotation")
        return positions.reshape(shape + (2, 3)), rotations.reshape(shape + (2, 3, 3))

    def check_sanity(self):
        # The palm frame is upstream of the finger joints. Assert this structurally.
        ancestry = []
        for body in self.bodies:
            ids = []
            while body:
                ids.extend(
                    range(
                        int(self.model.body_jntadr[body]),
                        int(self.model.body_jntadr[body] + self.model.body_jntnum[body]),
                    )
                )
                body = int(self.model.body_parentid[body])
            names = [self.model.joint(i).name for i in ids]
            require(
                len(names) == 6 and all("finger" not in name for name in names),
                "Palm depends on non-arm joints",
            )
            ancestry.append(names)
        zero = np.zeros((2, len(self.dimensions)))
        wrist = [d["name"] for d in self.dimensions].index("left_wrist_angle")
        zero[1, wrist] = 0.01
        p, r = self.poses(zero)
        np.testing.assert_allclose(rotation_error(r[0, 0], r[1, 0]), 0.01, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(p[0, 1], p[1, 1])
        np.testing.assert_array_equal(r[0, 1], r[1, 1])
        self.mj.mj_forward(self.model, self.data)
        np.testing.assert_allclose(p[1], self.data.xpos[self.bodies], atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            r[1], self.data.xmat[self.bodies].reshape(2, 3, 3), atol=1e-12, rtol=0
        )
        require(self.data.time == 0, "Physics time must never advance")
        return {
            "palm_ancestry": ancestry,
            "wrist_rotation_sanity_radians": 0.01,
            "other_arm_unchanged": True,
            "kinematics_forward_agree": True,
            "physics_steps": 0,
        }


def errors(predicted, target):
    return np.stack(
        (
            np.linalg.norm(predicted[0] - target[0], axis=-1),
            rotation_error(predicted[1], target[1]),
        ),
        axis=-1,
    )


def summarize(error, rows, episodes):
    result = {}
    for view, indices in rows.items():
        result[view] = {
            arm: {
                metric: {
                    "mean": float(error[indices, :, a, m].mean()),
                    "per_episode": {
                        str(episodes[i]): float(error[i, :, a, m].mean()) for i in indices
                    },
                }
                for m, metric in enumerate(("position_l2_m", "orientation_geodesic_rad"))
            }
            for a, arm in enumerate(("left", "right"))
        }
    return result


def run(plan, plan_path):
    started = time.monotonic()
    require(
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") == plan["image"], "Registered image required"
    )
    for name, sha in plan["sha256"].items():
        require(file_hash(name) == sha, f"Source drift: {name}")
    require(
        runtime_identity() == read_json(plan["runtime_identity"]), "Native assets/runtime drift"
    )
    meta = read_json(plan["metadata"])["metadata"]
    rows = split_rows(meta)
    with np.load(plan["scene_arrays"], allow_pickle=False) as a:
        coordinates, recorded, episodes = (a[k] for k in ("coordinates", "neighbors", "episodes"))
    require(episodes.tolist() == meta["episodes"], "Episode ordering differs")
    distance, nearest, allowed = geometry_neighbors(
        coordinates, meta["episodes"], rows["train40"], plan["k"]
    )
    require(np.array_equal(nearest, recorded), "Donor selection drift")
    fk = CommandKinematics(meta["dimensions"])
    metrics, arrays, pairs = {}, {}, {}
    # Pair order depends only on input geometry, before examining target poses.
    train_pairs = sorted(
        (
            (distance[i, j], i, j)
            for n, i in enumerate(rows["train40"])
            for j in rows["train40"][n + 1 :]
        ),
        key=lambda v: (v[0], meta["episodes"][v[1]], meta["episodes"][v[2]]),
    )[: plan["nearest_pair_count"]]
    with np.load(plan["aligned_arrays"], allow_pickle=False) as source:
        for coordinate in ("clock", "event_oracle"):
            target = source[coordinate + "_targets"]
            require(
                target.shape == (45, 30, len(meta["dimensions"])), "Aligned array shape differs"
            )
            joint_neighbor = target[nearest].mean(1)
            require(
                np.array_equal(joint_neighbor, source[coordinate + "_predictions"]),
                "Historical neighbor output differs",
            )
            true_pose = fk.poses(target)
            poses = {
                "neighbor_joint_mean": fk.poses(joint_neighbor),
                "constant_joint_mean": fk.poses(np.stack([target[d].mean(0) for d in allowed])),
                "constant_joint_median": fk.poses(
                    np.stack([np.median(target[d], axis=0) for d in allowed])
                ),
            }
            for method, donors in (
                ("neighbor_pose_mean", nearest),
                ("constant_pose_mean", allowed),
            ):
                poses[method] = (
                    np.stack([true_pose[0][d].mean(0) for d in donors]),
                    np.stack([rotation_mean(true_pose[1][d]) for d in donors]),
                )
            metrics[coordinate] = {}
            for method, pose in poses.items():
                error = errors(pose, true_pose)
                metrics[coordinate][method] = summarize(error, rows, meta["episodes"])
                arrays[f"{coordinate}_{method}_errors"] = error
            arrays[coordinate + "_target_positions"] = true_pose[0]
            arrays[coordinate + "_target_rotations"] = true_pose[1]
            pairs[coordinate] = []
            for dist, i, j in train_pairs:
                pair_error = errors(
                    tuple(v[i] for v in true_pose), tuple(v[j] for v in true_pose)
                ).mean(0)
                pairs[coordinate].append(
                    {
                        "episodes": [int(episodes[i]), int(episodes[j])],
                        "geometry_distance": float(dist),
                        "position_l2_m_left_right": pair_error[:, 0].tolist(),
                        "orientation_geodesic_rad_left_right": pair_error[:, 1].tolist(),
                    }
                )
    output = Path(plan["output"])
    require(output.is_dir() and not any(output.iterdir()), "Fresh empty output required")
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as restored:
        require(
            set(restored.files) == set(arrays)
            and all(np.array_equal(v, restored[k]) for k, v in arrays.items()),
            "Reload mismatch",
        )
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(
        elapsed < plan["maximum_seconds"] and rss < plan["maximum_rss_bytes"], "Budget exceeded"
    )
    result = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": file_hash(plan_path),
        "array_sha256": file_hash(output / "arrays.npz"),
        "array_reload_exact": True,
        "mapping": fk.mapping,
        "sanity": fk.sanity,
        "metrics": metrics,
        "nearest_train_pairs": pairs,
        "seconds": elapsed,
        "peak_rss_bytes": rss,
        "poses_are_commanded_not_achieved": True,
        "frame": "world",
        "bodies": ["vx300s_left/gripper_link", "vx300s_right/gripper_link"],
        "policy_forwards": 0,
        "optimizer_steps": 0,
        "raw_rows_materialized": 0,
        "hidden_loaded": False,
        "simulation_steps": 0,
        "gate3": "not measured",
        "gate4": "not measured",
        "m2_complete": False,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("metrics", "nearest_train_pairs")
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-runtime", type=Path)
    mode.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.inspect_runtime:
        record = runtime_identity()
        with args.inspect_runtime.open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "versions": record["versions"],
                    "asset_tree_sha256": record["asset_tree_sha256"],
                    "asset_count": len(record["asset_sha256"]),
                }
            )
        )
    else:
        run(read_json(args.plan), args.plan)


if __name__ == "__main__":
    main()
