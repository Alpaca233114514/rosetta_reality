"""Measure nonadditivity of already observed head effects without a model."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def compare_changes(full, singles):
    if singles.shape[1:] != full.shape or singles.shape[0] != 5:
        raise ValueError("Five aligned single-head deltas required")
    if not np.isfinite(full).all() or not np.isfinite(singles).all():
        raise ValueError("Nonfinite deltas")
    actual = full.reshape(full.shape[0], -1).astype(np.float64)
    summed = singles.astype(np.float64).sum(0).reshape(actual.shape)
    residual = actual - summed
    energy = np.square(actual).mean(1)
    residual_energy = np.square(residual).mean(1)
    summed_energy = np.square(summed).mean(1)
    dot = (actual * summed).mean(1)
    identity = energy + summed_energy - 2 * dot
    if not np.allclose(residual_energy, identity, atol=1e-12, rtol=1e-12):
        raise ValueError("Residual identity failed")
    ratios, cosines = [], []
    for e, s, r, d in zip(energy, summed_energy, residual_energy, dot, strict=True):
        ratios.append(float(r / e) if e > 0 else None)
        cosines.append(float(d / np.sqrt(e * s)) if e > 0 and s > 0 else None)
    return dict(
        all_head_delta_energy=energy.tolist(),
        additive_delta_energy=summed_energy.tolist(),
        residual_energy=residual_energy.tolist(),
        residual_fraction=ratios,
        cosine=cosines,
        approximately_additive=all(
            r is not None and c is not None and r <= 0.10 and c >= 0.95
            for r, c in zip(ratios, cosines, strict=True)
        ),
    )


def main():
    plan_path = Path(sys.argv[1])
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    for name, expected in plan["sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError("Input SHA differs")
    source = Path(plan["source"])
    manifest = json.loads((source / "handoff-manifest.json").read_text())
    meta_path = Path("/source") / plan["metadata"]
    if hashlib.sha256(meta_path.read_bytes()).hexdigest() != plan["source_metadata_sha256"]:
        raise ValueError("Metadata SHA differs")
    meta = json.loads(meta_path.read_text())["metadata"]
    views = {
        name: [meta["episodes"].index(ep) for ep in meta["views"][name]]
        for name in ("train40", "dev5")
    }
    groups = {}
    for unit, kind in (("radian", "joint"), ("normalized", "gripper")):
        groups[kind] = [i for i, d in enumerate(meta["dimensions"]) if d["unit"] == unit]
        for side in ("left", "right"):
            groups[f"{side}_{kind}"] = [
                i for i in groups[kind] if meta["dimensions"][i]["name"].startswith(side + "_")
            ]
    conditions = ["native", "kdirection"] + [f"kh{h}" for h in range(5)]
    arrays = {}
    for condition in conditions:
        name = f"base640_{condition}/arrays.npz"
        path = source / name
        entry = manifest["files"][name]
        if path.stat().st_size != entry["bytes"] or (
            hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
        ):
            raise ValueError("Array SHA or size differs")
        with np.load(path, allow_pickle=False) as saved:
            arrays[condition] = {k: saved[k].astype(np.float64) for k in saved.files}
        for key in ("noise", "standard_targets", "normalized_targets"):
            if not np.array_equal(arrays[condition][key], arrays["native"][key]):
                raise ValueError("Unaligned targets or noise")
    results = []
    for space in ("standard", "normalized"):
        for view, rows in views.items():
            for window, (start, stop) in {"first": (0, 1), "full": (0, 50)}.items():
                for group, dims in groups.items():
                    predictions = {
                        c: a[space + "_predictions"][:, rows, start:stop][..., dims]
                        for c, a in arrays.items()
                    }
                    native = predictions["native"]
                    target = arrays["native"][space + "_targets"][rows, start:stop][..., dims]
                    full = predictions["kdirection"] - native
                    singles = np.stack([predictions[f"kh{h}"] - native for h in range(5)])
                    proxy = native + singles[1] + singles[2]
                    axis = (1, 2, 3)
                    results.append(dict(
                        space=space, view=view, window=window, group=group,
                        **compare_changes(full, singles),
                        head12_additive_proxy_mse=np.square(proxy - target).mean(axis).tolist(),
                        head12_additive_proxy_mae=np.abs(proxy - target).mean(axis).tolist(),
                        proxy_gap_to_all_head_energy=np.square(proxy - native - full)
                        .mean(axis).tolist(),
                        proxy_is_measured_joint_intervention=False,
                    ))
    primary = next(r for r in results if all(
        r[k] == v for k, v in dict(
            space="standard", view="dev5", window="full", group="left_joint"
        ).items()
    ))
    result = dict(
        id=plan["id"], plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        primary=primary, results=results, model_forwards=0, optimizer_steps=0,
        hidden_loaded=False, real_head12_intervention="not measured",
    )
    with Path(plan["output"]).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))


if __name__ == "__main__":
    main()
