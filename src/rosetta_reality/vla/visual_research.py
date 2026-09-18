"""Pure, offline evidence arithmetic for paired visual research.

No import-time model/data access. Episode is the statistical unit. All
normalization and comparator fitting exclude development labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def seal_bundle(output, metadata):
    output = Path(output)
    files = {
        p.relative_to(output).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    write_json(output / "manifest.json", {"schema_version": 1, **metadata, "files": files})


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    require(bool(manifest.get("files")), "Empty bundle inventory")
    for name, entry in manifest["files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), "Unsafe bundle path")
        require(
            path.is_file() and sha256(path) == entry["sha256"], "Bundle checksum drift: " + name
        )
    return manifest


def donors(identities, train_episodes, mode="baseline"):
    """Deterministic train-only LOO baselines or split-local cyclic image donors."""
    import numpy as np

    ids = np.asarray(identities)
    require(ids.ndim == 2 and ids.shape[1] == 2 and ids.dtype.kind in "iu", "Invalid identities")
    require(len(set(map(tuple, ids))) == len(ids), "Duplicate sample identity")
    train = np.isin(ids[:, 0], train_episodes)
    result = []
    for i, (episode, frame) in enumerate(ids):
        same_frame = ids[:, 1] == frame
        if mode == "baseline":
            allowed = np.flatnonzero(same_frame & train & (ids[:, 0] != episode))
        elif mode == "cyclic":
            cohort = np.flatnonzero(same_frame & (train == train[i]))
            cohort = sorted(cohort, key=lambda j: int(ids[j, 0]))
            require(len(cohort) > 1, "No nonself image donor")
            allowed = np.array([cohort[(cohort.index(i) + 1) % len(cohort)]])
        else:
            raise ValueError("Unknown donor mode")
        require(len(allowed) > 0, "No train-only comparator at this offset")
        result.append(allowed)
    return result


def decomposition(prediction, target):
    """MSE across scenes = target variance + prediction variance - 2 covariance + bias²."""
    import numpy as np

    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    require(p.shape == y.shape and p.ndim >= 2 and len(p) > 1, "Invalid scene arrays")
    pc, yc = p - p.mean(axis=0), y - y.mean(axis=0)
    parts = {
        "mse": float(np.square(p - y).mean()),
        "prediction_variance": float(np.square(pc).mean()),
        "target_variance": float(np.square(yc).mean()),
        "covariance": float((pc * yc).mean()),
        "bias_squared": float(np.square(p.mean(axis=0) - y.mean(axis=0)).mean()),
    }
    reconstructed = (
        parts["prediction_variance"]
        + parts["target_variance"]
        - 2 * parts["covariance"]
        + parts["bias_squared"]
    )
    require(np.isclose(reconstructed, parts["mse"], atol=1e-12, rtol=1e-10), "MSE identity failed")
    return parts


def episode_summary(values, episodes):
    import numpy as np

    x, ep = np.asarray(values, dtype=np.float64), np.asarray(episodes)
    require(x.ndim == 1 and x.shape == ep.shape and np.isfinite(x).all(), "Invalid episode metric")
    unique = sorted(set(ep.tolist()))
    means = np.array([x[ep == e].mean() for e in unique])
    leave = [float(np.delete(means, i).mean()) for i in range(len(means))] if len(means) > 1 else []
    return {
        "mean": float(means.mean()),
        "episode_count": len(unique),
        "per_episode": dict(zip(map(str, unique), means.tolist(), strict=True)),
        "leave_one_episode_out": leave,
        "leave_one_out_range": [min(leave), max(leave)] if leave else None,
    }


def exposure_strata(ids, seen, reference_seen, metrics, prediction, target):
    """Slice already-paired per-episode metrics; never change image/baseline donors."""
    cohorts = {
        "current_seen": (seen, "checkpoint_specific_membership"),
        "current_unseen": (~seen, "checkpoint_specific_membership"),
    }
    if reference_seen is not None:
        cohorts.update({
            "seen_by_step2500": (reference_seen, "fixed_step2500_membership"),
            "unseen_by_step2500": (~reference_seen, "fixed_step2500_membership"),
        })
    result = {}
    for name, (selected, comparison) in cohorts.items():
        count = int(selected.sum())
        result[name] = {
            "status": "measured" if count else "empty_cohort_not_measured",
            "episode_count": count,
            "identities": ids[selected].tolist(),
            "current_seen_count": int(seen[selected].sum()),
            "training_fit_eligible": bool(count and seen[selected].all()),
            "checkpoint_comparison": comparison,
            "donor_rule": "unchanged_full_split_images_and_train_only_baselines",
            "metrics": {
                metric: episode_summary(values[selected], ids[selected, 0])
                for metric, values in metrics.items()
            } if count else None,
            "decomposition": decomposition(prediction[selected], target[selected])
            if count > 1 else None,
        }
    return result


def analyze_predictions(arrays, train_episodes, groups):
    import numpy as np

    p, wrong, y = (np.asarray(arrays[k], dtype=np.float64) for k in ("correct", "wrong", "targets"))
    ids, valid = arrays["identities"], arrays["valid_mask"]
    require(
        p.ndim == 4 and p.shape == wrong.shape and p.shape[1:] == y.shape, "Prediction axes differ"
    )
    require(valid.shape == y.shape[:2] and valid.dtype == np.bool_, "Padding mask contract differs")
    require(all(np.isfinite(a).all() for a in (p, wrong, y)), "Nonfinite research arrays")
    require(valid.all(), "Main four-offset protocol must not include padded samples")
    references = donors(ids, train_episodes)
    mean = np.stack([y[j].mean(axis=0) for j in references])
    median = np.stack([np.median(y[j], axis=0) for j in references])
    train = np.isin(ids[:, 0], train_episodes)
    seen = arrays.get("input_seen_at_checkpoint")
    if seen is not None:
        seen = np.asarray(seen)
        require(
            seen.shape == train.shape and seen.dtype == np.bool_, "Invalid input exposure flags"
        )
        require(not seen[~train].any(), "Development input appears in training exposure")
    reference_seen = arrays.get("input_seen_by_step2500")
    if reference_seen is not None:
        reference_seen = np.asarray(reference_seen)
        require(
            seen is not None and reference_seen.shape == train.shape
            and reference_seen.dtype == np.bool_, "Invalid fixed exposure cohort"
        )
        require(not reference_seen[~train].any(), "Development input in fixed exposure cohort")
        require(not (reference_seen & ~seen).any(), "Step2500 cohort exceeds current exposure")
    records = []
    for noise in range(len(p)):
        for frame in sorted(set(ids[:, 1].tolist())):
            for split, split_mask in (("train_loo", train), ("development", ~train)):
                ix = np.flatnonzero((ids[:, 1] == frame) & split_mask)
                require(len(ix) > 1, "Insufficient scene cohort")
                for window, horizon in (("first", slice(0, 1)), ("full_chunk", slice(None))):
                    for group, dimensions in groups.items():
                        q, t = (
                            p[noise, ix, horizon][:, :, dimensions],
                            y[ix, horizon][:, :, dimensions],
                        )
                        mean_q = mean[ix, horizon][:, :, dimensions]
                        med_q = median[ix, horizon][:, :, dimensions]
                        correct_mse = np.square(q - t).mean(axis=(1, 2))
                        if frame == 0:
                            # Reuse scene outputs only after nonvisual input and noise parity.
                            require(
                                arrays["frame0_nonvisual_equal"].item() is True,
                                "Frame0 controls absent",
                            )
                            wmse, wmae = [], []
                            for j in range(len(ix)):
                                other = np.delete(q, j, axis=0)
                                wmse.append(np.square(other - t[j]).mean())
                                wmae.append(np.abs(other - t[j]).mean())
                            wmse, wmae = np.array(wmse), np.array(wmae)
                        else:
                            w = wrong[noise, ix, horizon][:, :, dimensions]
                            wmse, wmae = (
                                np.square(w - t).mean(axis=(1, 2)),
                                np.abs(w - t).mean(axis=(1, 2)),
                            )
                        record = {
                            "noise_index": noise,
                            "frame": frame,
                            "split": split,
                            "window": window,
                            "group": group,
                            "mismatch_protocol": "all_nonself"
                            if frame == 0
                            else "cyclic_same_offset",
                            "decomposition": decomposition(q, t),
                            "input_seen_at_checkpoint": None
                            if seen is None
                            else {str(int(ids[i, 0])): bool(seen[i]) for i in ix},
                            "cohort_interpretation": "development"
                            if split == "development"
                            else "training_input_exposure_unverified"
                            if seen is None
                            else "seen_training_inputs"
                            if seen[ix].all()
                            else "training_split_contains_unseen_inputs",
                        }
                        metrics = {
                            "correct_mse": correct_mse,
                            "correct_mae": np.abs(q - t).mean(axis=(1, 2)),
                            "wrong_mse": wmse,
                            "wrong_mae": wmae,
                            "image_gain_mse": wmse - correct_mse,
                            "train_mean_mse": np.square(mean_q - t).mean(axis=(1, 2)),
                            "train_mean_mae": np.abs(mean_q - t).mean(axis=(1, 2)),
                            "train_median_mse": np.square(med_q - t).mean(axis=(1, 2)),
                            "train_median_mae": np.abs(med_q - t).mean(axis=(1, 2)),
                        }
                        for name, value in metrics.items():
                            record[name] = episode_summary(value, ids[ix, 0])
                        if split == "train_loo" and seen is not None:
                            record["exposure_strata"] = exposure_strata(
                                ids[ix], seen[ix],
                                None if reference_seen is None else reference_seen[ix],
                                metrics, q, t,
                            )
                        record["per_dimension_mae"] = np.abs(q - t).mean(axis=(0, 1)).tolist()
                        records.append(record)
    return {
        "records": records,
        "statistical_unit": "episode",
        "training_replicates": 1,
        "fixed_step2500_cohorts_available": reference_seen is not None,
        "checkpoint_selection": False,
        "hidden_test_loaded": False,
        "m2_complete": False,
    }
