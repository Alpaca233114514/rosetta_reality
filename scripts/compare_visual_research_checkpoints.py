"""Independent CPU replay of checkpoint pairs with fixed input exposure cohorts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bundle(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = (root / name).resolve()
        assert path.is_relative_to(root.resolve()) and path.stat().st_size == entry["bytes"]
        assert digest(path) == entry["sha256"]
    with np.load(root / "arrays.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return arrays, json.loads((root / "result.json").read_text())


def summarize(values, episodes):
    if len(values) == 0:
        return None
    grouped = {str(int(e)): float(np.mean(values[episodes == e])) for e in sorted(set(episodes))}
    a = np.asarray(list(grouped.values()))
    leave = [np.delete(a, j).mean() for j in range(len(a))] if len(a) > 1 else []
    return dict(
        mean=float(a.mean()),
        per_episode=grouped,
        episode_count=len(a),
        leave_one_episode_out_range=[float(min(leave)), float(max(leave))] if leave else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early", type=Path, required=True)
    parser.add_argument("--late", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    early, em = load_bundle(args.early / "collect")
    late, lm = load_bundle(args.late / "collect")
    assert (em["checkpoint_step"], lm["checkpoint_step"]) == (2500, 5000)
    assert (
        em["checkpoint_sha256"]
        == "a3494600183d0a24c71ab7db0976afbc57567ac714ac678fd572c607e06a2f0c"
    )
    assert (
        lm["checkpoint_sha256"]
        == "d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef"
    )
    common = [
        "identities",
        "targets",
        "raw_targets",
        "normalized_targets",
        "valid_mask",
        "noise",
        "input_state",
        "input_language_tokens",
        "input_language_attention_mask",
        "input_seen_by_step2500",
        "cyclic_donors",
        "frame0_nonvisual_equal",
    ]
    for key in common:
        np.testing.assert_array_equal(early[key], late[key])
    assert em["inputs"] == lm["inputs"] and em["groups"] == lm["groups"]
    for meta in (em, lm):
        assert meta["parameters_unchanged"] and meta["optimizer_steps"] == 0
        assert meta["forwards"] == 1268 and len(meta["controls"]) == 8
        assert all(c["exact"] for c in meta["controls"])
    np.testing.assert_array_equal(early["input_seen_at_checkpoint"], late["input_seen_by_step2500"])
    assert early["input_seen_at_checkpoint"].sum() == 83
    assert late["input_seen_at_checkpoint"].sum() == 160
    ids, train, reference = (
        late["identities"],
        late["input_seen_at_checkpoint"],
        late["input_seen_by_step2500"],
    )
    assert late["valid_mask"].all() and late["frame0_nonvisual_equal"].item()
    y = late["targets"].astype(float)
    endpoints = [early, late]
    prior = [
        json.loads((r / "analyze/result.json").read_text())["records"]
        for r in (args.early, args.late)
    ]
    lookup = [
        {(r["noise_index"], r["frame"], r["split"], r["window"], r["group"]): r for r in rr}
        for rr in prior
    ]
    output, checks = [], 0
    for n in range(4):
        for frame in (0, 125, 250, 375):
            for split, split_mask in (("train_loo", train), ("development", ~train)):
                ix = np.flatnonzero((ids[:, 1] == frame) & split_mask)
                cohorts = {"all": np.ones(len(ix), dtype=bool)}
                if split == "train_loo":
                    cohorts.update(
                        seen_by_step2500=reference[ix], unseen_by_step2500=~reference[ix]
                    )
                for window, width in (("first", 1), ("full_chunk", 50)):
                    for group, dims in em["groups"].items():
                        both = []
                        target = y[ix, :width][:, :, dims]
                        for endpoint, arrays in enumerate(endpoints):
                            p = arrays["correct"][n, ix, :width][:, :, dims].astype(float)
                            mse = ((p - target) ** 2).mean(axis=(1, 2))
                            mae = abs(p - target).mean(axis=(1, 2))
                            wrong_mse, wrong_mae, mean_mse, mean_mae, med_mse, med_mae = (
                                [] for _ in range(6)
                            )
                            for j, row in enumerate(ix):
                                wrong = (
                                    np.delete(p, j, axis=0)
                                    if frame == 0
                                    else arrays["wrong"][n, row, :width][:, dims].astype(float)[
                                        None
                                    ]
                                )
                                wrong_mse.append(float(np.mean((wrong - target[j]) ** 2)))
                                wrong_mae.append(float(np.mean(abs(wrong - target[j]))))
                                pool = np.flatnonzero(
                                    train & (ids[:, 1] == frame) & (ids[:, 0] != ids[row, 0])
                                )
                                donors = y[pool, :width][:, :, dims]
                                for estimate, dst_mse, dst_mae in (
                                    (donors.mean(axis=0), mean_mse, mean_mae),
                                    (np.median(donors, axis=0), med_mse, med_mae),
                                ):
                                    delta = estimate - target[j]
                                    dst_mse.append(float(np.mean(delta**2)))
                                    dst_mae.append(float(np.mean(abs(delta))))
                            metrics = dict(
                                correct_mse=mse,
                                correct_mae=mae,
                                wrong_mse=np.array(wrong_mse),
                                wrong_mae=np.array(wrong_mae),
                                image_gain_mse=np.array(wrong_mse) - mse,
                                train_mean_mse=np.array(mean_mse),
                                train_mean_mae=np.array(mean_mae),
                                train_median_mse=np.array(med_mse),
                                train_median_mae=np.array(med_mae),
                            )
                            old = lookup[endpoint][(n, frame, split, window, group)]
                            for key, values in metrics.items():
                                expected = [
                                    old[key]["per_episode"][str(int(ids[r, 0]))] for r in ix
                                ]
                                np.testing.assert_allclose(values, expected, atol=1e-12, rtol=1e-10)
                                checks += len(ix)
                            both.append((metrics, p))
                        for cohort, mask in cohorts.items():
                            ep = ids[ix[mask], 0]
                            result = dict(
                                noise_index=n,
                                frame=frame,
                                split=split,
                                window=window,
                                group=group,
                                cohort=cohort,
                                identities=ids[ix[mask]].tolist(),
                                input_count=int(mask.sum()),
                                early_seen=int(early["input_seen_at_checkpoint"][ix[mask]].sum()),
                                late_seen=int(late["input_seen_at_checkpoint"][ix[mask]].sum()),
                            )
                            result["metrics"] = {
                                key: dict(
                                    early=summarize(both[0][0][key][mask], ep),
                                    late=summarize(both[1][0][key][mask], ep),
                                    late_minus_early=summarize(
                                        both[1][0][key][mask] - both[0][0][key][mask], ep
                                    ),
                                )
                                for key in both[0][0]
                            }
                            for label, (metrics, p) in zip(("early", "late"), both):
                                q, t = p[mask], target[mask]
                                if not len(q):
                                    result[label + "_decomposition"] = None
                                    continue
                                pc, yc = q - q.mean(axis=0), t - t.mean(axis=0)
                                parts = dict(
                                    prediction_variance=float(np.mean(pc**2)),
                                    target_variance=float(np.mean(yc**2)),
                                    covariance=float(np.mean(pc * yc)),
                                    bias_squared=float(
                                        np.mean((q.mean(axis=0) - t.mean(axis=0)) ** 2)
                                    ),
                                    mse=float(np.mean((q - t) ** 2)),
                                )
                                np.testing.assert_allclose(
                                    parts["mse"],
                                    parts["prediction_variance"]
                                    + parts["target_variance"]
                                    - 2 * parts["covariance"]
                                    + parts["bias_squared"],
                                    atol=1e-12,
                                )
                                result[label + "_decomposition"] = parts
                            output.append(result)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "comparisons.jsonl").open("x") as f:
        for row in output:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    np.savez_compressed(
        args.output / "arrays.npz",
        identities=ids,
        targets=y,
        noise=late["noise"],
        early_correct=early["correct"],
        late_correct=late["correct"],
        early_wrong=early["wrong"],
        late_wrong=late["wrong"],
        early_seen=early["input_seen_at_checkpoint"],
        late_seen=train,
        reference_seen=reference,
        valid_mask=late["valid_mask"],
    )
    result = dict(
        status="passed",
        common_arrays_exact=common,
        image_identity_exact=True,
        independent_metric_checks=checks,
        records=len(output),
        model_calls=0,
        optimizer_updates=0,
        source_manifests={
            "early": digest(args.early / "collect/manifest.json"),
            "late": digest(args.late / "collect/manifest.json"),
        },
        interpretation=(
            "Within-one-training-run development diagnostics; "
            "no checkpoint selection or causal scene-count claim."
        ),
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    files = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in args.output.iterdir()}
    (args.output / "manifest.json").write_text(json.dumps(dict(files=files), indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
