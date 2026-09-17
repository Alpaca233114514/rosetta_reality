"""Exercise saved Iris processors on temporal raw inputs without loading policy weights."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def audit():
    import torch
    import yaml
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors

    from rosetta_reality.data import resolve_prepared_cache
    from rosetta_reality.data.config import load_dataset_config
    from rosetta_reality.experiment import file_sha256
    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.vla.action_space import load_smolvla_action_space, load_smolvla_experiment
    from rosetta_reality.vla.processor import (
        ensure_smolvla_action_boundary,
        standard_aloha_action_to_model,
        standard_aloha_state_to_pi,
    )

    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Saved-chain audit requires the offline Docker runtime")
    plan = yaml.safe_load(
        (ROOT / "reports/training/iris-preparation-20260913/historical-main1280.yaml").read_text()
    )
    exp = load_smolvla_experiment(ROOT / plan["parent_experiment"]["config"], ROOT)
    cfg = load_dataset_config(ROOT / "configs/data/aloha_sim_insertion_m2.yaml")
    root, _ = resolve_prepared_cache(cfg, ROOT, validate_checksums=True)
    contract_path = ROOT / exp["action_contract"]["derived"]
    contract = load_action_contract(contract_path)
    space = load_smolvla_action_space(exp)
    norm_path = ROOT / plan["normalization"]["report"]
    if file_sha256(norm_path) != plan["normalization"]["report_sha256"]:
        raise ValueError("Normalization identity drift")
    stats = json.loads(norm_path.read_text())["effective_stats"]
    episodes = exp["dataset"]["train_episodes"][:2]
    ds = LeRobotDataset(
        cfg.repo_id,
        root=root,
        revision=cfg.revision,
        episodes=episodes,
        download_videos=False,
        return_uint8=True,
        video_backend="pyav",
        delta_timestamps={"action": [i / contract.frequency_hz for i in range(50)]},
    )
    starts = dict(
        zip(ds.meta.episodes["episode_index"], ds.meta.episodes["dataset_from_index"], strict=True)
    )
    records = {}
    for arm in ("control", "treatment"):
        export = ROOT / "runs/iris-recovered-20260913-002/verified/exports" / arm
        policy_cfg = PreTrainedConfig.from_pretrained(export)
        pre, post = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=export,
            pretrained_revision=None,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        before = [[type(s).__name__ for s in p.steps] for p in (pre, post)]
        ensure_smolvla_action_boundary(
            pre,
            post,
            contract,
            space,
            action_contract_sha256=file_sha256(contract_path),
            upstream_revision=exp["upstream"]["revision"],
        )
        if before != [[type(s).__name__ for s in p.steps] for p in (pre, post)]:
            raise ValueError("Saved processor was missing an action boundary")
        checked = []
        for ep in episodes:
            for frame in (0, 249, 499):
                raw = ds[ds.absolute_to_relative_idx[int(starts[ep]) + frame]]
                raw[cfg.cameras["top"]] = raw[cfg.cameras["top"]].float() / 255
                projected = (
                    raw["action"].maximum(contract.lower_bounds).minimum(contract.upper_bounds)
                )
                model_values = {
                    "action": standard_aloha_action_to_model(
                        projected, space.representation_adapter
                    ),
                    "observation.state": standard_aloha_state_to_pi(raw["observation.state"]),
                }
                processed = pre(copy.deepcopy(raw))
                errors = {}
                for key, value in model_values.items():
                    mean = torch.tensor(stats[key]["mean"], dtype=value.dtype)
                    std = torch.tensor(stats[key]["std"], dtype=value.dtype)
                    expected = (value - mean) / (std + 1e-8)
                    actual = processed[key].squeeze(0)
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                    errors[key] = float((actual - expected).abs().max())
                torch.testing.assert_close(
                    processed["observation.images.camera1"].squeeze(0),
                    raw[cfg.cameras["top"]],
                    atol=0,
                    rtol=0,
                )
                if not torch.equal(processed["action_is_pad"].reshape(-1), raw["action_is_pad"]):
                    raise ValueError("Saved processor changed action padding")
                decoded = post(processed["action"]).squeeze(0)
                torch.testing.assert_close(decoded, projected, atol=2e-6, rtol=1e-5)
                checked.append(
                    {
                        "episode": ep,
                        "frame": frame,
                        "normalization_max_error": errors,
                        "roundtrip_max_error": float((decoded - projected).abs().max()),
                    }
                )
        records[arm] = {
            "samples": checked,
            "saved_steps": before,
            "preprocessor_sha256": file_sha256(export / "policy_preprocessor.json"),
            "postprocessor_sha256": file_sha256(export / "policy_postprocessor.json"),
        }
    return {
        "status": "passed",
        "arms": records,
        "model_weights_loaded": False,
        "optimizer_steps": 0,
        "hidden_rows_materialized": 0,
        "normalization_report_sha256": file_sha256(norm_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "incomplete"}
    try:
        result = audit()
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        with (args.output / "result.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
