"""Compare full train-only transformed statistics and saved Iris processor state locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main():
    import numpy as np
    import pyarrow.dataset as arrow
    import torch
    import yaml
    from safetensors.numpy import load_file

    from rosetta_reality.data import resolve_prepared_cache
    from rosetta_reality.data.config import load_dataset_config
    from rosetta_reality.experiment import file_sha256
    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.vla.action_space import load_smolvla_action_space, load_smolvla_experiment
    from rosetta_reality.vla.processor import (
        standard_aloha_action_to_model,
        standard_aloha_state_to_pi,
    )
    from rosetta_reality.vla.vision_diagnostics import validate_splits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "incomplete",
        "model_loaded": False,
        "hidden_rows_materialized": 0,
        "optimizer_steps": 0,
        "normalization": {},
        "saved_processors": {},
    }
    try:
        cfg = load_dataset_config(ROOT / "configs/data/aloha_sim_insertion_m2.yaml")
        plan = yaml.safe_load(
            (
                ROOT / "reports/training/iris-preparation-20260913/historical-main1280.yaml"
            ).read_text()
        )
        exp = load_smolvla_experiment(ROOT / plan["parent_experiment"]["config"], ROOT)
        d = exp["dataset"]
        validate_splits(d["train_episodes"], d["validation_episodes"], d["test_episodes"])
        root, _ = resolve_prepared_cache(cfg, ROOT, validate_checksums=True)
        norm_path = ROOT / plan["normalization"]["report"]
        if file_sha256(norm_path) != plan["normalization"]["report_sha256"]:
            raise ValueError("Historical normalization identity drift")
        norm = json.loads(norm_path.read_text())
        if norm["train_episodes"] != d["train_episodes"] or norm["hidden_test_loaded"] is not False:
            raise ValueError("Normalization split differs")
        contract = load_action_contract(ROOT / exp["action_contract"]["derived"])
        space = load_smolvla_action_space(exp)
        scanner = arrow.dataset(root / "data", format="parquet").scanner(
            columns=["episode_index", "observation.state", "action"],
            filter=arrow.field("episode_index").isin(d["train_episodes"]),
            batch_size=256,
        )
        totals = {
            key: {"n": 0, "sum": np.zeros(14), "square": np.zeros(14)}
            for key in ("observation.state", "action")
        }
        for batch in scanner.to_batches():
            rows = batch.to_pylist()
            if not rows:
                continue
            if any(row["episode_index"] not in d["train_episodes"] for row in rows):
                raise ValueError("Scanner escaped the training split")
            states = torch.tensor([row["observation.state"] for row in rows], dtype=torch.float64)
            actions = torch.tensor([row["action"] for row in rows], dtype=torch.float64)
            actions = actions.maximum(contract.lower_bounds).minimum(contract.upper_bounds)
            values = {
                "observation.state": standard_aloha_state_to_pi(states).numpy(),
                "action": standard_aloha_action_to_model(
                    actions, space.representation_adapter
                ).numpy(),
            }
            for key, array in values.items():
                totals[key]["n"] += len(array)
                totals[key]["sum"] += array.sum(axis=0)
                totals[key]["square"] += (array * array).sum(axis=0)
        for key, total in totals.items():
            mean = total["sum"] / total["n"]
            std = np.sqrt(np.maximum(total["square"] / total["n"] - mean * mean, 0))
            expected = norm["effective_stats"][key]
            errors = {
                "mean": float(np.max(np.abs(mean - expected["mean"]))),
                "std": float(np.max(np.abs(std - expected["std"]))),
            }
            if total["n"] != norm["train_rows"] or max(errors.values()) > 1e-7:
                raise ValueError("Train-only normalization recomputation differs")
            result["normalization"][key] = {"rows": total["n"], "max_absolute_error": errors}
        # Compare serialized normalizer arrays to the exact historical train-only
        # report; this does not instantiate a policy or deserialize its weights.
        for arm in ("control", "treatment"):
            export = ROOT / "runs/iris-recovered-20260913-002/verified/exports" / arm
            records = {}
            for filename in (
                "policy_preprocessor_step_7_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            ):
                arrays = load_file(export / filename)
                checked = []
                for name, array in arrays.items():
                    for feature in ("observation.state", "action"):
                        for stat in ("mean", "std"):
                            if name == f"{feature}.{stat}":
                                expected = np.asarray(
                                    norm["effective_stats"][feature][stat], dtype=array.dtype
                                )
                                if not np.array_equal(array, expected):
                                    raise ValueError(
                                        f"Saved processor normalization differs: {name}"
                                    )
                                checked.append(name)
                if not {"action.mean", "action.std"} <= set(checked):
                    raise ValueError(
                        f"Normalizer tensor naming or action statistics differ: {sorted(arrays)}"
                    )
                records[filename] = {"sha256": file_sha256(export / filename), "checked": checked}
            result["saved_processors"][arm] = records
        result.update(status="passed", normalization_report_sha256=file_sha256(norm_path))
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        with (args.output / "result.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
