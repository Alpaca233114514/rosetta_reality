"""Counterexamples for complete-grid, source-bound validation-only selection."""

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

from rosetta_reality.experiment import file_sha256
from rosetta_reality.vla.checkpoint_selection import METRICS, select_reports

IDENTITY = {"formal_plan_sha256": "a" * 64, "experiment_id": "unit-experiment"}
VALIDATION = {"episodes": [3, 1], "frame_offsets": [0, 4]}
FIXED = {"noise": "zeros", "flow_time": 0.5}


def records():
    result = []
    for kind, step in [("base", 0), ("checkpoint", 8), ("checkpoint", 4)]:
        model_sha = f"{step + 1:064x}"
        result.append(
            {
                "report_sha256": f"{step + 100:064x}",
                "source": {"kind": kind, "step": step, "model_sha256": model_sha},
                "report": {
                    "status": "complete",
                    "stage": "smolvla_fixed_validation",
                    **IDENTITY,
                    "hidden_test_loaded": False,
                    "gradients_enabled": False,
                    "optimizer_created": False,
                    "network_disabled": True,
                    "validation_episodes": [3, 1],
                    "materialized_episodes": [1, 3],
                    "frame_offsets": [0, 4],
                    "sample_count": 4,
                    "fixed_input": FIXED,
                    "model_source": {
                        "kind": kind,
                        "step": step,
                        "model_safetensors_sha256": model_sha,
                    },
                    "metrics": {key: 0.2 for key in METRICS},
                },
            }
        )
    return result


def select(rows):
    return select_reports(
        rows, identity=IDENTITY, validation=VALIDATION, checkpoint_steps=[4, 8], fixed_input=FIXED
    )


def test_ties_select_earlier_checkpoint_regardless_of_report_order():
    rows = records()
    assert select(rows)["selected_checkpoint_step"] == 4
    assert select(list(reversed(rows)))["selected_checkpoint_step"] == 4


def test_zero_base_has_no_fraction_and_worse_candidate_is_honestly_labeled():
    rows = records()
    rows[0]["report"]["metrics"]["first_action_mae"] = 0
    decision = select(rows)
    assert decision["improvement_over_base_fraction"] is None
    assert decision["improves_over_base"] is False
    assert decision["base_role"] == "reference_only"
    assert decision["m2_complete"] is False


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True, "0.1"])
def test_all_metrics_must_be_finite_real_numbers(metric, bad):
    rows = records()
    rows[1]["report"]["metrics"][metric] = bad
    with pytest.raises(ValueError, match="metric"):
        select(rows)


@pytest.mark.parametrize(
    "key,value",
    [
        ("formal_plan_sha256", "b" * 64),
        ("hidden_test_loaded", True),
        ("sample_count", 3),
        ("frame_offsets", [0]),
        ("materialized_episodes", [1, 3, 5]),
        ("fixed_input", {"noise": "gaussian", "flow_time": 0.5}),
    ],
)
def test_identity_and_cohort_drift_are_rejected(key, value):
    rows = records()
    rows[1]["report"][key] = value
    with pytest.raises(ValueError):
        select(rows)


def test_grid_source_and_duplicate_reports_are_rejected():
    with pytest.raises(ValueError, match="complete"):
        select(records()[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        select(records() + [records()[1]])
    rows = records()
    rows[1]["report"]["model_source"]["step"] = 4
    with pytest.raises(ValueError, match="source"):
        select(rows)


@pytest.mark.parametrize("family", ["zen", "vcdropout", "vfunfreeze"])
def test_frozen_selector_tie_counterexample(tmp_path, monkeypatch, family):
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module(f"scripts.select_smolvla_{family}_checkpoint")
    protocol = module.protocol
    if family == "zen":
        plan_id, spec = next(iter(protocol.ZEN_SPECS.items()))
        prefix, run = spec["validation_prefix"], spec["run_name"]
    else:
        tag = "VCD" if family == "vcdropout" else "VFU"
        plan_id = getattr(protocol, f"{tag}_PLAN_ID")
        prefix, run = (
            getattr(protocol, f"{tag}_{key}") for key in ("VALIDATION_PREFIX", "RUN_NAME")
        )
    plan_path = tmp_path / "synthetic-plan.json"
    plan_path.write_text("{}")
    # Only isolate input acquisition; execute the untouched historical ranking.
    monkeypatch.setattr(protocol, "resolve_plan", lambda _: ({}, plan_id))
    monkeypatch.setattr(module, "workspace_code_identity", lambda _: {})
    monkeypatch.setattr(
        sys, "argv", ["selector", "--plan", str(plan_path), "--run-root", str(tmp_path)]
    )
    directory = tmp_path / protocol.EXPERIMENT_ID / "validation"
    directory.mkdir(parents=True)
    for step in [0, *protocol.CHECKPOINT_STEPS]:
        label = "base" if step == 0 else f"step-{step:06d}"
        report = copy.deepcopy(records()[0]["report"])
        report["formal_plan_sha256"] = file_sha256(plan_path)
        report["model_source"] = {"kind": "base" if not step else "checkpoint", "step": step}
        (directory / f"{prefix}-{label}.json").write_text(json.dumps(report))
    module.main()
    decision = json.loads(
        (tmp_path / protocol.EXPERIMENT_ID / "selection" / f"{run}-selection.json").read_text()
    )
    assert decision["tie_break"] == "earlier_checkpoint"
    # Preserved negative control: historical code still chooses the wrong end.
    assert decision["selected_checkpoint_step"] == max(protocol.CHECKPOINT_STEPS)
    assert select(records())["selected_checkpoint_step"] == 4


def test_new_cli_checks_real_source_and_report_seals_before_create_only_output(
    tmp_path, monkeypatch
):
    from test_smolvla_training_plan_schema import _base_plan

    from rosetta_reality.vla.action_space import load_smolvla_experiment
    from rosetta_reality.vla.training.integrity import CORE_IMPLEMENTATION
    from scripts import select_smolvla_checkpoint_v2 as cli

    repository = Path(__file__).resolve().parents[1]
    parent_name = "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
    exp = load_smolvla_experiment(repository / parent_name, repository)
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(exp))
    contract = tmp_path / exp["action_contract"]["derived"]
    contract.parent.mkdir(parents=True)
    contract.write_bytes((repository / exp["action_contract"]["derived"]).read_bytes())
    plan = _base_plan()
    plan["parent_experiment"] = {
        "config": "parent.json",
        "sha256": file_sha256(parent),
        "experiment_id": exp["experiment_id"],
    }
    plan["training"].update(
        episodes=exp["dataset"]["train_episodes"],
        steps=8,
        save_freq=4,
        log_freq=1,
        checkpoint_steps=[4, 8],
    )
    plan["training"]["scheduler"].update(num_warmup_steps=2, num_decay_steps=8)
    plan["selection_contract"] = {
        "primary_metric": "first_action_mae",
        "tie_break": "earlier_checkpoint",
        "base_role": "reference_only",
        "fixed_input": FIXED,
    }
    plan["implementation_files"] = {}
    for name in CORE_IMPLEMENTATION | {
        "scripts/select_smolvla_checkpoint_v2.py",
        "src/rosetta_reality/vla/checkpoint_selection.py",
    }:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repository / name).read_bytes())
        plan["implementation_files"][name] = file_sha256(target)
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan))
    rows, entries = records(), []
    for index, row in enumerate(rows):
        report = row["report"]
        report.update(
            formal_plan_sha256=file_sha256(plan_path),
            experiment_id=exp["experiment_id"],
            experiment_config_sha256=file_sha256(parent),
            action_contract_sha256=file_sha256(contract),
            normalization_report_sha256=plan["normalization"]["report_sha256"],
            dataset_view_manifest_sha256=plan["normalization"]["dataset_view_manifest_sha256"],
            model_revision=exp["model"]["revision"],
            dataset_revision=exp["dataset"]["revision"],
            validation_episodes=plan["validation"]["episodes"],
            materialized_episodes=sorted(plan["validation"]["episodes"]),
            frame_offsets=plan["validation"]["frame_offsets"],
        )
        path = tmp_path / f"report-{index}.json"
        path.write_text(json.dumps(report))
        entries.append({"path": path.name, "sha256": file_sha256(path), "source": row["source"]})
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"plan_sha256": file_sha256(plan_path), "reports": entries}))
    output = tmp_path / "selection.json"
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setitem(cli._resolve_plan.__globals__, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selector",
            "--plan",
            str(plan_path),
            "--inputs",
            str(inputs),
            "--inputs-sha256",
            file_sha256(inputs),
            "--output",
            str(output),
        ],
    )
    assert cli.main() == 0
    assert json.loads(output.read_text())["selected_checkpoint_step"] == 4
    with pytest.raises(FileExistsError):
        cli.main()
    sys.argv[-1] = str(tmp_path / "selection-drift.json")
    (tmp_path / "report-1.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        cli.main()
    assert not (tmp_path / "selection-drift.json").exists()
