"""Create Hestia's fixed 1280-update plan and an intentionally non-launchable contract."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
CONTROL = "configs/vla/m2-smolvla450m-visual-native-b4-pilot-003.yaml"


def build_plan(control):
    from rosetta_reality.vla import visual_fit as fit
    from rosetta_reality.vla import visual_fit_contract as checks

    plan = copy.deepcopy(control)
    name = fit.RUN_NAMES["C"]
    plan.update(
        status="preregistered",
        plan_id=name,
        run_name=name,
        hypothesis="Increase native fit budget and matched cosine decay from 256 to 1280 updates.",
        formal_training_claim=False,
        formal_training_authorized=False,
    )
    for section in ("training", "optimizer_smoke"):
        plan[section].update(
            episodes=fit.TRAIN40[:],
            batch_size=4,
            steps=1280,
            save_freq=320,
            log_freq=16,
            num_workers=0,
            persistent_workers=False,
        )
    plan["optimizer_smoke"]["run_name"] = name
    plan["training"]["checkpoint_steps"] = [320, 640, 960, 1280]
    plan["training"]["scheduler"] = checks.expected_scheduler("C")
    plan["preflight"]["run_name"] = name + "-forward"
    plan["features"][2]["sample_identities"] = [{"episode": ep, "frame": 0} for ep in fit.TRAIN40]
    plan["stop_conditions"] = [
        "identity, sample or native optimizer/scheduler mismatch",
        "nonfinite loss/gradient/action or illegal unclipped standard action",
        "allocated above 8 GiB, reserved above 10 GiB or host RSS above 10 GiB",
        "insufficient disk budget; preserve every historical checkpoint",
        "shared 1800-second deadline or failed prerequisite/reload",
        "failed training fit or development comparison; no automatic retry or next furnace",
    ]
    checks.validate_plan(plan, "C")
    return plan


def prepare(output):
    from run_smolvla_v2 import _resolve_plan

    from rosetta_reality.vla import visual_fit as fit
    from rosetta_reality.vla import visual_fit_contract as checks

    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Prepared artifacts must stay inside this workspace")
    control = yaml.safe_load((ROOT / CONTROL).read_text())
    for name, sha in control["implementation_files"].items():
        if fit.file_hash(ROOT / name) != sha:
            raise ValueError("Audited native training source changed: " + name)
    plan = build_plan(control)
    output.mkdir(parents=True, exist_ok=False)

    def reference(path):
        return {"path": path.relative_to(ROOT).as_posix(), "sha256": fit.file_hash(path)}

    def save(path, value):
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)

    path = output / "main1280.yaml"
    with path.open("x") as stream:
        yaml.safe_dump(plan, stream, sort_keys=False)
    loaded, _, experiment = _resolve_plan(path)
    if loaded != plan:
        raise ValueError("Native launcher plan changed after YAML round trip")
    checks.validate_experiment(experiment)
    history = json.loads((ROOT / checks.HISTORY).read_text())
    review = {
        "schema_version": 1,
        "protocol": fit.PROTOCOL,
        "run_name": fit.RUN_NAMES["C"],
        "status": "draft_pending_live_checks",
        "executable": False,
        "data": {
            "train40": fit.TRAIN40,
            "development_validation5": fit.DEV5,
            "sealed_hidden5": fit.HIDDEN5,
        },
        "intervention": "updates_and_cosine_decay_256_to_1280",
        "pure_repetition_causal_claim": False,
        "noise_conditions": [None, 20260905, 20260906, 20260907],
        "candidate_selection": "fixed_final_step_1280",
        "main_phase": "smoke",
        "minimum_joint_gripper_aggregate_common_conditions": 3,
        "correct_below_mean_mismatch_and_mean_floor": True,
        "standard_chunk_and_first_action_nonregression": True,
        "independent_reload_all_seven_arrays_exact": True,
        "development_is_independent_test": False,
        "hidden_test_loaded": False,
        "base_optimizer_resumed": False,
        "control_retrained": False,
        "m2_complete": False,
    }
    review_path = output / "review.json"
    save(review_path, review)
    sources = dict(control["implementation_files"])
    for name in (
        "scripts/prepare_visual_fit.py",
        "scripts/run_hestia_fit.py",
        "scripts/evaluate_visual_fit.py",
        "scripts/evaluate_visual_native_small.py",
        "scripts/inspect_hestia_schedule.py",
        "scripts/seal_visual_fit_candidate.py",
        "scripts/evaluate_visual_fit_evidence.py",
        "src/rosetta_reality/vla/visual_fit.py",
        "src/rosetta_reality/vla/visual_fit_contract.py",
        "src/rosetta_reality/vla/visual_fit_job.py",
        "src/rosetta_reality/vla/visual_coverage.py",
        "src/rosetta_reality/vla/vision_diagnostics.py",
        "src/rosetta_reality/vla/training/observation.py",
        "src/rosetta_reality/vla/training/observed_launch.py",
        checks.HISTORY,
    ):
        sources[name] = fit.file_hash(ROOT / name)
    cp = history["checkpoint_identities"]["B"]["checkpoint_relative_to_root"]
    candidate_cp = cp.replace(fit.RUN_NAMES["B"], fit.RUN_NAMES["C"]).replace("000256", "001280")
    template = {
        "schema_version": 1,
        "protocol": fit.PROTOCOL,
        "status": "draft",
        "model_execution_authorized": False,
        "started_unix": None,
        "deadline_unix": None,
        "watchdog": None,
        "review_plan": reference(review_path),
        "implementation_files": sources,
        "upstream_files": checks.UPSTREAM,
        "resource_limits": {
            "cuda_allocated_bytes": 8 * 1024**3,
            "cuda_reserved_bytes": 10 * 1024**3,
            "host_rss_bytes": 10 * 1024**3,
        },
        "prerequisite_evidence": {
            name: None
            for name in (
                "environment",
                "sample_contract",
                "resource_preflight",
                "two_step_smoke_reload",
                "B_training_integrity",
                "C_training_integrity",
            )
        },
        "arms": {
            "B": history["checkpoint_identities"]["B"],
            "C": {
                "run_name": fit.RUN_NAMES["C"],
                "plan": reference(path),
                "checkpoint_relative_to_root": candidate_cp,
                "checkpoint_files": None,
            },
        },
        "m2_complete": False,
    }
    save(output / "execution-contract.template.json", template)
    return {
        "status": "draft_prepared",
        "plan": reference(path),
        "model_loaded": False,
        "data_loaded": False,
        "optimizer_steps": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output)))
