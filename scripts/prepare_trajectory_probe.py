"""Create a sealed, deliberately non-executable full-trajectory probe draft."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rosetta_reality.vla.training.features import FEATURE_FACTORIES  # noqa: E402
from rosetta_reality.vla.training.integrity import (  # noqa: E402
    CORE_IMPLEMENTATION,
    validate_local_implementation,
)
from rosetta_reality.vla.training.plan import validate_plan_structure  # noqa: E402
from rosetta_reality.vla.training.trajectory_probe import (  # noqa: E402
    NOISE_CONDITIONS,
    canonical_bytes,
    schedule,
)

SOURCE = (
    "runs/canonical-furnace-received-20260914-002/runs/canonical-fullframes-20260914-001/smoke.yaml"
)
SOURCE_SHA256 = "7cedb79742257df6188b298056a1e3f77261e296da32c55b6cb3257fea4871fd"
THRESHOLDS = "configs/vla/trajectory_probe_thresholds_20260922_001.json"
THRESHOLDS_SHA = "927acd9946e79bbe2ad0fa02e968aa89bbd9ae7c39c309325e01cf7fcf3e21f3"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Probe paths must stay in the repository")
    return path.relative_to(ROOT).as_posix()


def _save(path, value):
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value))


def implementation_members():
    owners = set(CORE_IMPLEMENTATION) | {
        "scripts/prepare_trajectory_probe.py",
    }
    owners.update(
        p.relative_to(ROOT).as_posix() for p in (ROOT / "src/rosetta_reality/vla").rglob("*.py")
    )
    return sorted(owners)


def _draft(source, *, name, episodes, batch_size, steps, scheduler_steps, samples, schedule_path):
    plan = copy.deepcopy(source)
    plan.update(
        status="draft",
        plan_id=f"trajectory-probe-{name}-draft-003",
        run_name=f"trajectory-probe-{name}-draft-003",
        hypothesis="Full-trajectory fit probe; no causal or Gate claim",
        training_authorized=False,
        formal_training_authorized=False,
        rollout_authorized=False,
        model_execution_authorized=False,
        execution=None,
        closed_loop_claim=False,
        formal_training_claim=False,
    )
    training = plan["training"]
    reference_steps = scheduler_steps if name == "smoke" else steps
    training.update(
        episodes=list(episodes),
        batch_size=batch_size,
        steps=reference_steps,
        save_freq=(reference_steps // 4),
        log_freq=1,
        checkpoint_steps=[reference_steps * i // 4 for i in range(1, 5)],
        hidden_test_loaded=False,
    )
    training["scheduler"].update(num_warmup_steps=16, num_decay_steps=scheduler_steps)
    # The historical optimizer_smoke overlay cannot express selected frames.
    # Keep its count informational; only the sealed schedule defines identities.
    plan["optimizer_smoke"].update(
        run_name=plan["run_name"],
        episodes=list(episodes),
        batch_size=batch_size,
        steps=steps,
        save_freq=(steps // 4 if name != "smoke" else steps),
        log_freq=1,
        save_checkpoint=True,
    )
    plan["preflight"].update(run_name=plan["run_name"] + "-preflight", episodes=[2], batch_size=1)
    features = plan["features"]
    declaration = next(f for f in features if f["name"] == "explicit_sample_schedule")
    declaration.update(path=_relative(schedule_path), sha256=_sha(schedule_path))
    plan["probe"] = {
        "status": "draft",
        "execution": None,
        "historical_source": {"path": SOURCE, "sha256": SOURCE_SHA256},
        "schedule": {"path": _relative(schedule_path), "sha256": _sha(schedule_path)},
        "sample_exposures": len(samples),
        "unique_episode_frames": len({tuple(item) for item in samples}),
        "rounds": 4 if name != "smoke" else None,
        "smoke_is_main_furnace": False if name == "smoke" else None,
        "observer_entrypoint": "rosetta_reality.vla.training.observed_launch.run_observed_launch",
        "launcher_rejection": "v2 requires status preregistered; this plan is draft",
        "normalization_scope": "canonical_train40_statistics_inherited_without_recalibration",
        "only_execution_phase": "optimizer_smoke",
        "active_steps": steps,
        "active_checkpoint_steps": [steps] if name == "smoke" else training["checkpoint_steps"],
    }
    plan["stop_conditions"] = [
        f"Stop at exactly {steps} fresh optimizer updates / {len(samples)} sample exposures",
        "Stop on source, schedule, runtime contract, finite or resource mismatch",
        "Preserve all complete checkpoints and failures; no automatic retry or Gate claim",
    ]
    plan["implementation_files"] = {
        member: _sha(ROOT / member) for member in implementation_members()
    }
    plan["probe"]["historical_implementation_files"] = source["implementation_files"]
    if name == "smoke":
        plan["probe"]["scheduler_scope"] = "A scheduler retained; two-step smoke budget only"
        plan["probe"]["training_section_scope"] = "reference_only_not_500_update_authorization"
        plan["probe"]["frame_offsets"] = [0, 499]
    return plan


def prepare(plan_path: Path, output: Path):
    if _relative(plan_path) != SOURCE or _sha(plan_path) != SOURCE_SHA256:
        raise ValueError("Historical canonical smoke plan path or SHA-256 differs")
    source = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if _sha(ROOT / THRESHOLDS) != THRESHOLDS_SHA:
        raise ValueError("Train-only event threshold identity differs")
    thresholds = json.loads((ROOT / THRESHOLDS).read_text())
    if source["plan_id"] != "canonical-fullframes-20260914-001":
        raise ValueError("Historical canonical plan identity differs")
    if not set((2, 49, 4, 23)) <= set(source["training"]["episodes"]):
        raise ValueError("Probe episodes are not in historical training split")
    if source["training"]["policy"]["empty_cameras"] != 2:
        raise ValueError("Canonical policy contract changed")
    output = output.resolve()
    _relative(output)
    if output == plan_path.parent or output.is_relative_to(plan_path.parent):
        raise ValueError("Output must not modify the historical evidence tree")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    definitions = {
        "smoke": ((2,), 1, 2, 500, [[2, 0], [2, 499]]),
        "a": ((2,), 4, 500, 500, schedule((2,))),
        "b": ((2, 49, 4, 23), 4, 2000, 2000, schedule((2, 49, 4, 23))),
    }
    files = {}
    for name, (episodes, batch, steps, decay, samples) in definitions.items():
        schedule_path = output / f"{name}-schedule.json"
        _save(
            schedule_path,
            {
                "status": "draft",
                "execution": None,
                "training_authorized": False,
                "rollout_authorized": False,
                "sample_identities": samples,
                "seed": 20260809,
                "seed_per_round": [20260809 + i for i in range(4)] if name != "smoke" else None,
            },
        )
        draft = _draft(
            source,
            name=name,
            episodes=episodes,
            batch_size=batch,
            steps=steps,
            scheduler_steps=decay,
            samples=samples,
            schedule_path=schedule_path,
        )
        draft_path = output / f"{name}-draft.yaml"
        # Structural rehearsal is read-only and does not authorize a launch.
        candidate = copy.deepcopy(draft)
        candidate["status"] = "preregistered"
        validate_plan_structure(candidate, known_features=FEATURE_FACTORIES)
        validate_local_implementation(candidate, ROOT)
        with draft_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(draft, stream, sort_keys=False)
        files[draft_path.name] = _sha(draft_path)
        files[schedule_path.name] = _sha(schedule_path)
    protocol = {
        "status": "draft",
        "execution": None,
        "training_authorized": False,
        "rollout_authorized": False,
        "hidden_test_loaded": False,
        "scoring": "saved predictions only; valid action mask excludes tail padding",
        "noise_conditions": list(NOISE_CONDITIONS),
        "metrics": [
            "first_joint_mae",
            "first_gripper_mae",
            "full_joint_mae",
            "full_gripper_mae",
            "gripper_event_bias",
            "gripper_hold_bias",
        ],
        "candidate_selection": "fixed final step per arm; no offline-only Gate inference",
        "independent_reload": "required before any real Gate, presently unmeasured",
        "gate3_gate4": "not authorized or measured",
    }
    _save(output / "thresholds.json", thresholds)
    files["thresholds.json"] = _sha(output / "thresholds.json")
    protocol["thresholds"] = {
        "source_path": _relative(output / "thresholds.json"),
        "source_sha256": files["thresholds.json"],
        "left": thresholds["left"],
        "right": thresholds["right"],
    }
    _save(output / "evaluation-protocol.json", protocol)
    files["evaluation-protocol.json"] = _sha(output / "evaluation-protocol.json")
    manifest = {
        "status": "draft",
        "execution": None,
        "training_authorized": False,
        "rollout_authorized": False,
        "source_plan": {"path": SOURCE, "sha256": SOURCE_SHA256},
        "files": files,
        "model_loaded": False,
        "data_loaded": False,
        "optimizer_steps": 0,
        "structural_and_source_preflight": "passed_for_all_three_drafts",
    }
    _save(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.plan, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
