"""CPU-only acceptance of Hestia collection contracts; never launches training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
FILES = [
    "scripts/check_hestia_collector_cpu.py",
    "scripts/run_hestia_fit.py",
    "scripts/evaluate_visual_fit.py",
    "scripts/inspect_hestia_schedule.py",
    "scripts/prepare_visual_fit.py",
    "scripts/seal_visual_fit_candidate.py",
    "src/rosetta_reality/vla/visual_fit.py",
    "src/rosetta_reality/vla/visual_fit_contract.py",
    "src/rosetta_reality/vla/visual_fit_job.py",
    "tests/test_smolvla_visual_fit.py",
    "tests/test_smolvla_visual_fit_contract.py",
    "tests/test_smolvla_visual_fit_job.py",
]
TESTS = [
    "visual_fit_contract",
    "visual_fit_job",
    "visual_fit",
    "observed_launch",
    "training_observation",
    "visual_coverage",
    "fixed_visual_samples",
    "tracking_composition",
    "v2_error_boundaries",
    "training_plan_schema",
    "training_launch",
]


def verify(output: Path):
    from run_smolvla_v2 import _resolve_plan

    from rosetta_reality.vla import visual_fit as fit
    from rosetta_reality.vla import visual_fit_contract as checks

    if platform.system() != "Linux" or not os.environ.get("ROSETTA_AUTODL_RUNTIME_PROFILE"):
        raise ValueError("This acceptance requires the registered AutoDL Linux runner")
    if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")):
        raise ValueError("Offline execution is required")
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)

    def run(name, arguments):
        with (output / (name + ".log")).open("x") as stream:
            subprocess.run(
                [sys.executable, *arguments],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=True,
            )

    def source_identity():
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        return {name: fit.file_hash(ROOT / name) for name in names if name}

    source = source_identity()
    save("source.json", source)
    run("ruff", ["-m", "ruff", "check", *FILES])
    installed = Path(next(iter(find_spec("lerobot").submodule_search_locations)))
    for name, sha in checks.UPSTREAM.items():
        if fit.file_hash(installed / name) != sha:
            raise ValueError("Installed native upstream changed: " + name)
    run(
        "pytest",
        [
            "-m",
            "pytest",
            "-q",
            *["tests/test_smolvla_" + name + ".py" for name in TESTS],
            "--junitxml=" + str(output / "pytest.xml"),
        ],
    )
    suite = ET.parse(output / "pytest.xml").getroot()
    if any(
        int(s.attrib.get(k, 0))
        for s in suite.iter("testsuite")
        for k in ("failures", "errors", "skipped")
    ):
        raise ValueError("CPU regressions contain failure, error or skipped cases")
    run("prepare", ["scripts/prepare_visual_fit.py", "--output", str(output / "draft")])
    run(
        "schedule",
        ["scripts/inspect_hestia_schedule.py", "--output", str(output / "schedule.json")],
    )
    template = json.loads((output / "draft/execution-contract.template.json").read_text())
    if template["status"] != "draft" or template["model_execution_authorized"] is not False:
        raise ValueError("Preparation unexpectedly authorized model execution")
    import evaluate_visual_fit

    try:
        evaluate_visual_fit.collect(
            output / "draft/execution-contract.template.json", "C", output / "forbidden-model"
        )
    except ValueError as error:
        if "sealed" not in str(error):
            raise
    else:
        raise ValueError("Draft collector was not rejected")
    if (output / "forbidden-model").exists():
        raise ValueError("Draft created model output")
    history = json.loads((ROOT / checks.HISTORY).read_text())
    identity = history["checkpoint_identities"]["B"]
    pretrained = checks.relative_file(
        Path(os.environ["ROSETTA_CHECKPOINT_ROOT"]), identity["checkpoint_relative_to_root"]
    )
    checkpoint = pretrained.parent
    checks.check_inventory(pretrained, identity["checkpoint_files"])
    checks.check_native_recovery(checkpoint, "B")
    plan_b = checks.relative_file(ROOT, identity["plan"]["path"])
    if fit.file_hash(plan_b) != identity["plan"]["sha256"]:
        raise ValueError("Restored B plan hash changed")
    b, _, experiment_b = _resolve_plan(plan_b)
    checks.validate_experiment(experiment_b)
    saved_b = json.loads((pretrained / "train_config.json").read_text())
    recipe = checks.validate_saved_recipe(saved_b, b, "B", identity)
    save(
        "B-saved-recipe.json",
        {
            "status": "passed",
            "recipe": recipe,
            "checkpoint_files": identity["checkpoint_files"],
            "processor_sha256": checks.processor_identity(identity["checkpoint_files"]),
            "native_recovery_step_scheduler": "passed",
            "optimizer_state_loaded": False,
        },
    )
    # Decode a real saved native configuration, then serialize a synthetic candidate.
    # This is a config round trip, not evidence that C ran or that its optimizer exists.
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    smoke = checkpoint.parent.parent.parent / "m2-smolvla450m-visual-hestia-smoke2-002"
    smoke = smoke / "checkpoints/000002/pretrained_model/train_config.json"
    saved_smoke = json.loads(smoke.read_text())
    draccus.decode(TrainPipelineConfig, saved_smoke)
    candidate_plan, _, candidate_experiment = _resolve_plan(output / "draft/main1280.yaml")
    checks.validate_experiment(candidate_experiment)
    synthetic = copy.deepcopy(saved_smoke)
    synthetic.update(
        job_name=fit.RUN_NAMES["C"],
        steps=1280,
        batch_size=4,
        save_freq=320,
        log_freq=16,
        seed=20260809,
        num_workers=0,
        resume=False,
    )
    synthetic["dataset"]["episodes"] = fit.TRAIN40[:]
    synthetic["optimizer"] = copy.deepcopy(fit.OPTIMIZER)
    synthetic["scheduler"] = checks.expected_scheduler("C")
    for name, value in {
        "optimizer_lr": 1e-4,
        "scheduler_warmup_steps": 16,
        "scheduler_decay_steps": 1280,
        "scheduler_decay_lr": 2.5e-6,
    }.items():
        synthetic["policy"][name] = value
    decoded = draccus.decode(TrainPipelineConfig, synthetic)
    if not isinstance(decoded.policy, SmolVLAConfig):
        raise ValueError("Native config decoded a different policy type")
    encoded = draccus.encode(decoded)
    # Encoding may represent tuples as lists; JSON round trip matches saved native format.
    roundtrip = json.loads(json.dumps(encoded))
    candidate_recipe = checks.validate_saved_recipe(
        roundtrip,
        candidate_plan,
        "C",
        {
            "run_name": fit.RUN_NAMES["C"],
            "checkpoint_files": {
                "train_config.json": hashlib.sha256(
                    json.dumps(roundtrip, sort_keys=True).encode()
                ).hexdigest()
            },
        },
    )
    # The hash above identifies synthetic config bytes, never an existing C checkpoint.
    candidate_recipe.pop("train_config_sha256")
    save(
        "native-config-roundtrip.json",
        {
            "status": "passed",
            "synthetic_candidate_only": True,
            "candidate_executed": False,
            "smoke_config_sha256": fit.file_hash(smoke),
            "candidate_recipe": candidate_recipe,
            "native_config_type": type(decoded).__name__,
            "weights_loaded": False,
        },
    )
    sizes = [p.stat().st_size for p in checkpoint.rglob("*") if p.is_file()]
    free = shutil.disk_usage(checkpoint).free
    minimum = 5 * sum(sizes) + 2 * 1024**3
    save(
        "disk.json",
        {
            "status": "passed" if free >= minimum else "insufficient_for_main",
            "available_bytes": free,
            "B_complete_checkpoint_bytes": sum(sizes),
            "new_checkpoint_copies": 5,
            "reserve_bytes": 2 * 1024**3,
            "minimum_new_bytes": minimum,
            "existing_smoke_counted_as_new": False,
            "estimate_requires_prelaunch_recheck": True,
        },
    )
    if source_identity() != source:
        raise ValueError("Source changed during CPU acceptance")
    save(
        "result.json",
        {
            "status": "passed",
            "tests": sum(int(s.attrib["tests"]) for s in suite.iter("testsuite")),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip(),
            "source_manifest_sha256": fit.file_hash(output / "source.json"),
            "upstream_files": checks.UPSTREAM,
            "packages": {
                p: version(p) for p in ("torch", "lerobot", "accelerate", "numpy", "draccus")
            },
            "runtime_profile_sha256": fit.file_hash(
                Path(os.environ["ROSETTA_AUTODL_RUNTIME_PROFILE"])
            ),
            "python": platform.python_version(),
            "nested_docker_used": False,
            "model_weights_loaded": False,
            "real_samples_loaded": False,
            "optimizer_steps": 0,
            "main_training": "not measured",
            "m2_complete": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.output.resolve())
