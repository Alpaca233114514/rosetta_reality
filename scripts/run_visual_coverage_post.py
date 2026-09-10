"""Hermes: evaluate existing A/B checkpoints after a recorded prefetch audit failure."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import run_visual_coverage_job as j

OLD = j.ROOT / "runs/visual-coverage40-unattended-003"
j.JOB = j.ROOT / "runs/visual-hermes-coverage40-eval-001"
j.SELF = Path(__file__).resolve()


def verify_sample_observation(observed, expected, *, consumed, batch_size):
    if len(observed) != consumed + batch_size or observed != expected[: len(observed)]:
        raise ValueError("Sampler observation differs from consumed plus one prefetched batch")
    return {"consumed": consumed, "prefetched_unused": batch_size, "exact_order": True}


def worker():
    stage = "post-training-integrity"
    try:
        import numpy as np
        import torch
        from accelerate.data_loader import DataLoaderShard
        import accelerate.data_loader as loader
        import lerobot.scripts.lerobot_train as trainer

        assert j.digest(loader.__file__) == "e08c5f5cd3f7a8b8aaece03ad51f1c37994e390faf47d368281ad49c31bc84d1"
        assert j.digest(trainer.__file__) == "4d15d283ea54583f552b32088db0b6c195250905ca6daf06d4670383790e2059"
        observed = []

        class Sampler:
            def __iter__(self):
                for index in range(1040):
                    observed.append(index)
                    yield index

            def __len__(self):
                return 1040

        synthetic = iter(DataLoaderShard(list(range(1040)), batch_size=4, sampler=Sampler(), num_workers=0))
        consumed = []
        for _ in range(256):
            consumed.extend(next(synthetic).tolist())
        assert consumed == list(range(1024)) and observed == list(range(1028))
        samples = j.load(OLD / "sample-contract.json")
        episodes = samples["actual_train_episodes"]
        mapping = dict(zip(episodes, samples["actual_train_view_indices"], strict=True))
        expected = []
        for epoch in range(26):
            seed = int(np.random.SeedSequence([20260809, epoch]).generate_state(1, dtype=np.uint64)[0])
            expected.extend(mapping[episodes[i]] for i in torch.randperm(40, generator=torch.Generator().manual_seed(seed)).tolist())
        resources = j.load(OLD / "main256-resources.json")
        proof = verify_sample_observation(resources["sampled_indices"], expected, consumed=1024, batch_size=4)
        assert resources["status"] == "passed" and len(resources["learning_rates"]) == 256
        for index, values in enumerate(resources["learning_rates"]):
            factor = ((1 / 17 - 1) * (1 - index / 16) + 1) if index < 16 else (0.975 * 0.5 * (1 + math.cos(math.pi * index / 256)) + 0.025)
            assert len(values) == 1 and math.isclose(values[0], 1e-4 * factor, rel_tol=1e-12, abs_tol=1e-16)
        plan = OLD / "plans/main256.yaml"
        name = j.load(plan)["optimizer_smoke"]["run_name"]
        cp = Path(os.environ["ROSETTA_CHECKPOINT_ROOT"]) / j.EXP / "smoke" / name / "checkpoints/000256"
        state = j.load(cp / "training_state/scheduler_state.json")
        assert state["last_epoch"] == 256 and state["_step_count"] == 257 and state["_last_lr"] == [2.5e-6]
        metric = j.load(cp / "rosetta_checkpoint_metrics.json")
        assert metric["step"] == 256 and all(math.isfinite(v) for v in metric["metrics"].values())
        saved = j.load(cp / "pretrained_model/train_config.json")
        contract = j.load(OLD / "A-before-B-contract.json")
        control = Path(os.environ["ROSETTA_CHECKPOINT_ROOT"]) / contract["arms"]["A"]["checkpoint_relative_to_root"]
        old_saved = j.load(control / "train_config.json")
        for key in ("optimizer", "scheduler", "seed", "steps", "batch_size"):
            assert saved[key] == old_saved[key], key
        assert saved["num_workers"] == 0 and saved["dataset"]["episodes"] == episodes
        proof.update(status="passed", source_training_code="e11c7aaddc8c1eab84365d399ef8696cb5d0f83b", synthetic_accelerate_prefetch_verified=True, resources=j.evidence(OLD / "main256-resources.json"), update_audit=j.evidence(OLD / "B-update-audit.json"), additional_optimizer_steps=0)
        j.save(j.JOB / "accepted-B_training_integrity.json", proof)
        watch = j.load(j.JOB / "watchdog.json")
        contract.update(started_unix=watch["started_unix"], deadline_unix=watch["deadline_unix"])
        contract["implementation_files"]["scripts/run_visual_coverage_post.py"] = j.digest(j.SELF)
        contract["prerequisite_evidence"]["B_training_integrity"] = j.evidence(j.JOB / "accepted-B_training_integrity.json")
        contract["arms"]["B"].update(plan={"path": plan.relative_to(j.ROOT).as_posix(), "sha256": j.digest(plan)}, run_name=name, checkpoint_relative_to_root=(cp / "pretrained_model").relative_to(Path(os.environ["ROSETTA_CHECKPOINT_ROOT"])).as_posix())
        for arm in ("A", "B"):
            contract["arms"][arm]["checkpoint_files"] = j.inventory(Path(os.environ["ROSETTA_CHECKPOINT_ROOT"]) / contract["arms"][arm]["checkpoint_relative_to_root"])
        sealed = j.JOB / "execution-contract.json"
        j.save(sealed, contract)

        def run(name, args, heavy=False):
            nonlocal stage
            stage = name
            j.event("stage_started", stage=name)
            command = [sys.executable, "scripts/evaluate_visual_coverage.py", *args]
            if heavy:
                command = [sys.executable, "scripts/run_visual_coverage_job.py", "wrap", str(j.JOB / f"{name}-resources.json"), *command[1:]]
            with (j.JOB / f"{name}.log").open("x") as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=max(1, watch["deadline_unix"] - time.time()))
            j.event("stage_passed", stage=name)

        for arm in ("A", "B"):
            for suffix in ("first", "reload"):
                run(f"{arm}-{suffix}", ["collect", "--contract", str(sealed), "--arm", arm, "--output", str(j.JOB / f"{arm}-{suffix}")], True)
            run(f"{arm}-reload-check", ["reload", "--first", str(j.JOB / f"{arm}-first"), "--second", str(j.JOB / f"{arm}-reload"), "--output", str(j.JOB / f"{arm}-reload-check.json")])
        run("compare-arms", ["compare", "--first", str(j.JOB / "A-first"), "--second", str(j.JOB / "B-first"), "--output", str(j.JOB / "comparison.json")])
        j.save(j.JOB / "closure.json", {"status": "completed", "m2_complete": False, "additional_optimizer_steps": 0})
    except BaseException:
        j.save(j.JOB / "closure.json", {"status": "stopped", "stage": stage, "error": traceback.format_exc(), "m2_complete": False, "additional_optimizer_steps": 0})
        raise
    finally:
        os.sync()


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        source = Path(sys.argv[2])
        assert not OLD.exists() and not j.JOB.exists()
        shutil.copytree(source / OLD.relative_to(j.ROOT), OLD)
        shutil.copytree(source / "runs/visual-native-small-001", j.ROOT / "runs/visual-native-small-001")
        j.save(j.JOB / "registration.json", {"job_sources": {str(j.SELF.relative_to(j.ROOT)): j.digest(j.SELF)}, "optimizer_steps": 0, "budget_seconds": 1800})
    elif sys.argv[1] == "supervise":
        j.supervise()
    elif sys.argv[1] == "worker":
        worker()
    else:
        raise ValueError("Unknown mode")
