"""Create hash-bound, non-launchable coverage plans without loading models/data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REVIEW = "reports/training/m2-smolvla-native-visual-coverage40-plan-2026-09-10.json"
CONTROL = "configs/vla/m2-smolvla450m-visual-native-b4-pilot-003.yaml"
AUTHORITY = "reports/training/m2-smolvla-native-visual-authority-2026-09-10.json"
UPSTREAM = {
    "policies/smolvla/modeling_smolvla.py": "37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d",
    "datasets/sampler.py": "f715aaaa1118ef8901928f92975f7bc08bca412860f072bbcca84555303cc38e",
    "optim/schedulers.py": "05d57770348fbb3f412f52804a8d0f015f8128a37c55d5ce8ed9af6ed8522bc9",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> dict:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}


def build_plans(control: dict, review: dict) -> dict[str, dict]:
    """Only the active sample set differs in the main experimental arm."""
    if (control["scope"] != "bounded_visual_overfit" or control["optimizer_smoke"]["steps"] != 256
            or control["optimizer_smoke"]["batch_size"] != 4):
        raise ValueError("Unexpected control recipe.")
    train = review["data"]["train40"]
    if control["optimizer_smoke"]["episodes"] != train[:8] or control["training"]["episodes"] != train:
        raise ValueError("Historical active episode scope differs from registration.")
    result = {}
    for stage, batch, steps in (("preflight-b1", 1, 2), ("preflight-b4", 4, 2),
                                ("smoke2", 4, 2), ("main256", 4, 256)):
        value = copy.deepcopy(control)
        name = f"m2-smolvla450m-visual-coverage40-{stage}-001"
        value.update(status="pending_compute_authorization", plan_id=name, run_name=name,
            hypothesis="Increase frame-zero scene coverage from 8 to 40 at fixed 256 updates and 1024 exposures; repetition allocation also changes.",
            formal_training_claim=False, formal_training_authorized=False)
        value["optimizer_smoke"].update(run_name=name, episodes=train, batch_size=batch,
                                       steps=steps, save_freq=steps, log_freq=min(16, steps))
        value["preflight"].update(run_name=f"{name}-preflight", batch_size=batch)
        for feature in value["features"]:
            if feature["name"] == "fixed_frame_sampler":
                feature["sample_identities"] = [{"episode": ep, "frame": 0} for ep in train]
        value["stop_conditions"] = [
            "identity, split or registered sample mismatch",
            "nonfinite loss, gradient or action; illegal standard action",
            "CUDA allocated above 8 GiB or reserved above 10 GiB; host RSS above 10 GiB",
            "disk budget insufficient without deleting any old evidence",
            "fresh authorized shared deadline exhausted",
            "failed prerequisite or independent reload; no automatic retries or new run",
        ]
        result[stage] = value
    return result


def prepare(output: Path) -> dict:
    if not output.resolve().is_relative_to(ROOT):
        raise ValueError("Output must be inside the repository.")
    control = yaml.safe_load((ROOT / CONTROL).read_text())
    review = json.loads((ROOT / REVIEW).read_text())
    authority = json.loads((ROOT / AUTHORITY).read_text())
    for name, expected in control["implementation_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"The audited native source is not active: {name}.")
    output.mkdir(parents=True, exist_ok=False)
    stages = {name: save(output / f"{name}.json", plan)
              for name, plan in build_plans(control, review).items()}
    implementation = {**control["implementation_files"], **{name: digest(ROOT / name) for name in (
        "scripts/evaluate_visual_native_small.py", "scripts/evaluate_visual_coverage.py",
        "scripts/prepare_visual_coverage.py", "src/rosetta_reality/vla/visual_coverage.py",
        "src/rosetta_reality/vla/vision_diagnostics.py")}}
    checkpoint = authority["control_checkpoint"]
    files = {}
    for item in checkpoint["files"]:
        if "/pretrained_model/" in item["path"]:
            name = item["path"].split("/pretrained_model/", 1)[1]
            files[name] = item.get("sha256")
    for name, item in checkpoint["processor_stat_files"].items():
        files[name.removeprefix("pretrained_model/")] = item["sha256"]
    prefix = checkpoint["path_relative_to_durable_root"].removeprefix("checkpoints/")
    registration = {
        "schema_version": 1, "protocol": review["evaluation"]["protocol_id"],
        "status": "draft_no_compute_authorization", "model_execution_authorized": False,
        "deadline_unix": None, "review_plan": {"path": REVIEW, "sha256": digest(ROOT / REVIEW)},
        "implementation_files": implementation, "upstream_files": UPSTREAM, "stages": stages,
        "stage_order": ["doctor_and_static_checks", "benchmark", "preflight-b1", "preflight-b4",
                        "smoke2", "smoke2_independent_reload", "control_reproduction",
                        "main256_fresh_base", "frozen_and_optimizer_audit", "seal_checkpoints",
                        "A_collect", "A_fresh_process_collect", "A_reload_compare",
                        "B_collect", "B_fresh_process_collect", "B_reload_compare", "compare_arms"],
        "prerequisite_evidence": {name: None for name in (
            "environment", "sample_contract", "resource_preflight", "control_reproduction",
            "two_step_smoke_reload", "B_training_integrity")},
        "resource_limits": {"cuda_allocated_bytes": 8 * 1024**3,
            "cuda_reserved_bytes": 10 * 1024**3, "host_rss_bytes": 10 * 1024**3},
        "pending": ["fresh GPU/compute authorization and shared deadline/watchdog",
            "live device, disk budget, benchmark, preflight and smoke/reload",
            "remaining control tokenizer hash and actual consumed processor dependency inventory",
            "full sample/nonvisual contract and 1024-exposure schedule before optimization",
            "fresh B checkpoint identity and frozen/updated tensor audit after training"],
        "arms": {
            "A": {"run_name": control["run_name"], "plan": {"path": CONTROL, "sha256": digest(ROOT / CONTROL)},
                  "checkpoint_relative_to_root": prefix + "/pretrained_model", "checkpoint_files": files},
            "B": {"run_name": build_plans(control, review)["main256"]["run_name"],
                  "plan": stages["main256"], "checkpoint_relative_to_root":
                  f"{control['parent_experiment']['experiment_id']}/smoke/"
                  "m2-smolvla450m-visual-coverage40-main256-001/checkpoints/000256/pretrained_model",
                  "checkpoint_files": None}},
        "main_phase": "smoke", "B_initialization": "revision_pinned_base_model",
        "A_retrained": False, "old_optimizer_resumed": False,
        "sampler_exposures": 1024, "equal_wall_time_claim": False, "m2_complete": False,
    }
    save(output / "execution-contract.template.json", registration)
    return {"status": registration["status"], "stages": stages, "model_loaded": False,
            "data_loaded": False, "optimizer_steps": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output.resolve())))


if __name__ == "__main__":
    main()
