"""Fail-closed post-training adapter for the canonical full-frame endpoint.

The recovery window uses this module only in ``validate-recovery`` mode. That
mode reads manifests, schedules, checkpoint inventories and reload evidence;
it never imports a model runtime or opens a dataset. Model evaluation/export
and CUDA Gates are separate stages and require the sealed post-training plan
plus a separately admitted CUDA worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    ROOT
    / "reports/training/m2-smolvla-canonical-fullframes-posttrain-gate-plan-004-2026-09-15.json"
)
DEFAULT_RECOVERY = ROOT / "runs/canonical-furnace-received-20260914-002"
DEFAULT_CHECKPOINTS = ROOT / "runs/canonical-checkpoints-received-20260914-002"
DEFAULT_RECEIVER_RECEIPT = (
    ROOT / "runs/canonical-furnace-preparation-20260914-001/luna-monitor/recovery-receipt-002.json"
)

TRAIN_EPISODES = [
    49,
    4,
    23,
    43,
    21,
    37,
    18,
    34,
    0,
    47,
    38,
    29,
    3,
    26,
    14,
    17,
    44,
    30,
    15,
    42,
    10,
    35,
    25,
    32,
    19,
    36,
    41,
    28,
    8,
    27,
    16,
    11,
    2,
    20,
    9,
    39,
    46,
    48,
    12,
    40,
]
DEV_EPISODES = [22, 13, 7, 33, 45]
HIDDEN_EPISODES = {31, 6, 1, 24, 5}
HIDDEN_EPISODES_ORDER = [31, 6, 1, 24, 5]
FRAME_OFFSETS = [0, 125, 250, 375]
GATE4_SEEDS = [1000, 1001, 1002, 1003, 1004]
EXPECTED_ENGINE_SHA = "33ee39c897232baa876dfd55a6e76a7b1a9373784293f75f1a5778886a6147cd"
EXPECTED_PROTOCOL_SHA = "20386f7ad5dda19d9d5ffa1668bfa45df5ae3f6441d94f45661eb9f10fa442f3"
EXPECTED_CONTRACT_SHA = "fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    json.dumps(value, allow_nan=False)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_relative(raw: str) -> Path:
    path = Path(raw)
    _require(not path.is_absolute() and ".." not in path.parts, "Unsafe manifest path")
    return path


def expected_training_schedule() -> list[list[int]]:
    samples = [[episode, frame] for episode in TRAIN_EPISODES for frame in range(500)]
    random.Random(20260809).shuffle(samples)
    return samples


def validate_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    plan = _load_json(path)
    _require(
        plan.get("plan_id") == "canonical-fullframes-20260914-001-posttrain-004",
        "Post-training plan id drift",
    )
    _require(plan.get("status") == "preregistered", "Post-training plan is not preregistered")
    _require(plan.get("fixed_endpoint_step") == 5000, "Endpoint must be fixed at step 5000")
    _require(plan.get("optimizer_updates") == 0, "Post-training adapter cannot update an optimizer")
    _require(plan.get("hidden_test_loaded") is False, "Hidden-test boundary is not sealed")
    source_identity = plan.get("source_identity")
    _require(isinstance(source_identity, dict), "Source identity binding is missing")
    for key, relative, expected in (
        (
            "training_config_sha256",
            "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml",
            "0e9dd0499d0708939ac73cc5d517849f133cf6deab072d9cde09f2880ae22210",
        ),
        (
            "normalization_report_sha256",
            "runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/normalization/train-only-3e3c6b9d347e5e71.json",
            "263880ec3adfddb8517a50fa5483e7c8f32f0c208243c229cbc72f8e9cf8d988",
        ),
        (
            "dataset_view_manifest_sha256",
            "runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/dataset_views/train-only-3e3c6b9d347e5e71/view_manifest.json",
            "9853c191ae87016379fc1a16ebfbb87e05ab5147cd8a03d82f9c2a894c9b531e",
        ),
    ):
        source = ROOT / relative
        _require(source_identity.get(key) == expected, f"Source identity hash drift: {key}")
        _require(
            source.is_file() and _sha256(source) == expected,
            f"Source identity source changed: {relative}",
        )
    _require(
        source_identity.get("native_train_config_sha256")
        == "30f2c2b014ef3aab2d2b7ec3081ce857b25a0172a0f9a5601b17829e3c27093b",
        "Native train config evidence hash drift",
    )
    _require(
        source_identity.get("checkpoint_seal_sha256")
        == "ac87bdf2d3ece4240a507f1536185ae69969dba05caf3976d11b67178fa3bb30",
        "Checkpoint seal evidence hash drift",
    )
    _require(
        source_identity.get("reload_proof_sha256")
        == "0fa1a4d7c03367f6bc73eb5c11f16d31ba5b9356d3fd099cafd74840d5473615",
        "Reload proof evidence hash drift",
    )
    endpoint = plan.get("endpoint")
    _require(isinstance(endpoint, dict), "Fixed endpoint identity is missing")
    _require(endpoint.get("checkpoint_step") == 5000, "Endpoint checkpoint step drift")
    _require(
        endpoint.get("model_safetensors_sha256")
        == "d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef",
        "Endpoint model hash drift",
    )
    _require(
        endpoint.get("pretrained_model_bytes") == 1201362478
        and endpoint.get("model_bytes") == 1197789256,
        "Endpoint byte budget drift",
    )
    _require(
        endpoint.get("zero_copy_artifact") is True, "Endpoint must use zero-copy artifact admission"
    )
    implementation = plan.get("implementation")
    _require(
        isinstance(implementation, dict) and implementation,
        "Post-training implementation inventory is missing",
    )
    for relative, expected in implementation.items():
        path = ROOT / relative
        _require(
            path.is_file() and _sha256(path) == expected,
            f"Post-training implementation hash drift: {relative}",
        )
    data = plan.get("data")
    _require(isinstance(data, dict), "Post-training data contract is missing")
    _require(
        data.get("model_revision") == "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        "Model revision drift",
    )
    _require(
        data.get("dataset_revision") == "cc571a3c661df81b566dbfde3d5c1e85fcdf7884",
        "Dataset revision drift",
    )
    _require(
        data.get("sealed_hidden_episodes") == HIDDEN_EPISODES_ORDER,
        "Hidden episode allowlist drift",
    )
    offline = plan.get("offline")
    _require(isinstance(offline, dict), "Offline contract is missing")
    _require(
        offline.get("train_episodes_in_order") == TRAIN_EPISODES, "Train episode allowlist drift"
    )
    _require(
        offline.get("development_episodes_in_order") == DEV_EPISODES,
        "Development episode allowlist drift",
    )
    _require(offline.get("frame_offsets") == FRAME_OFFSETS, "Offline frame allowlist drift")
    _require(offline.get("train_sample_count") == 160, "Train offline cohort size drift")
    _require(offline.get("development_sample_count") == 20, "Development offline cohort size drift")
    _require(offline.get("primary_noise") == "zeros", "Primary offline noise drift")
    _require(offline.get("primary_flow_time") == 0.5, "Primary offline flow time drift")
    _require(
        offline.get("diagnostic_policy_noise_seeds") == GATE4_SEEDS, "Diagnostic noise seed drift"
    )
    _require(offline.get("hidden_test_loaded") is False, "Offline hidden-test boundary drift")
    gates = plan.get("gate_protocol_authority")
    _require(isinstance(gates, dict), "Gate protocol authority is missing")
    for key, expected in (
        ("engine_sha256", EXPECTED_ENGINE_SHA),
        ("protocol_config_sha256", EXPECTED_PROTOCOL_SHA),
        ("action_contract_sha256", EXPECTED_CONTRACT_SHA),
    ):
        _require(gates.get(key) == expected, f"Gate authority hash drift: {key}")
    _require(gates.get("actual_accelerator") == "cuda", "Gate execution must be CUDA")
    runtime = plan.get("posttrain_runtime")
    _require(isinstance(runtime, dict), "CUDA post-training runtime budget is missing")
    _require(
        runtime.get("requires_separate_cuda_window") is True, "CUDA window boundary is missing"
    )
    _require(runtime.get("nested_docker_used") is False, "Nested Docker is forbidden")
    _require(runtime.get("work_deadline_seconds") == 3600, "Post-training deadline drift")
    _require(runtime.get("protected_shutdown_seconds") == 4200, "Shutdown deadline drift")
    _require(
        runtime.get("minimum_durable_free_bytes") == 1073741824, "Durable free-space budget drift"
    )
    storage = runtime.get("storage_budget_basis")
    _require(isinstance(storage, dict), "Storage budget basis is missing")
    _require(
        storage.get("required_free_bytes") == 1073741824, "Storage budget required bytes drift"
    )
    _require(
        storage.get("source_pretrained_model_bytes_not_duplicated") == 1201362478,
        "Zero-copy storage basis drift",
    )
    _require(
        storage.get("artifact_metadata_max_bytes") == 1048576, "Artifact metadata budget drift"
    )
    _require(storage.get("offline_reports_max_bytes") == 16777216, "Offline report budget drift")
    _require(
        storage.get("gate_reports_and_episode_metrics_max_bytes") == 33554432,
        "Gate report budget drift",
    )
    _require(
        storage.get("simulation_recordings_max_bytes") == 0
        and storage.get("trajectory_artifacts_max_bytes") == 0,
        "Simulation recording policy drift",
    )
    _require(storage.get("runtime_scratch_max_bytes") == 268435456, "Runtime scratch budget drift")
    _require(storage.get("runner_logs_max_bytes") == 16777216, "Runner log budget drift")
    _require(
        storage.get("shutdown_metadata_max_bytes") == 1048576, "Shutdown metadata budget drift"
    )
    _require(storage.get("safety_reserve_bytes") == 736100352, "Storage safety reserve drift")
    _require(
        storage.get("full_checkpoint_copy_forbidden") is True,
        "Full checkpoint copy must remain forbidden",
    )
    gate4 = plan.get("gate4")
    _require(isinstance(gate4, dict), "Gate 4 contract is missing")
    _require(gate4.get("seeds") == GATE4_SEEDS, "Gate 4 seed allowlist drift")
    _require(gate4.get("policy_noise_seeds") == GATE4_SEEDS, "Gate 4 policy-noise allowlist drift")
    _require(gate4.get("maximum_steps") == 500, "Gate 4 horizon drift")
    _require(gate4.get("minimum_task_success_rate") == 0.2, "Gate 4 success threshold drift")
    _require(gate4.get("maximum_unexpected_collisions") == 0, "Gate 4 collision threshold drift")
    for relative, expected in (
        (gates["engine"], EXPECTED_ENGINE_SHA),
        (gates["protocol_config"], EXPECTED_PROTOCOL_SHA),
        (gates["action_contract"], EXPECTED_CONTRACT_SHA),
    ):
        source = ROOT / relative
        _require(source.is_file(), f"Gate authority source is missing: {relative}")
        _require(_sha256(source) == expected, f"Gate authority source changed: {relative}")
    return plan


def _validate_manifest(recovery_root: Path, receiver_receipt_path: Path) -> dict[str, Any]:
    manifest_path = recovery_root / "runs/canonical-fullframes-20260914-001/handoff-manifest.json"
    manifest = _load_json(manifest_path)
    _require(
        _sha256(manifest_path)
        == "66b8e8dfdb5fe33f5b1ddb03d035280fcd52570a1c331d37d7c66deed3627f96",
        "Original handoff seal drift",
    )
    files = manifest.get("files")
    _require(isinstance(files, dict) and files, "Handoff file manifest is empty")
    checked = 0
    for raw, record in files.items():
        relative = _safe_relative(raw)
        local = recovery_root / relative
        _require(local.is_file(), f"Recovered handoff member is missing: {raw}")
        _require(isinstance(record, dict), f"Malformed handoff record: {raw}")
        _require(local.stat().st_size == record.get("bytes"), f"Recovered size drift: {raw}")
        _require(_sha256(local) == record.get("sha256"), f"Recovered SHA drift: {raw}")
        checked += 1
    _require(
        manifest.get("model_weights_included") is False,
        "Handoff unexpectedly claims checkpoint weights",
    )
    remote_receipt_path = (
        recovery_root / "runs/canonical-fullframes-20260914-001/transfer-receipt.json"
    )
    remote_receipt_present = remote_receipt_path.is_file()
    receiver_receipt_present = receiver_receipt_path.is_file()
    receiver_receipt_verified = False
    manifest_sha256 = _sha256(manifest_path)
    if receiver_receipt_present:
        receipt = _load_json(receiver_receipt_path)
        _require(
            receipt.get("kind") == "local_receiver_manifest", "Local receiver receipt kind drift"
        )
        _require(receipt.get("verified") is True, "Local receiver receipt is not verified")
        _require(
            isinstance(receipt.get("verification_time_utc"), str)
            and receipt["verification_time_utc"],
            "Local receiver verification time is missing",
        )
        _require(
            receipt.get("manifest_sha256") == manifest_sha256,
            "Local receiver receipt manifest SHA drift",
        )
        _require(receipt.get("files_checked") == checked, "Local receiver receipt file count drift")
        _require(
            receipt.get("recovered_handoff_root") == "runs/canonical-furnace-received-20260914-002",
            "Local receiver handoff path drift",
        )
        _require(
            receipt.get("remote_receipt_present") is False,
            "Local receiver receipt must state remote receipt absence",
        )
        _require(
            receipt.get("remote_not_rewritten") is True,
            "Local receiver receipt must state remote was not rewritten",
        )
        _require(
            receipt.get("checkpoint_backup_paths")
            == [
                "runs/canonical-checkpoints-received-20260914-002/step-002500/verified",
                "runs/canonical-checkpoints-received-20260914-002/step-005000/verified",
            ],
            "Local receiver checkpoint paths drift",
        )
        receiver_receipt_verified = True
    return {
        "manifest_sha256": manifest_sha256,
        "files_checked": checked,
        "remote_transfer_receipt_present": remote_receipt_present,
        "receiver_receipt_path": str(receiver_receipt_path.relative_to(ROOT)).replace("\\", "/"),
        "receiver_receipt_present": receiver_receipt_present,
        "receiver_receipt_verified": receiver_receipt_verified,
    }


def _validate_schedule(job: Path) -> dict[str, Any]:
    schedule = _load_json(job / "schedule.json")
    samples = schedule.get("sample_identities")
    _require(schedule.get("seed") == 20260809, "Training schedule seed drift")
    _require(samples == expected_training_schedule(), "Training schedule order or coverage drift")
    _require(
        len(samples) == 20000 and len({tuple(pair) for pair in samples}) == 20000,
        "Training schedule is not unique",
    )
    return {
        "seed": schedule["seed"],
        "sample_count": len(samples),
        "schedule_sha256": _sha256(job / "schedule.json"),
    }


def _validate_ingress(job: Path) -> dict[str, Any]:
    ingress = _load_json(job / "model-ingress.json")
    _require(ingress.get("status") == "passed", "Model-ingress seal did not pass")
    _require(ingress.get("optimizer_steps") == 5000, "Optimizer update count drift")
    _require(ingress.get("unique_input_frames") == 20000, "Unique input coverage drift")
    records = ingress.get("records")
    _require(
        isinstance(records, list) and len(records) == 20000, "Model-ingress record count drift"
    )
    _require(
        [[row["episode"], row["frame"]] for row in records] == expected_training_schedule(),
        "Actual ingress differs from registered order or split",
    )
    pairs = {(int(row["episode"]), int(row["frame"])) for row in records}
    _require(len(pairs) == 20000, "Model-ingress records contain duplicate input frames")
    _require(
        all(0 <= ep < 50 and 0 <= frame < 500 for ep, frame in pairs),
        "Model-ingress pair out of range",
    )
    return {"optimizer_steps": ingress["optimizer_steps"], "unique_input_frames": len(pairs)}


def _validate_recovery_checkpoints(job: Path, checkpoints: Path) -> dict[str, Any]:
    checkpoint = _load_json(job / "checkpoint.json")
    _require(checkpoint.get("steps") == 5000, "Checkpoint seal endpoint drift")
    _require(checkpoint.get("frozen_unchanged") is True, "Frozen tensor seal failed")
    recovery = checkpoint.get("recovery_checkpoints")
    _require(
        isinstance(recovery, dict) and set(recovery) == {"2500", "5000"},
        "Recovery checkpoint set drift",
    )
    checked: dict[str, int] = {}
    for step in ("2500", "5000"):
        root = checkpoints / f"step-00{step}" / "verified"
        sums = checkpoints / f"step-00{step}" / "sha256sums.txt"
        _require(root.is_dir() and sums.is_file(), f"Checkpoint backup is missing for step {step}")
        count = 0
        checked_files = {}
        for line in sums.read_text(encoding="utf-8").splitlines():
            digest, raw = line.split(maxsplit=1)
            relative = _safe_relative(raw.removeprefix("./"))
            path = root / relative
            _require(
                path.is_file() and _sha256(path) == digest,
                f"Checkpoint SHA drift: {step}/{relative}",
            )
            checked_files[relative.as_posix()] = digest
            count += 1
        _require(checked_files == recovery[step], "Recovery inventory differs from training seal")
        _require(
            {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
            == set(checked_files),
            "Recovery file set incomplete",
        )
        _require(count > 0, f"Checkpoint checksum set is empty: {step}")
        checked[step] = count
    return {"checkpoint_steps": [2500, 5000], "files_checked": checked}


def _validate_worker(job: Path) -> dict[str, Any]:
    worker = _load_json(job / "worker-exited.json")
    stages = worker.get("stages")
    expected = [
        "prepare",
        "doctor",
        "benchmark",
        "preflight",
        "parity",
        "train",
        "collect-first",
        "collect-reload",
        "verify-reload",
    ]
    _require(isinstance(stages, list), "Worker stage evidence is missing")
    names = [row.get("stage") for row in stages if isinstance(row, dict)]
    _require(names == expected, "Worker stage order or set drift")
    _require(
        all(isinstance(row, dict) and row.get("exit_code") == 0 for row in stages),
        "Worker stage failed",
    )
    _require(worker.get("error") is None, "Worker reported an error")
    return {
        "stage_count": len(stages),
        "stages": names,
        "train_seconds": stages[5]["seconds"],
        "m2_complete": worker.get("m2_complete"),
    }


def _validate_reload(job: Path) -> dict[str, Any]:
    proof = _load_json(job / "reload-proof.json")
    _require(proof.get("status") == "passed", "Reload proof did not pass")
    _require(proof.get("exact_tensor_equality") is True, "Reload arrays are not exactly equal")
    arrays_equal = proof.get("arrays_equal")
    required_arrays = {
        "noise",
        "normalized_actions",
        "sample_identities",
        "standard_actions",
        "valid_mask",
    }
    _require(isinstance(arrays_equal, dict), "Reload array equality evidence is missing")
    _require(
        set(arrays_equal) == required_arrays and all(arrays_equal.values()),
        "Reload array equality is incomplete",
    )
    _require(
        proof.get("proof_scope") == "saved_full_arrays_from_distinct_collection_processes",
        "Reload proof scope drift",
    )
    _require(
        proof.get("model_execution_proven_by_this_comparison") is False,
        "Reload proof overclaims model execution",
    )
    identity = proof.get("identity")
    _require(isinstance(identity, dict), "Reload identity is missing")
    _require(identity.get("sample_count") == 16, "Reload sample count drift")
    _require(identity.get("chunk_size") == 50, "Reload chunk size drift")
    _require(
        identity.get("normalized_action_dim") == 14, "Reload normalized action dimension drift"
    )
    _require(identity.get("standard_action_dim") == 14, "Reload standard action dimension drift")
    _require(identity.get("noise_action_dim") == 32, "Reload noise action dimension drift")

    manifests = []
    for label in ("collect-first", "collect-reload"):
        manifest = _load_json(job / label / "manifest.json")
        _require(manifest.get("identity") == identity, f"{label} reload identity drift")
        _require(isinstance(manifest.get("process"), dict), f"{label} process identity is missing")
        _require(
            manifest.get("arrays_sha256")
            == "e2b900f51a0890e9680a2b87da543dbadc40da46d16efed261498d50fb97e531",
            f"{label} arrays digest drift",
        )
        manifests.append(manifest)
    _require(
        manifests[0]["process"]["id"] != manifests[1]["process"]["id"],
        "Reload processes are not distinct",
    )
    _require(
        manifests[0]["process"]["pid"] != manifests[1]["process"]["pid"],
        "Reload PIDs are not distinct",
    )

    executions = []
    for label in ("collect-first", "collect-reload"):
        execution = _load_json(job / f"{label}-execution.json")
        _require(execution.get("model_loaded") is True, f"{label} did not load the model")
        _require(
            execution.get("saved_processors_loaded") is True,
            f"{label} did not load saved processors",
        )
        _require(execution.get("forwards") == 16, f"{label} forward count drift")
        _require(execution.get("parameters_unchanged") is True, f"{label} changed parameters")
        _require(execution.get("optimizer_steps") == 0, f"{label} performed optimizer updates")
        _require(
            execution["pid"] == manifests[len(executions)]["process"]["pid"],
            "Model execution PID differs from array collector",
        )
        executions.append(execution)
    return {
        "status": proof["status"],
        "exact_tensor_equality": proof["exact_tensor_equality"],
        "distinct_processes": True,
        "process_ids": [manifest["process"]["id"] for manifest in manifests],
        "pids": [manifest["process"]["pid"] for manifest in manifests],
        "forwards_per_process": [execution["forwards"] for execution in executions],
        "model_loaded_per_process": [execution["model_loaded"] for execution in executions],
        "saved_processors_loaded_per_process": [
            execution["saved_processors_loaded"] for execution in executions
        ],
        "optimizer_steps_per_process": [execution["optimizer_steps"] for execution in executions],
        "coverage": {
            "sample_count": identity["sample_count"],
            "chunk_size": identity["chunk_size"],
            "normalized_action_dim": identity["normalized_action_dim"],
            "standard_action_dim": identity["standard_action_dim"],
            "noise_action_dim": identity["noise_action_dim"],
            "arrays_sha256": manifests[0]["arrays_sha256"],
        },
    }


def validate_recovery(
    plan_path: Path,
    recovery_root: Path,
    checkpoints: Path,
    receiver_receipt_path: Path = DEFAULT_RECEIVER_RECEIPT,
) -> dict[str, Any]:
    plan = validate_plan(plan_path)
    job = recovery_root / "runs/canonical-fullframes-20260914-001"
    handoff = _validate_manifest(recovery_root, receiver_receipt_path)
    schedule = _validate_schedule(job)
    ingress = _validate_ingress(job)
    checkpoint = _validate_recovery_checkpoints(job, checkpoints)
    worker = _validate_worker(job)
    reload = _validate_reload(job)
    receipt_verified = handoff["receiver_receipt_verified"]
    return {
        "schema_version": 1,
        "status": "passed" if receipt_verified else "blocked_receiver_receipt_missing",
        "stage": "canonical_fullframes_posttrain_recovery_preflight",
        "plan_id": plan["plan_id"],
        "fixed_endpoint_step": plan["fixed_endpoint_step"],
        "hidden_test_loaded": False,
        "handoff": handoff,
        "worker": worker,
        "schedule": schedule,
        "model_ingress": ingress,
        "checkpoints": checkpoint,
        "reload": reload,
        "optimizer_updates": 0,
        "model_forwards": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-plan", "validate-recovery"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--receiver-receipt", type=Path, default=DEFAULT_RECEIVER_RECEIPT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "validate-plan":
        result = {"status": "passed", "plan_id": validate_plan(args.plan.resolve())["plan_id"]}
    else:
        result = validate_recovery(
            args.plan.resolve(),
            args.recovery_root.resolve(),
            args.checkpoint_root.resolve(),
            args.receiver_receipt.resolve(),
        )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Create-only output exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
