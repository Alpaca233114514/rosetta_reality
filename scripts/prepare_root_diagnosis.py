"""Create-only evidence inventory and non-executable reproducibility registrations."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from analyze_root_neighborhoods import ROOT, seal, sha, write

sys.path.insert(0, str(ROOT / "src"))
from rosetta_reality.eval.gate_diagnostic_protocol import (  # noqa: E402
    REQUIRED_SOURCES,
    artifact_config_digest,
    validate_plan,
)

FURNACE = "runs/canonical-furnace-received-20260914-002/runs/canonical-fullframes-20260914-001"
CHECKPOINTS = "runs/canonical-checkpoints-received-20260914-002"
GATES = {
    "004": "runs/canonical-posttrain-received-20260915-004/verified",
    "005": "runs/canonical-posttrain-received-20260916-005-ssh/verified",
}
GATE_SUBDIR = "results/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/gates"


def load(relative):
    return json.loads((ROOT / relative).read_text())


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    records = []

    def record(name, kind, expected=None):
        path = ROOT / name
        row = {
            "path": name,
            "kind": kind,
            "expected_sha256": expected,
            "status": "missing",
            "scientific_execution": "not established by file existence",
        }
        if path.is_file():
            actual = sha(path)
            row.update(
                bytes=path.stat().st_size,
                sha256=actual,
                status="file_verified"
                if expected == actual
                else "file_hashed_no_prior_seal"
                if expected is None
                else "hash_mismatch",
            )
        records.append(row)
        return row

    checkpoint = load(FURNACE + "/checkpoint.json")
    for step, files in checkpoint["recovery_checkpoints"].items():
        for name, expected in files.items():
            record(
                f"{CHECKPOINTS}/step-{int(step):06d}/verified/{name}",
                "checkpoint_" + step,
                expected,
            )
    for name in (
        "checkpoint.json",
        "schedule.json",
        "model-ingress.json",
        "reload-proof.json",
        "smoke.yaml",
        "registration.json",
        "worker-exited.json",
        "observation/result.json",
    ):
        record(FURNACE + "/" + name, "historical_training_evidence")
    for attempt, prefix in GATES.items():
        for path in sorted((ROOT / prefix).rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".npz"):
                record(path.relative_to(ROOT).as_posix(), "historical_gate_" + attempt)
    for directory in (
        "runs/visual-research-package-20260915-001/data",
        "runs/visual-research-package-20260915-001/supervision",
    ):
        manifest = load(directory + "/manifest.json")
        record(directory + "/manifest.json", "saved_data_manifest")
        for name, entry in manifest["files"].items():
            record(directory + "/" + name, "saved_data", entry["sha256"])
    paired = "runs/visual-research-process-comparison-20260916-001"
    for path in sorted((ROOT / paired).iterdir()):
        if path.is_file():
            record(path.relative_to(ROOT).as_posix(), "paired_prediction_evidence")
    # The draft records historical source identities separately from current source seals.
    source_names = set(REQUIRED_SOURCES) | {
        "scripts/diagnose_smolvla_gate.py",
        "scripts/smolvla_sim_gate.py",
        "src/rosetta_reality/eval/reproducibility.py",
        "src/rosetta_reality/eval/reproducibility_capture.py",
        "src/rosetta_reality/eval/gate_diagnostic_protocol.py",
        "src/rosetta_reality/eval/gate_diagnostic_capture.py",
        "src/rosetta_reality/eval/gate_diagnostic_io.py",
        "src/rosetta_reality/eval/rollout_trace.py",
        "src/rosetta_reality/eval/rollout_trace_verify.py",
    }
    current_sources = {name: sha(ROOT / name) for name in sorted(source_names)}
    for name, value in current_sources.items():
        record(name, "current_implementation", value)
    checkpoint_dir = f"{CHECKPOINTS}/step-005000/verified/pretrained_model"
    historical = load("configs/vla/closed_loop_reproducibility_001.json")
    binding_gaps = []
    repro = output / "reproducibility"
    repro.mkdir()
    for role in ("baseline_a", "baseline_b", "full_trace"):
        plan = copy.deepcopy(historical)
        plan.update(
            run_id="closed-loop-root-20260922-002-" + role.replace("_", "-"),
            status="draft",
            execution=None,
        )
        plan["output"] = "runs/" + plan["run_id"]
        plan["rollout"].update(seed=1002, policy_noise_seed=1002)
        plan["reproducibility"] = {
            "pair_id": "closed-loop-root-20260922-002",
            "role": role,
            "seed": 1002,
        }
        plan["sources"] = current_sources
        plan["checkpoint"]["path"] = checkpoint_dir
        plan["training_plan"] = {
            "path": FURNACE + "/smoke.yaml",
            "sha256": sha(ROOT / FURNACE / "smoke.yaml"),
        }
        for key, name in (
            ("artifact_config", "artifact-metadata/config.json"),
            ("artifact_manifest", "artifact-metadata/manifest.json"),
            ("gate3_report", GATE_SUBDIR + "/gate3-smolvla-sim-471.json"),
        ):
            relative = GATES["005"] + "/" + name
            plan[key] = {"path": relative, "sha256": sha(ROOT / relative)}
        write(repro / (role + ".json"), plan)
        if plan["artifact_config"]["sha256"] != historical["artifact_config"]["sha256"]:
            binding_gaps.append("artifact_config differs from pinned authority")
        artifact = load(plan["artifact_manifest"]["path"])
        if artifact.get("files", {}).get("config.json") != artifact_config_digest(
            ROOT, plan["artifact_config"]
        ):
            binding_gaps.append(
                "historical artifact manifest config member differs from retained config"
            )
        gate = load(plan["gate3_report"]["path"])
        if gate.get("artifact_manifest_sha256") != plan["artifact_manifest"]["sha256"]:
            binding_gaps.append("historical Gate3 does not bind retained artifact manifest")
        validate_plan(plan, ROOT, check_files=False)
        try:
            validate_plan(plan, ROOT)
        except ValueError as exc:
            if "Draft is not executable" not in str(exc):
                raise
        else:
            raise ValueError("Draft unexpectedly executable")
        if role == "baseline_a":
            # Read-only file-identity preflight. This transient copy is never saved
            # or handed to authorize/collection and has no execution permission.
            check = dict(plan, status="registered")
            identity_check = validate_plan(check, ROOT)
    write(
        repro / "readiness.json",
        {
            "execution_ready": False,
            "seed": 1002,
            "local_sources_bound": True,
            "full_runtime_source_closure_verified": False,
            "unbound": [
                "current remote library/GPU fingerprint",
                "authorization window",
                "watchdog",
                "complete runtime source closure",
            ],
            "local_binding_gaps": sorted(set(binding_gaps)),
            "identity_preflight": identity_check,
            "all_draft_roles_rejected_execution": True,
            "config_serialization_relation": "exact_fixed_transport_bytes_plus_final_LF",
            "scope": "New three-arm comparison; old 004 first divergence is not recoverable.",
        },
    )
    write(
        output / "evidence-inventory.json",
        {
            "schema_version": 1,
            "files": records,
            "historical_source_manifest": FURNACE + "/registration.json",
            "current_sources": current_sources,
            "mismatches": [r["path"] for r in records if r["status"] == "hash_mismatch"],
            "missing": [r["path"] for r in records if r["status"] == "missing"],
            "execution_repeated": False,
            "model_loaded": False,
        },
    )
    write(
        output / "evidence-levels.json",
        {
            "file_verified": "Matches recorded hash; not scientific replication.",
            "file_hashed_no_prior_seal": "Current hash only; prior transfer not reverified.",
            "report_claim": "Dated interpretation; compare with original metrics.",
            "historical_execution": "Retained records only; no model or rollout repeated.",
            "missing": "Unavailable locally; not filled with an inferred success.",
        },
    )
    schedule = load(FURNACE + "/schedule.json")["sample_identities"]
    ingress = load(FURNACE + "/model-ingress.json")
    ledger = load(FURNACE + "/observation/result.json")["observation"]
    actual = [[r["episode"], r["frame"]] for r in ingress["records"]]
    if (
        actual != schedule
        or len(set(map(tuple, actual))) != 20000
        or ledger["completed_updates"] != 5000
    ):
        raise ValueError("Retained native sampler/update evidence disagrees")
    gate_facts = {}
    for attempt, prefix in GATES.items():
        gate = load(prefix + "/" + GATE_SUBDIR + "/gate4-smolvla-sim-471.json")
        gate_facts[attempt] = {
            "status": gate["status"],
            "aggregate": gate["aggregate"],
            "acceptance_criteria": gate.get("acceptance_criteria"),
            "evidence_level": "retained_original_run_not_reexecuted",
        }
    write(
        output / "execution-facts.json",
        {
            "training": {
                "native_ingress_matches_schedule": True,
                "completed_updates": 5000,
                "unique_inputs": 20000,
                "exposures": len(actual),
                "evidence_level": "retained_original_run_not_reexecuted",
            },
            "gates": gate_facts,
        },
    )
    seal(output)
    if any(r["status"] == "hash_mismatch" for r in records):
        raise ValueError("Historical content mismatch; retained inventory identifies members")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
