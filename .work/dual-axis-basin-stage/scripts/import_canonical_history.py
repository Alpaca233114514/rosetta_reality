#!/usr/bin/env python3
"""Create-only, read-only-source import of selected canonical Rosetta evidence."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from basin.adapters import import_native
from basin.api import BasinAPI
from basin.io import child, dumps, loads, read_bytes, sha, write_new
from basin.store import Store


KINDS = ("checkpoint_2500", "checkpoint_5000", "historical_training_evidence",
         "historical_gate_004", "historical_gate_005", "saved_data", "saved_data_manifest")
TRAIN = "canonical-furnace-received-20260914-002/runs/canonical-fullframes-20260914-001"
GATES = {"004": "canonical-posttrain-received-20260915-004/verified",
         "005": "canonical-posttrain-received-20260916-005-ssh/verified"}
GATE_REPORT = "results/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/gates/gate4-smolvla-sim-471.json"
MODEL_5000 = "canonical-checkpoints-received-20260914-002/step-005000/verified/pretrained_model/model.safetensors"
CODE_PATHS = (
    "scripts/canonical_furnace.py", "scripts/run_canonical_furnace.py",
    "scripts/sim_gate.py", "src/rosetta_reality/vla/processor.py",
    "src/rosetta_reality/vla/training/temporal_sampler.py",
    "configs/data/aloha_sim_insertion_m2.yaml",
    "configs/sim/aloha_insertion_smolvla.yaml",
)


def original(root, relative):
    return child(root, relative)


def file_sha(path):
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(block)
            if count > 4 * 1024 ** 3:
                raise ValueError(f"Source exceeds 4 GiB: {path.name}")
            digest.update(block)
    return digest.hexdigest(), count


def native(store, output, run_id, source_run, identity, source_paths):
    value = {"schema_version": 1, "evidence_kind": "historical_observation",
             "source_run": source_run, "status": "recorded",
             "parameters": {"identity": identity, "source_paths": source_paths},
             "dimensions": [], "events": []}
    path = output / "native-inputs" / (run_id + ".json")
    write_new(path, dumps(value))
    return import_native(store, path, run_id)


def check(api, run_id, pointer, source, path, category, checks, *, claim_source=None):
    result = api.call("basin_check_identity", {"run_id": run_id, "pointer": pointer,
        "source": source, "path": path})["data"]
    checks.append({"category": category, "run_id": run_id, "pointer": pointer,
                   "source": source, "path": path, "claim": result["claim"],
                   "claim_source": claim_source or "Basin identity pointer",
                   "actual_sha256": result["checks"]["source_file"]["sha256"],
                   "bytes": result["checks"]["source_file"]["bytes"],
                   "status": result["status"],
                   "verification": "prior_claim_vs_source_bytes" if result["claim"] is not None
                   else "computed_at_import"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosetta", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rosetta, runs, output = (Path(value).resolve() for value in
                             (args.rosetta, args.runs, args.output))
    if not rosetta.is_dir() or not runs.is_dir() or output.exists():
        raise ValueError("Sources must exist and output must be a new path")
    if output.is_relative_to(rosetta) or output.is_relative_to(runs):
        raise ValueError("Output must be outside read-only source roots")

    inventory_path = original(runs, "root-evidence-20260922-002/evidence-inventory.json")
    facts_path = original(runs, "root-evidence-20260922-002/execution-facts.json")
    inventory = loads(read_bytes(inventory_path))
    facts = loads(read_bytes(facts_path))
    if inventory.get("schema_version") != 1 or not isinstance(inventory.get("files"), list):
        raise ValueError("Unsupported inventory")
    selected = [row for row in inventory["files"] if row.get("kind") in KINDS]
    inventory_claims = {row["path"].removeprefix("runs/"): row.get("expected_sha256")
                        for row in selected}
    if len(selected) != len({row["path"] for row in selected}):
        raise ValueError("Duplicate selected inventory path")
    output.mkdir(parents=True, exist_ok=False)
    store = Store(output / "history")
    api = BasinAPI(store.root, {"runs": runs, "rosetta": rosetta})
    checks = []
    originals = {"evidence-inventory.json": read_bytes(inventory_path),
                 "execution-facts.json": read_bytes(facts_path)}
    original_sources = {"evidence-inventory.json": "root-evidence-20260922-002/evidence-inventory.json",
                        "execution-facts.json": "root-evidence-20260922-002/execution-facts.json"}

    def retain(label, path):
        data = read_bytes(original(runs, path))
        originals[label] = data
        original_sources[label] = path
        return loads(data) if label.endswith(".json") else None

    training = retain("train-registration.json", TRAIN + "/registration.json")
    retain("train-checkpoint.json", TRAIN + "/checkpoint.json")
    retain("train-model-ingress.json", TRAIN + "/model-ingress.json")
    retain("train-exit.json", TRAIN + "/train-exit.json")
    source_claims = {path: training["sources"].get(path) for path in CODE_PATHS}
    native(store, output, "canonical-training-001", training["id"],
           {"registration_id": training["id"], "source_files": source_claims,
            "completed_updates": facts["training"]["completed_updates"]},
           [TRAIN + "/registration.json", TRAIN + "/checkpoint.json"])
    for path in CODE_PATHS:
        # Current checkout drift is reported separately from historical source integrity.
        check(api, "canonical-training-001", "/parameters/identity/source_files/" +
              path.replace("~", "~0").replace("/", "~1"), "rosetta", path,
              "historical_source_vs_current_checkout", checks)

    for step, kind in ((2500, "checkpoint_2500"), (5000, "checkpoint_5000")):
        prefix = f"canonical-checkpoints-received-20260914-002/step-{step:06d}/verified/"
        rows = [row for row in selected if row["kind"] == kind]
        files = {row["path"].removeprefix("runs/" + prefix): row.get("expected_sha256") for row in rows}
        run_id = f"canonical-step-{step}"
        native(store, output, run_id, training["id"],
               {"checkpoint_step": step, "files": files}, ["runs/" + prefix])
        for name in files:
            relative = prefix + name
            check(api, run_id, "/parameters/identity/files/" +
                  name.replace("~", "~0").replace("/", "~1"), "runs", relative,
                  kind, checks)

    for gate in ("004", "005"):
        prefix = GATES[gate]
        registration = retain(f"gate-{gate}-registration.json", prefix + "/registration.json")
        report = retain(f"gate-{gate}-report.json", prefix + "/" + GATE_REPORT)
        manifest = retain(f"gate-{gate}-artifact-manifest.json", prefix + "/artifact-metadata/manifest.json")
        seal = retain(f"gate-{gate}-seal.json", prefix + "/gate-seal.json")
        handoff = retain(f"gate-{gate}-handoff-manifest.json", prefix + "/handoff-manifest.json")
        receipt = retain(f"gate-{gate}-transfer-receipt.json", prefix + "/transfer-receipt.json")
        gate3_path = GATE_REPORT.replace("gate4-", "gate3-")
        retain(f"gate-{gate}-gate3-report.json", prefix + "/" + gate3_path)
        if registration["id"] != f"canonical-fullframes-posttrain-202609{'15' if gate == '004' else '16'}-{gate}":
            raise ValueError("Gate registration run identity mismatch")
        if report["artifact_id"] != manifest["artifact_id"]:
            raise ValueError("Gate artifact ID mismatch")
        if handoff["run"] != registration["id"] or receipt["run"] != registration["id"]:
            raise ValueError("Gate handoff run identity mismatch")
        if receipt["all_file_sha256_matched"] is not True:
            raise ValueError("Handoff receipt did not report file hash verification")
        report_claim = handoff["files"][GATE_REPORT]
        if report_claim["bytes"] != len(originals[f"gate-{gate}-report.json"]):
            raise ValueError("Gate report byte count differs from handoff manifest")
        run_id = f"canonical-gate-{gate}"
        identity = {"registration_id": registration["id"], "artifact_id": report["artifact_id"],
                    "selected_checkpoint_model_sha256": manifest["selected_checkpoint_model_sha256"],
                    "report_sha256": report_claim["sha256"],
                    "report_claim_source": "handoff-manifest.json/files/" + GATE_REPORT,
                    "handoff_manifest_sha256": receipt["manifest_sha256"],
                    "artifact_manifest_sha256": report["artifact_manifest_sha256"],
                    "simulation_plan_sha256": report["simulation_plan_sha256"],
                    "gate3_report_sha256": report["gate3_report_sha256"],
                    "code_identity": report["code_identity"]}
        native(store, output, run_id, registration["id"], identity,
               [prefix + "/" + GATE_REPORT, prefix + "/artifact-metadata/manifest.json"])
        check(api, run_id, "/parameters/identity/selected_checkpoint_model_sha256",
              "runs", MODEL_5000, "gate_model_bytes", checks)
        check(api, run_id, "/parameters/identity/report_sha256",
              "runs", prefix + "/" + GATE_REPORT, "gate_report_bytes", checks,
              claim_source=prefix + "/handoff-manifest.json:files/" + GATE_REPORT)
        check(api, run_id, "/parameters/identity/handoff_manifest_sha256",
              "runs", prefix + "/handoff-manifest.json", "gate_handoff_manifest_bytes", checks,
              claim_source=prefix + "/transfer-receipt.json:manifest_sha256")
        check(api, run_id, "/parameters/identity/artifact_manifest_sha256",
              "runs", prefix + "/artifact-metadata/manifest.json", "gate_artifact_manifest_bytes", checks)
        check(api, run_id, "/parameters/identity/gate3_report_sha256",
              "runs", prefix + "/" + gate3_path, "gate3_report_bytes", checks)
        checks.append({"category": "gate_plan_claims", "run_id": run_id,
                       "source": "runs", "path": prefix + "/gate-seal.json",
                       "claim": report["simulation_plan_sha256"],
                       "other_claim": seal["plan_sha256"],
                       "status": "match" if report["simulation_plan_sha256"] == seal["plan_sha256"] else "mismatch",
                       "verification": "two_recorded_claims_compared"})

    data_rows = [row for row in selected if row["kind"] in ("saved_data", "saved_data_manifest")]
    data_files = {row["path"].removeprefix("runs/"): row.get("expected_sha256") for row in data_rows}
    native(store, output, "canonical-saved-data", training["id"], {"files": data_files},
           list(data_files))
    for row in data_rows:
        path = row["path"].removeprefix("runs/")
        check(api, "canonical-saved-data", "/parameters/identity/files/" +
              path.replace("~", "~0").replace("/", "~1"), "runs", path,
              "saved_data_manifest_unsealed" if row["kind"] == "saved_data_manifest" else
              "saved_data_bytes", checks)

    # Keep original report bytes in Basin history, separate from generated native envelopes.
    retained_claims = {name: inventory_claims.get(path) for name, path in original_sources.items()}
    store.add({"schema_version": 1, "id": "canonical-original-reports",
               "adapter": "basin.raw_source_bundle.v1", "source_run": training["id"],
               "status": "retained", "evidence_kind": "historical_observation",
               "gate": {}, "parameters": {"source_paths": original_sources,
                                            "identity": {"files": retained_claims}},
               "dimensions": [], "verification": {"source_hashes": "computed_at_import"},
               "limitations": ["Raw bytes retained; source authorship and measurements not certified."]},
              [], originals)
    for name, path in original_sources.items():
        check(api, "canonical-original-reports", "/parameters/identity/files/" +
              name.replace("~", "~0").replace("/", "~1"), "runs", path,
              "source_report_bytes" if retained_claims[name] else "source_report_unsealed", checks)
    report = {"schema_version": 1, "source_run": training["id"],
              "units": [row["id"] for row in store.history()],
              "source_report_sha256": {key: sha(value) for key, value in originals.items()},
              "checks": checks, "counts": dict(Counter(row["status"] for row in checks)),
              "limitations": ["Historical run not re-executed.",
                              "Current checkout mismatches are drift, not proof the historical run was invalid.",
                              "Two saved-data manifests have no prior SHA seal in the inventory; their observed byte hashes are retained as unsealed evidence.",
                              "Unsealed source reports retain original bytes and computed hashes without a claimed prior SHA.",
                              "Hash matches do not independently establish training or Gate success."]}
    write_new(output / "identity-report.json", dumps(report))
    print(dumps({"output": str(output), "units": len(report["units"]),
                 "counts": report["counts"], "report_sha256": sha(read_bytes(output / "identity-report.json"))}))
    return 0 if not any(row["status"] != "match" and row["category"] not in
                        ("historical_source_vs_current_checkout", "saved_data_manifest_unsealed",
                         "source_report_unsealed")
                        for row in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
