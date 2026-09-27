"""Create a Basin history from the sealed pre/post Gate reports, without models.

This is an explicit derived import. The original reports remain authoritative;
Basin retains their complete JSON and exact source-file SHA256 values.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

RUN_ID = "prepost-gate4-20260924-001"
EXPERIMENT = "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
EXPECTED = {
    "worker-exited.json": "3eef079663e68229cdcc24c29ea1d23319693a53faf172b4521e31a5e5995945",
    f"results/{EXPERIMENT}/gates/gate3-smolvla-sim-481.json":
        "469a441e2d0ee72f7c63857be43b08ecab0c81795cadce39790991d9a950d19c",
    f"results/{EXPERIMENT}/gates/gate3-smolvla-sim-482.json":
        "5222f7de5ef3ad6e0fcff6b9015ac239c878aad9a7ef89f2d529bbcF53a32c78".lower(),
    f"results/{EXPERIMENT}/gates/gate4-smolvla-sim-482.json":
        "4db5f64c0b375d77e3f5965e1afd292d417c427466c5cd2a01f97c796abc8503",
}
MODEL_SHA = {
    "base": "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb",
    "trained": "d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef",
}
SUFFIX = {"base": "481", "trained": "482"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def report_path(source: Path, arm: str, gate: str) -> Path:
    return source / "results" / EXPERIMENT / "gates" / f"{gate}-smolvla-sim-{SUFFIX[arm]}.json"


def check_source(source: Path, basin_workspace: Path) -> dict[str, dict]:
    for name, expected in EXPECTED.items():
        if digest(source / name) != expected:
            raise ValueError(f"Native source identity changed: {name}")
    worker = load(source / "worker-exited.json")
    if worker != {
        "elapsed_seconds": worker.get("elapsed_seconds"), "error": None,
        "m2_complete": False, "optimizer_steps": 0,
        "outcomes": {"base_gate3": "failed", "base_gate4": "not measured: Gate 3 failed",
                     "trained_gate3": "passed", "trained_gate4": "failed"},
    }:
        raise ValueError("Worker outcome differs from registered comparison")
    if (source / "results" / EXPERIMENT / "gates/gate4-smolvla-sim-481.json").exists():
        raise ValueError("Base Gate 4 must remain unmeasured")
    reports = {}
    for arm, gate in (("base", "gate3"), ("trained", "gate3"), ("trained", "gate4")):
        path = report_path(source, arm, gate)
        report = load(path)
        plan = source / f"{arm}-gate.yaml"
        manifest = source / "artifacts" / EXPERIMENT / f"prepost-{arm}-aloha-interface-001/manifest.json"
        if (report.get("schema_version") != 1
                or report.get("experiment_id") != EXPERIMENT
                or report.get("hidden_test_loaded") is not False
                or report.get("artifact_id") != f"prepost-{arm}-aloha-interface-001"
                or report.get("simulation_plan_sha256") != digest(plan)
                or report.get("artifact_manifest_sha256") != digest(manifest)):
            raise ValueError(f"{arm} {gate} report binding failed")
        if gate == "gate4" and report.get("gate3_report_sha256") != digest(report_path(source, arm, "gate3")):
            raise ValueError("Gate 4 does not bind the matching passed Gate 3")
        if report.get("status") != worker["outcomes"][f"{arm}_{gate}"]:
            raise ValueError("Worker and Gate report status differ")
        reports[f"{arm}-{gate}"] = report
    payload = load(basin_workspace / "tools/payload-manifest.json")
    checked = 0
    for item in payload["source_files"]:
        if item["path"].startswith("tools/basin/basin/"):
            path = basin_workspace / item["path"]
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError(f"Basin source changed: {item['path']}")
            checked += 1
    if checked < 5:
        raise ValueError("Basin source inventory is incomplete")
    return reports


def envelope(source: Path, arm: str, gate: str, report: dict) -> dict:
    path = report_path(source, arm, gate)
    if gate == "gate3":
        metrics = report["metrics"]
        events = [{"step": 0, "stage": "gate3", "values": {
            "status": report["status"],
            "joint_limit_violations": metrics["joint_limit_violations"],
            "unexpected_collisions": metrics["unexpected_collisions"],
            "maximum_reward": metrics["maximum_reward"],
            "rollout_length": metrics["rollout_length"],
            "success": metrics["success"],
        }}]
    else:
        events = [{"step": index, "stage": "gate4_episode", "values": {
            "seed": item["seed"], "policy_noise_seed": item["policy_noise_seed"],
            "success": item["success"], "maximum_reward": item["maximum_reward"],
            "rollout_length": item["rollout_length"],
            "joint_limit_violations": item["joint_limit_violations"],
            "unexpected_collisions": item["unexpected_collisions"],
        }} for index, item in enumerate(report["episodes"])]
        if [event["values"]["seed"] for event in events] != [1000, 1001, 1002, 1003, 1004]:
            raise ValueError("Gate 4 seed identity changed")
    return {
        "schema_version": 1, "evidence_kind": "historical_observation",
        "source_run": RUN_ID, "status": report["status"],
        "gate": {"arm": arm, "stage": gate, "status": report["status"],
                 "acceptance_criteria": report["acceptance_criteria"]},
        "parameters": {
            "model_safetensors_sha256": MODEL_SHA[arm],
            "source_report_path": path.relative_to(source).as_posix(),
            "source_report_sha256": digest(path),
            "source_report": report,
            "interpretation": "Pinned base under saved ALOHA interface is not historical step zero",
        },
        "dimensions": [], "events": events,
    }


def deploy(source: Path, basin_workspace: Path, output: Path) -> None:
    reports = check_source(source, basin_workspace)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Create-only Basin deployment exists: {output}")
    output.mkdir(parents=True)
    write_new(output / "started.json", {
        "status": "started", "source_run": RUN_ID,
        "source_report_sha256": EXPECTED,
        "basin_workspace_payload_manifest_sha256": digest(basin_workspace / "tools/payload-manifest.json"),
    })
    inputs = output / "inputs"
    inputs.mkdir()
    specs = [("base-gate3", "base", "gate3"),
             ("trained-gate3", "trained", "gate3"),
             ("trained-gate4", "trained", "gate4")]
    for name, arm, stage in specs:
        write_new(inputs / f"{name}.json", envelope(source, arm, stage, reports[name]))
    worker = load(source / "worker-exited.json")
    write_new(inputs / "base-gate4-not-measured.json", {
        "schema_version": 1, "evidence_kind": "historical_observation",
        "source_run": RUN_ID, "status": "not_measured",
        "gate": {"arm": "base", "stage": "gate4", "status": "not_measured",
                 "reason": "Gate 3 failed"},
        "parameters": {"model_safetensors_sha256": MODEL_SHA["base"],
                       "source_report_path": "worker-exited.json",
                       "source_report_sha256": digest(source / "worker-exited.json"),
                       "source_report": worker},
        "dimensions": [], "events": [],
    })
    basin_root = basin_workspace / "tools/basin"
    history = output / "history"
    env = dict(os.environ, PYTHONPATH=str(basin_root))
    imported = []
    for name in ("base-gate3", "trained-gate3", "trained-gate4", "base-gate4-not-measured"):
        run_id = f"{RUN_ID}-{name}"
        result = subprocess.run([sys.executable, "-m", "basin", "--store", str(history),
                                 "import-native", str(inputs / f"{name}.json"), "--id", run_id],
                                env=env, capture_output=True, text=True, timeout=30, check=True)
        value = json.loads(result.stdout)
        if value != {"id": run_id, "status": "imported"}:
            raise ValueError(f"Basin import did not create a new run: {run_id}")
        imported.append(run_id)
    sys.path.insert(0, str(basin_root))
    from basin.api import BasinAPI

    api = BasinAPI(history)
    names = [tool["name"] for tool in api.tools()]
    if len(names) != 8 or any(name.startswith("basin_import") for name in names):
        raise ValueError("Read-only model-callable Basin tool set changed")
    listing = api.call("basin_history", {"source_run": RUN_ID})
    if listing["data"]["total"] != 4 or any(item["integrity"] != "verified" for item in listing["data"]["items"]):
        raise ValueError("Basin history identity/integrity failed")
    checks = {run_id: api.call("basin_verify", {"run_id": run_id})["data"] for run_id in imported}
    comparison = api.call("basin_compare", {
        "left": imported[0], "right": imported[1], "field": "/joint_limit_violations"})["data"]
    if comparison["compared_events"] != 1 or not comparison["comparable"]:
        raise ValueError("Basin Gate 3 comparison smoke failed")
    write_new(output / "verification.json", {
        "status": "verified", "source_run": RUN_ID, "runs": imported,
        "source_report_sha256": EXPECTED, "basin_source_files_checked": True,
        "history": listing["data"], "checks": checks,
        "gate3_comparison": comparison,
        "limitations": ["No base Gate 4 rollout exists because base Gate 3 failed.",
                        "Basin imports reported outcomes; it does not certify Gate acceptance.",
                        "No TorchLens or live model capture is included."],
    })
    launcher = output / "basin-mcp.sh"
    with launcher.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("#!/usr/bin/env bash\nset -Eeuo pipefail\n")
        stream.write(f"export PYTHONPATH='{basin_root}'\n")
        stream.write(f"exec '{sys.executable}' -m basin --store '{history}' mcp\n")
    launcher.chmod(0o755)
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "prepost-deploy-check", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "basin_history", "arguments": {"source_run": RUN_ID}}},
    ]
    result = subprocess.run([str(launcher)], input="\n".join(json.dumps(row) for row in requests) + "\n",
                            capture_output=True, text=True, timeout=20, check=True)
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    if ([row["id"] for row in responses] != [1, 2, 3]
            or len(responses[1]["result"]["tools"]) != 8
            or responses[2]["result"]["structuredContent"]["data"]["total"] != 4):
        raise ValueError("Basin MCP stdio handshake/query failed")
    write_new(output / "mcp-smoke.json", {"status": "passed", "responses": responses})


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--basin-workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deploy(args.source.resolve(strict=True), args.basin_workspace.resolve(strict=True), args.output)


if __name__ == "__main__":
    main()
