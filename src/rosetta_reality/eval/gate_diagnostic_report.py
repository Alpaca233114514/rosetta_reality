"""Human/machine diagnostic report, retaining uncertainty and missing physics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .gate_diagnostic_io import load_json, seal, sha, write_json
from .gate_diagnostic_replay import object_contacts
from .gate_diagnostic_training import finding
from .gate_diagnostic_verify import verify_bundle


def analyze(source, output, *, probe=None, training=None):
    source, output = Path(source), Path(output)
    checked = verify_bundle(source)
    if checked["stage"] != "collect":
        raise ValueError("Analyze requires collection evidence")
    source_result = load_json(source / "result.json")
    provenance = load_json(source / "identity.json")["identity"]["kind"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "identity.json",
        {
            "schema_version": 1,
            "stage": "analyze",
            "diagnostic_only": True,
            "source_manifest_sha256": sha(source / "manifest.json"),
        },
    )
    result = {
        "stage": "analyze",
        "status": "incomplete",
        "diagnostic_only": True,
        "optimizer_steps": 0,
        "findings": [],
        "error_type": None,
        "expert_deviation_measured": False,
        "formal_gate4_changed": False,
        "collection_status": source_result["status"],
        "task_success": source_result["task_success"],
        "provenance_kind": provenance,
    }
    try:
        rows = []
        trace = source / "trace/trace.jsonl"
        if trace.exists():
            rows = [
                r for r in map(json.loads, trace.read_text().splitlines()) if r["event"] == "step"
            ]
        with (output / "timeline.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "step",
                    "simulation_seconds",
                    "reward",
                    "success",
                    "object_contacts",
                    "joint_limit_count",
                    "left_command",
                    "right_command",
                    "physics_available",
                ),
            )
            writer.writeheader()
            names = load_json(source / "identity.json")["dimension_names"]
            for row in rows:
                snap = row["physics_after"]
                contacts = object_contacts(row)
                limits = (snap.get("value") or {}).get("joint_limit_violations")
                writer.writerow(
                    {
                        "step": row["step"],
                        "simulation_seconds": (row["step"] + 1) / 50,
                        "reward": row["reward"],
                        "success": row["success"],
                        "object_contacts": None if contacts is None else len(contacts),
                        "joint_limit_count": len(limits)
                        if snap["available"] and limits is not None
                        else None,
                        "left_command": row["executed_action"][names.index("left_gripper")],
                        "right_command": row["executed_action"][names.index("right_gripper")],
                        "physics_available": snap["available"],
                    }
                )
        evidence = {
            "source_manifest_sha256": sha(source / "manifest.json"),
            "path": "trace/trace.jsonl",
        }
        known = [object_contacts(r) for r in rows]
        result["rollout_metrics"] = source_result.get("metrics")
        observed_events = {}
        previous_contacts, previous_reward = None, 0.0
        for row, contacts in zip(rows, known):
            candidates = {
                "first_object_contact": bool(contacts),
                "first_contact_loss": previous_contacts is not None
                and contacts is not None
                and bool(set(previous_contacts) - set(contacts)),
                "first_reward_change": row["reward"] != previous_reward,
            }
            for event, occurred in candidates.items():
                if occurred and event not in observed_events:
                    observed_events[event] = row["step"]
                    result["findings"].append(
                        finding(
                            event,
                            "model_behavior",
                            {**evidence, "step": row["step"]},
                            "First observed event; not an expert-error or causal label.",
                        )
                    )
            previous_contacts, previous_reward = contacts, row["reward"]
        result["observed_events"] = observed_events
        if rows and all(c is not None for c in known) and not any(known):
            result["findings"].append(
                finding(
                    "contact",
                    "model_behavior",
                    evidence,
                    "No robot-object contact in the recorded steps.",
                )
            )
        if any(c is None for c in known) or not rows:
            result["findings"].append(
                finding(
                    "physics",
                    "insufficient_evidence",
                    evidence,
                    "Physical contact measurements are unavailable for some steps.",
                )
            )
        result["findings"].append(
            finding(
                "expert",
                "insufficient_evidence",
                evidence,
                "No matched Seed 3 expert; no first-error label is assigned.",
            )
        )
        if probe is not None:
            verify_bundle(probe, source=source)
            probe_result = load_json(Path(probe) / "result.json")
            if probe_result["stage"] not in ("probe", "replay"):
                raise ValueError("Expected replay or probe package")
            result["probe_status"] = probe_result["status"]
            result["probe_records"] = probe_result["records"]
            result["findings"].append(
                finding(
                    "model_dependency",
                    "model_behavior"
                    if probe_result["status"] == "complete"
                    else "insufficient_evidence",
                    {"manifest_sha256": sha(Path(probe) / "manifest.json")},
                    "Input interventions measure local dependence, not correct actions."
                    if probe_result["status"] == "complete"
                    else "Replay/control failed or incomplete; interventions are inconclusive.",
                )
            )
        if training is not None:
            verify_bundle(training)
            audit = load_json(Path(training) / "result.json")
            if audit["stage"] != "audit-training":
                raise ValueError("Expected training audit package")
            from .gate_diagnostic_protocol import WEIGHT_SHA

            audit_identity = load_json(Path(training) / "identity.json")
            if audit_identity.get("checkpoint_sha256") != WEIGHT_SHA:
                raise ValueError("Training audit belongs to a different endpoint")
            result["findings"].extend(audit["findings"])
        lines = [
            "# Seed 3 诊断报告",
            "",
            "仅作诊断；不改变正式 G4 或 M2 状态。",
            "",
            f"证据类型：{provenance}（synthetic 为框架自测，不是模型能力证据）。",
            "",
            f"采集完整性：{source_result['status']}；任务成功：{source_result['task_success']}。",
            "",
            "命令动作、实际位姿和未执行的未来 chunk 分开解释；空白物理字段表示未测。",
            "",
        ]
        for item in result["findings"]:
            lines.extend(
                [
                    f"## {item['check']} — {item['category']}",
                    "",
                    item["phenomenon"],
                    "",
                    f"证据：`{json.dumps(item['evidence'], ensure_ascii=False)}`",
                    "",
                    f"代码：`{item['code']}`",
                    "",
                    f"对照：`{json.dumps(item['control'], ensure_ascii=False)}`",
                    "",
                    item["boundary"],
                    "",
                    "下一项最小验证：" + item["next_minimal_check"],
                    "",
                ]
            )
        with (output / "report.md").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines))
        result["status"] = "complete"
        return result
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        raise
    finally:
        seal(output, result)
