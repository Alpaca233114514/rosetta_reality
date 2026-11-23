"""Convert independently audited paired evidence to sealed training inventories."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from audit_paired_collection_v2 import manifest_inventory, verified_bytes


def export(root, audit_path, camera_map):
    root = root.resolve()
    summary_bytes = (root / "summary.json").read_bytes()
    manifest_bytes = (root / "manifest.json").read_bytes()
    summary = json.loads(summary_bytes)
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if (audit.get("passed") is not True
            or audit["summary_sha256"] != hashlib.sha256(summary_bytes).hexdigest()
            or audit["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
            or audit.get("admitted_N") != summary["N"]
            or audit.get("matched_N") != summary["candidate_matched_N"]
            or audit.get("registered_attempts") != summary["registered_attempts"]):
        raise ValueError("independent audit absent or not bound")
    inventory = manifest_inventory(json.loads(manifest_bytes))
    # Even N=0 exports refer to the full attempt evidence. Verify the original
    # audit-bound inventory before copying any independently-verified claim.
    for relative in inventory:
        verified_bytes(root, relative, inventory)
    if verified_bytes(root, "summary.json", inventory) != summary_bytes:
        raise ValueError("summary changed during export")
    if not isinstance(camera_map, dict):
        raise ValueError("camera map must be explicit object")
    attempts = {a["attempt_id"]: a for a in summary["attempts"]}
    pairs, runtime, inputs = [], [], {"S": [], "R": []}
    exploratory = summary.get("research_scope") == "fresh_lineage_fixed_development_exploration_v1"
    if exploratory and (audit.get("research_scope") != summary["research_scope"]
            or audit.get("N_strict") != 0 or summary["N_strict"] != 0
            or audit.get("N_exploratory") != summary["N_exploratory"]
            or audit.get("fresh_lineage_verified") != summary["fresh_lineage_verified"]
            or audit.get("new_scene_disjoint_verified") != summary["new_scene_disjoint_verified"]
            or audit.get("legacy_object_scene_nonoverlap") != "not_established"):
        raise ValueError("finite exploration source boundary has no independent audit")
    admission = summary["phase"] == "Q2" and summary["N"] > 0 and (
        summary["fresh_lineage_verified"] and summary["new_scene_disjoint_verified"]
        and summary["legacy_object_scene_nonoverlap"] == "not_established"
        if exploratory else summary["protected_scene_inventory_complete"])
    for index, pair in enumerate(summary["pairs"]):
        if not pair["matched"] or not admission:
            continue
        exported = {"parent_id": pair["pair_id"], "scene_disjoint_verified": not exploratory,
                    "N_strict": 0 if exploratory else 1,
                    "research_scope": summary.get("research_scope", "strict_legacy_scene_disjoint"),
                    "fresh_lineage_verified": exploratory,
                    "new_scene_disjoint_verified": exploratory,
                    "legacy_object_scene_nonoverlap": summary.get("legacy_object_scene_nonoverlap", "strict_inventory_required")}
        sampler = None
        sampler_relative = pair["pair_id"]+"/pair-sampler.json"
        if summary.get("sampling_rule") == "symmetric_common_legal_chunk_grid_v1":
            sampler = json.loads(verified_bytes(root, sampler_relative, inventory))
            if sampler["sufficient_common_starts"] is not True or sampler["pre_deviation_prefix_exact"] is not True:
                raise ValueError("independently audited common grid is not sufficient")
        for role in ("S", "R"):
            a = attempts[pair[role]]
            relative = f"{pair['pair_id']}/{role}/labels.json"
            labels_bytes = verified_bytes(root, relative, inventory)
            labels = json.loads(labels_bytes)
            for row in labels["rows"]:
                verified_bytes(root, row["images"], inventory)
                if row["images_sha256"] != inventory[row["images"]]["sha256"]:
                    raise ValueError("label image identity differs from audited manifest")
            selected = sampler["common_valid_chunk_starts"] if sampler else labels["valid_chunk_starts"]
            exported[role] = {"effective_frames": len(selected),
                              "valid_target_slots": len(selected)*50,
                              "raw_effective_frames": a["effective_input_frames"],
                              "raw_valid_target_slots": a["valid_target_slots"],
                              "task_success": a["task_success"], "labels_independently_verified": True,
                              "recovery_chain_verified": a["verified_recovery_chain"],
                              "ordinary_success_verified": role == "S" and a["deviation"] is None,
                              "attempt_id": a["attempt_id"], "scene_fingerprint": a["scene_fingerprint"]}
            for frame in selected:
                if frame not in labels.get("native_valid_chunk_starts", labels["valid_chunk_starts"]):
                    raise ValueError("selected grid crosses nonnative/incorrect target")
                inputs[role].append({"episode": index, "frame": frame,
                                     "positive_supervision": True,
                                     "chunk_crosses_unqualified_action": False})
                runtime.append({"source": role, "episode": index, "frame": frame,
                                "labels": relative, "labels_sha256": inventory[relative]["sha256"],
                                **({"pair_sampler": sampler_relative,
                                    "pair_sampler_sha256": inventory[sampler_relative]["sha256"],
                                    "common_grid_sha256": sampler["common_starts_sha256"]} if sampler else {})})
        pairs.append(exported)
    assert len(pairs) == summary["N"]
    return {"schema": "rosetta.paired_training_cohort.v2", "N": len(pairs), "pairs": pairs,
            "N_strict": 0 if exploratory else len(pairs),
            "N_exploratory": len(pairs) if exploratory else 0,
            "research_scope": summary.get("research_scope", "strict_legacy_scene_disjoint"),
            "fresh_lineage_verified": summary.get("fresh_lineage_verified", False),
            "new_scene_disjoint_verified": summary.get("new_scene_disjoint_verified", False),
            "legacy_object_scene_nonoverlap": summary.get("legacy_object_scene_nonoverlap", "strict_inventory_required"),
            "S_inputs": inputs["S"], "R_inputs": inputs["R"], "runtime_inputs": runtime,
            "camera_map": camera_map, "provider_qualified": False,
            "provider_status": summary["provider_status"],
            "registered_attempts": summary["registered_attempts"],
            "all_attempts": summary["attempts"], "all_pair_exclusions": summary["pairs"],
            "candidate_matched_N": summary["candidate_matched_N"],
            "source_summary_sha256": audit["summary_sha256"],
            "source_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "status": "sealed" if pairs else "N_zero_no_eligible_training_inputs",
            "normalization_source": "D_train_only", "native_loss_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--camera-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.root, args.audit, json.loads(args.camera_map.read_text()))
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
