"""Independent saved-file paired audit: hashes, denominator, events and BC chunks."""
from __future__ import annotations
import argparse
import hashlib
import gzip
import importlib.util
import json
from pathlib import Path, PurePosixPath

_rules_spec = importlib.util.spec_from_file_location(
    "_paired_audit_rules", Path(__file__).resolve().parents[1] /
    "src/rosetta_reality/data/paired_collection_v2.py")
_rules = importlib.util.module_from_spec(_rules_spec)
_rules_spec.loader.exec_module(_rules)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or "\\" in relative or ":" in relative:
        raise ValueError("nonportable evidence path")
    target = root.joinpath(*part.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("evidence path escape")
    return target


def manifest_inventory(manifest):
    """Keep immutable audit-time payload identities, never bless a fresh rehash."""
    inventory = {}
    for entry in manifest["files"]:
        relative = entry["path"]
        if relative in inventory:
            raise ValueError("duplicate manifest payload")
        if type(entry.get("bytes")) is not int or entry["bytes"] < 0:
            raise ValueError("invalid manifest size")
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
            raise ValueError("invalid manifest digest")
        inventory[relative] = entry
    return inventory


def verified_bytes(root, relative, inventory):
    """Read once and compare these very bytes with the bound manifest entry."""
    if relative not in inventory:
        raise ValueError("payload absent from independently audited manifest")
    payload = safe(root, relative).read_bytes()
    expected = inventory[relative]
    if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
        raise ValueError("payload differs from independently audited identity")
    return payload


def receipt_json(root, relative, inventory):
    raw = verified_bytes(root, relative, inventory)
    return json.loads(gzip.decompress(raw) if relative.endswith(".gz") else raw)


def verify_recovery_window(rows, attempt, trigger, sustain_steps=5):
    """Require exactly five consecutive correction steps, never an empty slice."""
    if sustain_steps != 5 or trigger not in ("approach", "grasp", "align"):
        raise ValueError("unregistered recovery window")
    correction_start = attempt["correction_start"]
    end = attempt["recovery"]["step"]
    if type(correction_start) is not int or type(end) is not int:
        raise ValueError("recovery step indices must be integers")
    start = end - sustain_steps + 1
    if not 0 <= correction_start <= start <= end < len(rows):
        raise ValueError("recovery window outside correction segment")
    window = rows[start:end+1]
    if len(window) != sustain_steps or [r["step"] for r in window] != list(range(start, end+1)):
        raise ValueError("recovery window is not exactly five consecutive steps")
    if attempt["recovery"].get("sustain_steps") != sustain_steps:
        raise ValueError("recovery sustain count differs from registered event")
    for row in window:
        if row["action_source"] != "state_conditioned_teacher" or row["safe"] is not True:
            raise ValueError("recovery window contains nonexpert or unsafe step")
        geometry = row["after"]["geometry"]
        if geometry["socket_grasp_contact"] is not True or geometry["peg_grasp_contact"] is not True:
            raise ValueError("recovery window loses registered grasp contact")
        if trigger == "align" and geometry["peg_socket_contact"] is not True:
            raise ValueError("recovery window loses registered insertion contact")
    return window


def audit(root):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    inventory = manifest_inventory(manifest)
    plan = json.loads((root / "registration.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    local_protection_plan = dict(plan)
    if plan.get("protected_scene_inventory_complete") is True:
        original_ref = plan["protected_scene_inventory_ref"]
        local_protection_plan["protected_scene_inventory_ref"] = {
            name: {"path": "protected-scene-" + name + ".json", "sha256": original_ref[name]["sha256"]}
            for name in ("inventory", "audit")}
    protection = _rules.protected_inventory(root, local_protection_plan)
    assert summary["protected_scene_inventory_complete"] == protection["complete"]
    exploratory = plan.get("research_scope") == "fresh_lineage_fixed_development_exploration_v1"
    registry_path = root / "scene-registry"
    fresh = _rules.exploratory_registry(registry_path, registry_path/"audit.json", plan) if exploratory and registry_path.exists() else {
        "complete": False, "entries": {}, "forbidden": []}
    assert summary.get("fresh_lineage_verified", False) == fresh["complete"]
    if exploratory:
        assert summary["N_strict"] == 0 and summary["legacy_object_scene_nonoverlap"] == "not_established"
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(encoded).hexdigest() == manifest["plan_sha256"]
    paths = [row["path"] for row in manifest["files"]]
    assert len(paths) == len(set(paths))
    total = 0
    for row in manifest["files"]:
        path = safe(root, row["path"])
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]
        total += row["bytes"]
    assert total + (root / "manifest.json").stat().st_size <= plan["evidence_byte_cap"]
    expected = {pair["pair_id"] + "-" + kind for pair in plan["pairs"] for kind in ("S", "R")}
    attempts = summary["attempts"]
    assert len(attempts) == len(expected)
    assert {item["attempt_id"] for item in attempts} == expected
    records = {}
    event_checks = chunk_checks = 0
    restore_checks = 0
    restore_failures = 0
    restored_paths = {}
    for relative in paths:
        if not relative.endswith(("-restore.json", "-restore.json.gz")):
            continue
        restore = receipt_json(root, relative, inventory)
        equal = restore["replays"][0] == restore["replays"][1]
        assert restore["passed"] is equal
        replay_sha = [hashlib.sha256(json.dumps(r, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
                      for r in restore["replays"]]
        assert replay_sha == restore["exact_replays"]
        restore_checks += 1
        restore_failures += int(not equal)
        restored_paths[relative] = equal
    for attempt in attempts:
        path = root / attempt["pair_id"] / attempt["kind"] / "labels.json"
        if not path.is_file():
            assert attempt["rows"] == 0
            continue
        relative = path.relative_to(root).as_posix()
        labels = json.loads(verified_bytes(root, relative, inventory))
        rows = labels["rows"]
        if fresh["complete"]:
            registered_reset = fresh["entries"][attempt["seed"]]
            assert attempt["parent_id"] == registered_reset["parent_id"]
            assert attempt["scene_fingerprint"] == registered_reset["scene_fingerprint"]
            assert attempt["reset_state_sha256"] == registered_reset["reset_state_sha256"]
            assert attempt["model_sha256"] == registered_reset["model_sha256"]
            assert attempt["fresh_lineage_verified"] is True
        assert len(rows) == attempt["rows"] <= 500
        assert [r["step"] for r in rows] == list(range(len(rows)))
        success_steps = [r["step"] for r in rows if r["after"]["info"].get("is_success") is True]
        assert bool(success_steps) == attempt["task_success"]
        safety = any(not r["safe"] or r["after"]["joint_limits"] or r["after"]["unexpected_collisions"] for r in rows)
        assert not safety or attempt["safety_stop"]
        deliberate = [r for r in rows if r["action_source"] == "registered_deliberate_deviation"]
        assert len(deliberate) <= 1
        if attempt["kind"] == "S":
            assert not deliberate and attempt["deviation"] is None
        if attempt["verified_recovery_chain"]:
            assert len(deliberate) == 1 and attempt["kind"] == "R"
            dev = attempt["deviation"]
            assert dev["step"] == deliberate[0]["step"]
            names = {p["pair_id"]: p for p in plan["pairs"]}
            registered = names[attempt["pair_id"]]
            assert dev["delta_rad"] == registered["joint_delta_rad"]
            assert dev["trigger"] == deliberate[0]["phase"] == registered["trigger"]
            assert attempt["correction_start"] == dev["step"] + 1
            assert dev["verified"] and dev["observed_joint_change_rad"] >= plan["deviation_min_joint_change_rad"]
            end = attempt["recovery"]["step"]
            assert attempt["correction_start"] <= end < success_steps[-1]
            assert success_steps[-1] - attempt["correction_start"] < 150
            verify_recovery_window(rows, attempt, registered["trigger"], plan["recovery_sustain_steps"])
            event_checks += 1
        eligible = bool(attempt["task_success"] and not attempt["safety_stop"]
                        and (attempt["kind"] == "S" or attempt["verified_recovery_chain"]))
        for row in rows:
            assert row["positive_target"] == bool(eligible and row["safe"] and row["action_source"] == "state_conditioned_teacher")
            image_payload = verified_bytes(root, row["images"], inventory)
            assert hashlib.sha256(image_payload).hexdigest() == row["images_sha256"]
            assert row["pts_seconds"] == row["step"] / 50
            assert len(row["action"]) == len(row["state"]) == 14
        expected_starts = [i for i in range(max(0, len(rows)-49))
                           if all(r["positive_target"] is True and r["action_source"] == "state_conditioned_teacher"
                                  for r in rows[i:i+50])]
        assert labels["valid_chunk_starts"] == expected_starts
        assert labels.get("native_valid_chunk_starts", expected_starts) == expected_starts
        assert attempt["effective_input_frames"] == len(expected_starts)
        assert attempt["valid_target_slots"] == len(expected_starts)*50
        assert attempt["padding_slots"] == 0 and labels["native_loss_changed"] is False
        chunk_checks += len(expected_starts)
        records[attempt["attempt_id"]] = attempt
    matched = 0
    for pair in summary["pairs"]:
        sampler = None
        if (plan.get("sampling_rule") == "symmetric_common_legal_chunk_grid_v1"
                and pair.get("S") in records and pair.get("R") in records):
            relative = pair["pair_id"] + "/pair-sampler.json"
            sampler = json.loads(verified_bytes(root, relative, inventory))
            s_labels = json.loads(verified_bytes(root, pair["pair_id"]+"/S/labels.json", inventory))
            r_labels = json.loads(verified_bytes(root, pair["pair_id"]+"/R/labels.json", inventory))
            deviation = records[pair["R"]]["deviation"]
            expected_sampler = _rules.symmetric_sampler(s_labels["rows"], r_labels["rows"],
                deviation["step"] if deviation else None,
                minimum_common_starts=plan["minimum_common_chunk_starts"])
            assert sampler == expected_sampler
            for role in ("S", "R"):
                record = records[pair[role]]
                assert record["sampler_effective_input_frames"] == len(sampler["common_valid_chunk_starts"])
                assert record["sampler_valid_target_slots"] == len(sampler["common_valid_chunk_starts"])*50
                assert record["common_grid_sha256"] == sampler["common_starts_sha256"]
        if not pair["matched"]:
            continue
        a, b = records[pair["S"]], records[pair["R"]]
        for key in ("provider_id", "parent_id", "scene_fingerprint", "reset_state_sha256", "contract_sha256", "model_sha256"):
            assert a[key] == b[key]
        assert a["task_success"] and b["verified_recovery_chain"] and a["restore_exact"] and b["restore_exact"]
        suffix = ".json.gz" if plan.get("receipt_encoding") == "lossless_json_gzip_v1" else ".json"
        assert restored_paths.get(pair["pair_id"] + "/reset-restore" + suffix) is True
        assert restored_paths.get(pair["pair_id"] + "/loaded-restore" + suffix) is True
        assert not a["scene_overlap"] and not b["scene_overlap"]
        assert a["scene_fingerprint"] not in protection["fingerprints"]
        assert a["scene_fingerprint"] not in fresh["forbidden"]
        if sampler is not None:
            assert sampler["pre_deviation_prefix_exact"] and sampler["sufficient_common_starts"]
        else:
            for key in ("effective_input_frames", "valid_target_slots"):
                assert min(a[key], b[key]) > 0 and abs(a[key]-b[key])/max(a[key],b[key]) <= 0.05
        matched += 1
    assert matched == summary["candidate_matched_N"]
    if exploratory:
        expected_n = matched if plan["phase"] == "Q2" and fresh["complete"] else 0
        assert summary["N"] == summary["N_exploratory"] == expected_n
        assert summary["formal_training_eligible"] is False
        assert summary["exploratory_training_eligible"] == (expected_n > 0)
    elif plan["phase"] != "Q2" or not protection["complete"]:
        assert summary["N"] == 0 and summary["formal_training_eligible"] is False
    else:
        assert summary["N"] == matched and summary["formal_training_eligible"] == (matched > 0)
    assert summary["registered_attempts"] == len(expected)
    assert summary["environment_steps"] <= len(plan["pairs"])*(1006 if plan["phase"] == "bank" else 1004)
    return {"schema": "rosetta.paired_collection_audit.v2", "passed": True,
            "registered_attempts": len(expected), "matched_N": matched, "admitted_N": summary["N"],
            "N_strict": summary.get("N_strict", summary["N"]),
            "N_exploratory": summary.get("N_exploratory", 0),
            "research_scope": summary.get("research_scope", "strict_legacy_scene_disjoint"),
            "fresh_lineage_verified": fresh["complete"],
            "new_scene_disjoint_verified": fresh["complete"],
            "legacy_object_scene_nonoverlap": summary.get("legacy_object_scene_nonoverlap", "strict_inventory_required"),
            "files_verified": len(paths), "bytes_verified": total,
            "exact_restore_checks_verified": restore_checks,
            "restore_mismatch_results": restore_failures,
            "recovery_chains_verified": event_checks, "full_chunks_verified": chunk_checks,
            "source_qualified": False, "real_model_or_policy_gate_acceptance": False,
            "summary_sha256": sha(root / "summary.json"), "manifest_sha256": sha(root / "manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
