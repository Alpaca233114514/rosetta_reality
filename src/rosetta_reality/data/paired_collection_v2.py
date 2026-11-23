"""File-only paired collection rules, with no ML or simulation imports.

The CLI loads this source directly to avoid the historical data package's eager
type exports. No change to that existing initializer is needed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


SCHEMA = "rosetta.paired_collection.v2"


def protected_inventory(root, plan):
    """A boolean is never a substitute for actual independently sealed coverage.

This exact-geometry route requires stored original geometry. Sources containing
only motor states/video cannot use it; an alternate observational criterion
needs its own reviewed design and implementation instead of setting true here.
"""
    if plan.get("protected_scene_inventory_complete") is not True:
        return {"complete": False, "fingerprints": [], "inventory": None, "audit": None}
    ref = plan.get("protected_scene_inventory_ref")
    if not isinstance(ref, dict):
        raise ValueError("complete inventory requires independently sealed evidence reference")

    def bound_json(name):
        record = ref[name]
        relative = record["path"]
        part = PurePosixPath(relative)
        if part.is_absolute() or ".." in part.parts or ":" in relative or "\\" in relative:
            raise ValueError("protected inventory reference must be portable")
        path = root.joinpath(*part.parts).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("protected inventory reference escapes workspace")
        raw = path.read_bytes()
        if len(raw) > 16*1024*1024 or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("protected inventory evidence identity mismatch")
        return json.loads(raw), raw

    (inventory, inventory_bytes), (audit, audit_bytes) = bound_json("inventory"), bound_json("audit")
    if (inventory.get("schema") != "rosetta.protected_scene_inventory.v2"
            or inventory.get("criterion") != "complete_initial_geometry_exact_v1"
            or inventory.get("fingerprint_fields") != ["socket", "peg", "robot_state", "left_eef", "right_eef"]
            or audit.get("passed") is not True or audit.get("coverage_complete") is not True
            or audit.get("inventory_sha256") != ref["inventory"]["sha256"]):
        raise ValueError("protected geometry inventory is not independently complete")
    required = {"original_D": set(range(50)), "teacher_tuning": {10,1900,1901,1902,1903,1904},
                "policy_gate": set(range(1000,1005))}
    if plan["phase"] in ("Q2", "bank"):
        required["pilot"] = set(range(2100,2104))
    if plan["phase"] == "Q2":
        required["recovery_bank"] = set(range(2300,2310))
    fingerprints = []
    for category, identities in required.items():
        group = inventory["categories"].get(category)
        if not group or group.get("complete_initial_geometry_observed") is not True:
            raise ValueError("protected source lacks complete observed geometry: " + category)
        records = group["records"]
        if len(records) != len(identities) or {r["identity"] for r in records} != identities:
            raise ValueError("protected source coverage mismatch: " + category)
        for record in records:
            value = record["scene_fingerprint"]
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)
                    or not record.get("source_evidence_sha256")):
                raise ValueError("protected source geometry identity absent")
            fingerprints.append(value)
    original = inventory["categories"]["original_D"]
    if (original.get("dataset") != "lerobot/aloha_sim_insertion_human"
            or original.get("revision") != "cc571a3c661df81b566dbfde3d5c1e85fcdf7884"):
        raise ValueError("protected original D identity differs")
    return {"complete": True, "fingerprints": sorted(set(fingerprints)),
            "inventory": inventory, "audit": audit, "reference": ref,
            "inventory_bytes": inventory_bytes, "audit_bytes": audit_bytes}


def exploratory_registry(registry_root, audit_path, plan):
    """Use independently observed new scene families without inventing legacy states."""
    if registry_root is None or audit_path is None:
        return {"complete": False, "registry": None, "audit": None, "entries": {}, "forbidden": []}
    registry_root, audit_path = Path(registry_root).resolve(), Path(audit_path).resolve()
    manifest_path = registry_root / "manifest.json"
    manifest_raw, audit_raw = manifest_path.read_bytes(), audit_path.read_bytes()
    if (hashlib.sha256(manifest_raw).hexdigest() != plan.get("exploratory_scene_registry_manifest_sha256")
            or hashlib.sha256(audit_raw).hexdigest() != plan.get("exploratory_scene_registry_audit_sha256")):
        raise ValueError("new scene registry/audit is not bound to execution plan")
    manifest, audit = json.loads(manifest_raw), json.loads(audit_raw)
    payloads = {}
    for item in manifest["files"]:
        relative = item["path"]
        part = PurePosixPath(relative)
        if part.is_absolute() or ".." in part.parts or ":" in relative or "\\" in relative or relative in payloads:
            raise ValueError("unsafe or duplicate new scene registry payload")
        path = registry_root.joinpath(*part.parts).resolve()
        if not path.is_relative_to(registry_root):
            raise ValueError("new scene registry payload escapes root")
        if path.stat().st_size != item["bytes"] or sha_file(path) != item["sha256"]:
            raise ValueError("new scene registry payload changed after independent audit")
        payloads[relative] = item
    if "registry.json" not in payloads:
        raise ValueError("registry missing from independently audited inventory")
    registry_raw = (registry_root / "registry.json").read_bytes()
    if hashlib.sha256(registry_raw).hexdigest() != payloads["registry.json"]["sha256"]:
        raise ValueError("registry changed while binding")
    registry = json.loads(registry_raw)
    if (registry.get("schema") != "rosetta.exploratory_scene_registry.v1"
            or audit.get("passed") is not True or audit.get("groups_complete") is not True
            or audit.get("registry_manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest()
            or audit.get("registry_sha256") != hashlib.sha256(registry_raw).hexdigest()
            or registry.get("legacy_object_scene_nonoverlap") != "not_established"):
        raise ValueError("new source families have no complete independent audit")
    family = {key: plan[key] for key in ("phase", "pairs", "contract", "sampling_rule", "minimum_common_chunk_starts")}
    declared_family = registry["plans"][plan["phase"]]
    if (declared_family["family_definition"] != family
            or declared_family["family_definition_sha256"] != digest(family)):
        raise ValueError("collection family differs from actual reset registry")
    expected_groups = {"Q1": set(range(2410,2414)), "Q2": set(range(2420,2440)),
                       "bank": set(range(2440,2450)), "teacher_tuning": {10,1900,1901,1902,1903,1904},
                       "policy_gate": set(range(1000,1005)), "engineering": {2400}}
    if {key: set(value) for key,value in registry["groups"].items()} != expected_groups:
        raise ValueError("new source family registration changed")
    entries = {}
    for entry in registry["entries"]:
        seed, group = entry["seed"], entry["group"]
        if (type(seed) is not int or group not in expected_groups or seed not in expected_groups[group]
                or seed in entries or entry.get("complete_state_captured") is not True):
            raise ValueError("new source family has duplicate/missing complete state")
        if any(entry.get(key) is None for key in ("parent_id", "scene_fingerprint", "reset_state_sha256", "model_sha256")):
            raise ValueError("new scene identity incomplete")
        for key in ("physics_path", "python_state_path"):
            if entry[key] not in payloads:
                raise ValueError("complete new state is absent from audited registry manifest")
        entries[seed] = entry
    if set(entries) != set.union(*expected_groups.values()):
        raise ValueError("new source state coverage incomplete")
    phase = plan["phase"]
    if {p["seed"] for p in plan["pairs"]} != expected_groups[phase]:
        raise ValueError("collection differs from registered new source family")
    forbidden = [e["scene_fingerprint"] for e in entries.values() if e["group"] != phase]
    return {"complete": True, "registry": registry, "audit": audit, "entries": entries,
            "forbidden": forbidden, "registry_bytes": registry_raw,
            "manifest_bytes": manifest_raw, "audit_bytes": audit_raw,
            "registry_root": registry_root, "payloads": list(payloads.values())}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def valid_starts(rows, chunk_length=50):
    """Reject full chunks crossing deliberate/failed/unknown actions, no padding."""
    if chunk_length != 50:
        raise ValueError("registered native chunk length is 50")
    return [i for i in range(max(0, len(rows) - chunk_length + 1))
            if all(row.get("positive_target") is True
                   and row.get("action_source") == "state_conditioned_teacher"
                   and row.get("step") == j
                   for j, row in enumerate(rows[i:i + chunk_length], i))]


def symmetric_sampler(s_rows, r_rows, deviation_step, *, minimum_common_starts=50):
    """Preregistered symmetric exclusion plus common legal full-chunk anchors.

Raw expert labels and native legal starts remain untouched. The new sampling
eligibility is disclosed separately, including every rejected anchor.
"""
    if minimum_common_starts != 50:
        raise ValueError("003-B minimum common anchors must be 50")
    s_native, r_native = valid_starts(s_rows), valid_starts(r_rows)
    reached = (type(deviation_step) is int and 0 <= deviation_step < min(len(s_rows), len(r_rows)))
    prefix_exact = False
    excluded = []
    if reached:
        d = deviation_step
        keys = ("state", "action", "before", "after", "phase", "images_sha256", "action_source")
        prefix_exact = all(all(key in s_rows[i] and key in r_rows[i]
                               and s_rows[i][key] == r_rows[i][key] for key in keys)
                           for i in range(d))
        prefix_exact = prefix_exact and all(key in s_rows[d] and key in r_rows[d]
                                           and s_rows[d][key] == r_rows[d][key]
                                           for key in ("state", "before", "phase", "images_sha256"))
        excluded = [i for i in s_native if i <= d < i+50]
    s_selected = sorted(set(s_native)-set(excluded)) if reached and prefix_exact else []
    common = sorted(set(s_selected) & set(r_native))
    return {"schema": "rosetta.paired_symmetric_sampler.v1",
            "sampling_rule": "symmetric_common_legal_chunk_grid_v1",
            "deviation_step": deviation_step, "event_reached": reached,
            "pre_deviation_prefix_exact": prefix_exact,
            "prefix_actions_compared": deviation_step if reached else 0,
            "S_native_valid_chunk_starts": s_native, "R_native_valid_chunk_starts": r_native,
            "S_symmetric_excluded_starts": excluded,
            "S_after_symmetric_exclusion_starts": s_selected,
            "S_unshared_after_symmetric_exclusion": sorted(set(s_selected)-set(common)),
            "R_unshared_native_starts": sorted(set(r_native)-set(common)),
            "common_valid_chunk_starts": common, "common_starts_sha256": digest(common),
            "minimum_common_chunk_starts": minimum_common_starts,
            "sufficient_common_starts": len(common) >= minimum_common_starts,
            "raw_S_input_frames": len(s_native), "raw_R_input_frames": len(r_native),
            "raw_S_valid_target_slots": len(s_native)*50, "raw_R_valid_target_slots": len(r_native)*50,
            "sampler_effective_input_frames_per_role": len(common),
            "sampler_valid_target_slots_per_role": len(common)*50,
            "raw_labels_changed": False, "raw_frames_deleted": False, "native_loss_changed": False}


def match_pair(s, r, *, tolerance=0.05, sampler=None):
    reasons = []
    for key in ("provider_id", "pair_id", "parent_id", "scene_fingerprint",
                "reset_state_sha256", "contract_sha256", "model_sha256"):
        if s.get(key) != r.get(key) or s.get(key) is None:
            reasons.append("identity_mismatch:" + key)
    if s.get("kind") != "S" or r.get("kind") != "R":
        reasons.append("trajectory_kind")
    for item in (s, r):
        if not item.get("task_success") or item.get("safety_stop"):
            reasons.append("no_safe_task_success:" + item["kind"])
    if s.get("deviation") is not None:
        reasons.append("S_has_intervention")
    if not r.get("verified_recovery_chain"):
        reasons.append("R_missing_recovery_chain")
    if not s.get("restore_exact") or not r.get("restore_exact"):
        reasons.append("restore_not_exact")
    if sampler is None:
        for key in ("effective_input_frames", "valid_target_slots"):
            a, b = s.get(key, 0), r.get(key, 0)
            if min(a, b) <= 0 or abs(a - b) / max(a, b, 1) > tolerance:
                reasons.append("unmatched:" + key)
    else:
        if sampler.get("sampling_rule") != "symmetric_common_legal_chunk_grid_v1":
            reasons.append("unknown_sampling_rule")
        if not sampler.get("pre_deviation_prefix_exact"):
            reasons.append("pre_deviation_prefix_not_exact")
        if not sampler.get("sufficient_common_starts"):
            reasons.append("fewer_than_50_common_legal_chunks")
    return {"pair_id": s["pair_id"], "matched": not reasons,
            "reasons": sorted(set(reasons)), "S": s["attempt_id"], "R": r["attempt_id"],
            "sampling_rule": sampler["sampling_rule"] if sampler else "all_native_legal_chunks"}


def validate_plan(plan):
    if plan.get("schema") != "rosetta.paired_collection_plan.v2":
        raise ValueError("unknown plan schema")
    phase = plan["phase"]
    scope = plan.get("research_scope", "strict_legacy_scene_disjoint")
    if scope not in ("strict_legacy_scene_disjoint", "fresh_lineage_fixed_development_exploration_v1"):
        raise ValueError("unregistered research boundary")
    if scope == "fresh_lineage_fixed_development_exploration_v1" and (
            plan.get("N_strict") != 0 or plan.get("legacy_object_scene_nonoverlap") != "not_established"):
        raise ValueError("exploration cannot pretend strict legacy scene isolation")
    if phase not in ("Q1", "Q2", "bank"):
        raise ValueError("unknown collection phase")
    cap = {"Q1": 4, "Q2": 20, "bank": 10}[phase]
    pairs = plan["pairs"]
    if not 0 < len(pairs) <= cap:
        raise ValueError("pair count outside registered cap")
    ids = [p["pair_id"] for p in pairs]
    seeds = [p["seed"] for p in pairs]
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise ValueError("duplicate registered pair or seed")
    if set(seeds) & set(plan["forbidden_seeds"]):
        raise ValueError("protected/tuning/pilot/gate seed overlap")
    if plan["max_steps"] != 500 or plan["correction_max_steps"] != 150:
        raise ValueError("step budget drift")
    if plan["chunk_length"] != 50 or plan["fps"] != 50:
        raise ValueError("action contract temporal drift")
    if not 0 < plan["work_cap_seconds"] <= (1800 if phase == "Q1" else 7200):
        raise ValueError("work budget drift")
    if not 0 < plan["evidence_byte_cap"] <= (268435456 if phase == "Q1" else 1073741824):
        raise ValueError("evidence budget drift")
    if plan["recovery_sustain_steps"] != 5 or plan["deviation_min_joint_change_rad"] != 0.002:
        raise ValueError("event threshold drift")
    symmetric = plan.get("sampling_rule") == "symmetric_common_legal_chunk_grid_v1"
    if plan.get("sampling_rule", "all_native_legal_chunks") not in (
            "all_native_legal_chunks", "symmetric_common_legal_chunk_grid_v1"):
        raise ValueError("unregistered auxiliary sampling rule")
    if symmetric and plan.get("minimum_common_chunk_starts") != 50:
        raise ValueError("003-B minimum common chunk count drift")
    if plan.get("receipt_encoding", "plain_json_v1") not in ("plain_json_v1", "lossless_json_gzip_v1"):
        raise ValueError("unregistered complete-state receipt encoding")
    for p in pairs:
        if p["trigger"] not in (("grasp", "align") if symmetric else ("approach", "align")):
            raise ValueError("unregistered event")
        if p["perturbation_steps"] != 1 or abs(p["joint_delta_rad"]) != 0.04:
            raise ValueError("unregistered perturbation amplitude/duration")
        if p["joint_name"] != "left_waist":
            raise ValueError("unregistered perturbation dimension")
    return plan
