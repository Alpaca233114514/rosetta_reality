"""Evidence qualification for candidate S/R cohorts, never a training adapter."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Evidence paths must be portable relative paths")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Evidence path escapes its root")
    root = root.resolve()
    target = root.joinpath(*path.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Evidence symlink escapes its root")
    return target


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def evidence(root: Path, reference: dict | None) -> dict | None:
    if not reference:
        return None
    path = safe_path(root, reference["path"])
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Qualification evidence must be bounded JSON, not a model/data payload")
    if file_sha(path) != reference["sha256"]:
        raise ValueError("Qualification evidence hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Qualification evidence must be an object")
    return value


def qualify(record: dict, root: Path, expected_contract: str) -> dict:
    """Require actual row, contract, outcome and label evidence for each candidate."""
    reasons = []
    for key in ("dataset", "revision", "episode", "scene", "parent_trajectory", "collection_family"):
        if record.get(key) is None or record.get(key) == "":
            reasons.append("missing_" + key)
    if not isinstance(record.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", record["revision"]):
        reasons.append("invalid_source_revision")
    if not isinstance(record.get("content_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["content_sha256"]):
        reasons.append("missing_source_content_identity")
    if type(record.get("episode")) is not int or record["episode"] < 0:
        reasons.append("invalid_episode")
    if record.get("role") not in ("S", "R"):
        reasons.append("unknown_role")
    if record.get("already_in_D") is not False:
        reasons.append("not_verified_new_trajectory")
    if record.get("split") != "train":
        reasons.append("not_train_split")
    for key in ("effective_frames", "valid_target_slots"):
        if type(record.get(key)) is not int or record[key] <= 0:
            reasons.append("missing_" + key)
    padding = record.get("padding_fraction")
    if type(padding) not in (int, float) or not math.isfinite(padding) or not 0 <= padding <= 1:
        reasons.append("unknown_padding")
    for kind in ("contract", "rows", "labels"):
        proof = evidence(root, record.get(kind + "_evidence"))
        if proof is None:
            reasons.append("missing_" + kind + "_evidence")
            continue
        # Bind the report to the exact source trajectory and its immutable revision.
        if any(proof.get(k) != record.get(k) for k in ("dataset", "revision", "episode")):
            reasons.append(kind + "_source_binding_mismatch")
        if kind == "contract":
            if (proof.get("status") != "passed" or proof.get("action_contract_sha256") != expected_contract
                    or proof.get("semantic_equivalence_verified") is not True
                    or proof.get("gate1_passed") is not True or proof.get("gate2_passed") is not True):
                reasons.append("contract_not_verified")
        elif kind == "rows":
            if (proof.get("status") != "passed" or proof.get("hidden_loaded") is not False
                    or proof.get("finite_state_action") is not True
                    or proof.get("video_alignment_verified") is not True
                    or proof.get("effective_frames") != record.get("effective_frames")
                    or proof.get("valid_target_slots") != record.get("valid_target_slots")
                    or proof.get("padding_fraction") != record.get("padding_fraction")
                    or proof.get("source_records_sha256") != record.get("content_sha256")):
                reasons.append("rows_or_video_not_verified")
        else:
            if proof.get("task_outcome") != "success" or proof.get("outcome_verified") is not True:
                reasons.append("outcome_not_verified_success")
            if record.get("role") == "S":
                if proof.get("ordinary_success_verified") is not True or proof.get("recovery_present") is not False:
                    reasons.append("ordinary_success_not_verified")
            elif record.get("role") == "R":
                events = [proof.get(k) for k in ("deviation_frame", "correction_start", "correction_end", "recovery_frame", "success_frame")]
                if (not all(type(v) is int and v >= 0 for v in events)
                        or not events[0] <= events[1] <= events[2] <= events[3] <= events[4]
                        or not type(record.get("effective_frames")) is int
                        or events[4] >= record["effective_frames"]):
                    reasons.append("recovery_event_chain_missing_or_invalid")
                if (proof.get("state_conditioned_expert") is not True
                        or proof.get("continuous_context_verified") is not True
                        or proof.get("action_source") not in ("human_correction", "qualified_teacher")
                        or proof.get("time_indexed_replay") is not False):
                    reasons.append("recovery_supervision_not_verified")
                if proof.get("action_source") == "qualified_teacher":
                    teacher = evidence(root, proof.get("teacher_gate_evidence"))
                    if (teacher is None or teacher.get("status") != "passed"
                            or teacher.get("state_conditioned") is not True
                            or teacher.get("action_contract_sha256") != expected_contract):
                        reasons.append("teacher_gate_not_verified")
    return {"candidate": record, "eligible": not reasons, "reasons": reasons,
            "qualification_is_independent_truth_proof": False}


def match(records: list[dict], pairs: list[dict], protected: list[dict], root: Path,
          contract: str, tolerance: float = 0.05) -> dict:
    if not 0 <= tolerance <= 0.05:
        raise ValueError("Matching tolerance cannot exceed the design's 5 percent bound")
    if len(records) > 40 or len(pairs) > 20:
        raise ValueError("Candidate/pair count exceeds the first-round design bound")
    identities = [(r.get("dataset"), r.get("revision"), r.get("episode")) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate trajectory identity")
    findings = [qualify(r, root, contract) for r in records]
    # Protect validation, hidden, D, Gate, teacher/tuning scenes and descendants.
    for item in findings:
        candidate = item["candidate"]
        for other in protected:
            same_episode = all(candidate.get(k) == other.get(k) for k in ("dataset", "revision", "episode"))
            if same_episode or any(candidate.get(k) is not None and candidate.get(k) == other.get(k)
                                   for k in ("scene", "parent_trajectory", "content_sha256", "collection_family")):
                item["reasons"].append("scene_parent_or_content_overlap")
                item["eligible"] = False
                break
        for other in records:
            if other is candidate:
                continue
            if any(candidate.get(k) is not None and candidate.get(k) == other.get(k)
                   for k in ("parent_trajectory", "content_sha256")):
                item["reasons"].append("duplicate_candidate_parent_or_content")
                item["eligible"] = False
                break
    pair_results, used = [], set()
    for pair in pairs:
        s, r = pair["S"], pair["R"]
        if any(type(i) is not int or not 0 <= i < len(records) for i in (s, r)):
            raise ValueError("Pair indices outside candidate inventory")
        if s == r or s in used or r in used:
            raise ValueError("A trajectory cannot be reused in multiple pairs")
        used.update((s, r))
        left, right = records[s], records[r]
        reasons = []
        if left.get("role") != "S" or right.get("role") != "R":
            reasons.append("pair_role_mismatch")
        if not findings[s]["eligible"] or not findings[r]["eligible"]:
            reasons.append("unqualified_pair_member")
        for key in ("task", "embodiment", "scene_stratum", "initial_pose_stratum", "collector"):
            if left.get(key) is None or left.get(key) != right.get(key):
                reasons.append("unmatched_" + key)
        for key in ("effective_frames", "valid_target_slots", "duration_seconds"):
            a, b = left.get(key), right.get(key)
            if (type(a) not in (int, float) or type(b) not in (int, float)
                    or not math.isfinite(a) or not math.isfinite(b) or min(a, b) <= 0
                    or abs(a - b) / max(a, b) > tolerance):
                reasons.append("unmatched_" + key)
        a, b = left.get("padding_fraction"), right.get("padding_fraction")
        if type(a) not in (int, float) or type(b) not in (int, float) or abs(a - b) > tolerance:
            reasons.append("unmatched_padding")
        pair_results.append({"S": s, "R": r, "eligible": not reasons, "reasons": reasons})
    return {"schema": "rosetta.cohort_qualification.v1", "candidates": findings,
            "pairs": pair_results, "qualified_pair_count": sum(p["eligible"] for p in pair_results),
            "cohort_complete": bool(pair_results) and all(p["eligible"] for p in pair_results),
            "training_ready": False, "reason": "requires_separate_sampler_runtime_and_recovery_bank_seals",
            "scope": "hash_bound_evidence_validation_not_training_manifest_adaptation"}
