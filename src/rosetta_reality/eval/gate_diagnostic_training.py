"""Evidence-bound training contract checks; never executes training or loads weights."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .gate_diagnostic_io import load_json, seal, sha, write_json
from .gate_diagnostic_protocol import WEIGHT_SHA, reference

CHECKS = {
    "sample_consumption": "src/rosetta_reality/vla/training/observation.py",
    "time_chunk_padding": "src/rosetta_reality/vla/training/temporal_sampler.py",
    "split": "src/rosetta_reality/vla/training/plan.py",
    "camera": "scripts/smolvla_autodl_vfunfreeze_sim_gate.py",
    "pixels": "src/rosetta_reality/vla/image_scaling.py",
    "processor": "src/rosetta_reality/vla/processor.py",
    "loss": "src/rosetta_reality/vla/horizon_loss.py",
    "horizon": "scripts/smolvla_sim_gate.py",
    "freeze": "src/rosetta_reality/vla/training/integrity.py",
    "optimizer": "src/rosetta_reality/vla/training/features.py",
    "scheduler": "src/rosetta_reality/vla/training/observation.py",
    "gradients": "src/rosetta_reality/vla/training/features.py",
    "reload": "src/rosetta_reality/vla/reload_evidence.py",
}


def pointer(value, path):
    """JSON Pointer, including array indices; absent facts are not silently defaulted."""
    if path == "":
        return value
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("JSON pointer required")
    for token in path[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def check_fact(name, fact):
    """Check recorded operands, not a producer's pass/fail flag."""
    if name in ("sample_consumption", "camera", "pixels", "processor", "reload"):
        if fact["expected"] is None or fact["actual"] is None:
            raise ValueError("Missing measurement")
        return fact["actual"] == fact["expected"]
    if name == "split":
        train, dev, hidden = (set(fact[k]) for k in ("train", "development", "hidden"))
        return (
            not (train & dev or train & hidden or dev & hidden)
            and not set(fact["materialized"]) & hidden
        )
    if name == "time_chunk_padding":
        count = fact["episode_length"]
        frame = fact["frame"]
        horizon = fact["chunk_length"]
        return (
            0 <= frame < count
            and horizon > 0
            and fact["target_indices"] == [min(frame + i, count - 1) for i in range(horizon)]
            and fact["is_pad"] == [frame + i >= count for i in range(horizon)]
            and all(ep == fact["episode"] for ep in fact["target_episodes"])
            and len(fact["target_episodes"]) == horizon
            and fact["input_timestamp"] == fact["expected_timestamp"]
        )
    if name == "loss":
        weights, padding = fact["weights"], fact["is_pad"]
        if len(weights) != len(padding) or not weights:
            return False
        denominator = sum(w for w, pad in zip(weights, padding) if not pad) * fact["action_dim"]
        return (
            denominator > 0 and all(w >= 0 for w in weights) and fact["denominator"] == denominator
        )
    if name == "horizon":
        return fact["training_horizon"] == fact["executed_horizon"]
    if name == "freeze":
        return set(fact["actual_trainable"]) == set(fact["expected_trainable"]) and not set(
            fact["changed_parameters"]
        ) & set(fact["frozen_parameters"])
    if name == "optimizer":
        if not fact["trainable_parameters"]:
            raise ValueError("Missing trainable parameter inventory")
        return Counter(fact["optimizer_parameters"]) == Counter(set(fact["trainable_parameters"]))
    if name == "scheduler":
        return (
            fact["optimizer_updates"] == fact["scheduler_updates"]
            and fact["actual_lr"] == fact["expected_lr"]
        )
    if name == "gradients":
        import math

        values = (fact["preclip_norm"], fact["postclip_norm"], fact["clip_limit"])
        return (
            all(math.isfinite(v) and v >= 0 for v in values)
            and fact["postclip_norm"] <= min(fact["preclip_norm"], fact["clip_limit"]) + 1e-5
            and fact["nonfinite_count"] == 0
        )
    raise ValueError("Unknown training check")


def finding(name, category, evidence, detail, *, control=None):
    return {
        "check": name,
        "category": category,
        "phenomenon": detail,
        "evidence": evidence,
        "control": control,
        "code": CHECKS.get(name, "scripts/diagnose_smolvla_gate.py"),
        "boundary": "Recorded scope only; does not prove the cause of Seed 3 failure.",
        "next_minimal_check": "Reproduce this boundary with identical inputs and pinned code.",
    }


def audit_training(plan, root, output):
    """Evidence entries bind pointers into historical JSONs and archived source files."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "identity.json",
        {
            "schema_version": 1,
            "stage": "audit-training",
            "diagnostic_only": True,
            "run_id": plan["run_id"],
            "checkpoint_sha256": WEIGHT_SHA,
        },
    )
    findings, covered = [], set()
    result = {
        "stage": "audit-training",
        "status": "incomplete",
        "diagnostic_only": True,
        "optimizer_steps": 0,
        "findings": findings,
        "error_type": None,
    }
    try:
        for entry in plan.get("training_evidence", []):
            report_path = reference(root, entry)
            report = load_json(report_path)
            evidence = {"path": entry["path"], "sha256": sha(report_path)}
            try:
                binding = entry["binding"]
                if (
                    pointer(report, binding["checkpoint_pointer"]) != WEIGHT_SHA
                    or pointer(report, binding["training_plan_pointer"])
                    != plan["training_plan"]["sha256"]
                ):
                    raise ValueError("Historical endpoint differs")
                sources = pointer(report, binding["sources_pointer"])
                if not sources or not isinstance(sources, dict):
                    raise ValueError("No historical execution source inventory")
                for name, digest in sources.items():
                    archived = binding.get("archived_sources", {}).get(name, name)
                    reference(root, {"path": archived, "sha256": digest})
            except (KeyError, IndexError, TypeError, ValueError, FileNotFoundError) as exc:
                findings.append(
                    finding(
                        "provenance",
                        "insufficient_evidence",
                        evidence,
                        "Historical code/model binding unavailable: " + type(exc).__name__,
                    )
                )
                continue
            for name, location in entry.get("fact_pointers", {}).items():
                if name not in CHECKS:
                    raise ValueError("Unregistered training diagnostic")
                try:
                    required = entry.get("check_sources", {}).get(name, [CHECKS[name]])
                    if not required or any(path not in sources for path in required):
                        raise ValueError("Check is not bound to its executed source")
                    fact = pointer(report, location)
                    passed = check_fact(name, fact)
                except (KeyError, IndexError, TypeError, ValueError):
                    findings.append(
                        finding(
                            name,
                            "insufficient_evidence",
                            evidence,
                            "Required recorded operands are missing or invalid.",
                        )
                    )
                    continue
                covered.add(name)
                category = (
                    "checked_contract"
                    if passed
                    else ("contract_risk" if name == "horizon" else "reproduced_code_defect")
                )
                findings.append(
                    finding(
                        name,
                        category,
                        {**evidence, "pointer": location},
                        "Recorded contract matches."
                        if passed
                        else "Recorded operands violate the declared contract.",
                        control=fact,
                    )
                )
        for name in sorted(CHECKS.keys() - covered):
            findings.append(
                finding(
                    name,
                    "insufficient_evidence",
                    None,
                    "No identity-matched operands; historical values are not inferred.",
                )
            )
        result["status"] = "complete"
        return result
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        raise
    finally:
        seal(output, result)
