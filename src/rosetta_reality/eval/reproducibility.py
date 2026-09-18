"""File-only exact paired comparison; no torch, model, or simulator imports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gate_diagnostic_io import load_json, relative, seal, sha, write_json
from .gate_diagnostic_verify import _type_tree, arrays

BOUNDARIES = (
    "observation",
    "instruction",
    "batch",
    "noise",
    "normalized",
    "internal",
    "decoded",
    "projected",
)


def difference(a, b, path=""):
    """Return the first structural/bitwise difference, including dtype and signed zero."""
    if type(a) is not type(b):
        return path or "type"
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            return path or "tensor"
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            return path or "keys"
        for key in a:
            found = difference(a[key], b[key], f"{path}/{key}")
            if found:
                return found
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return path or "length"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            found = difference(x, y, f"{path}/{i}")
            if found:
                return found
    elif a != b:
        return path or "value"
    return None


def _files(directory):
    directory = Path(directory)
    manifest = load_json(directory / "manifest.json")
    paths = list(directory.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Evidence symlinks are forbidden")
    if {p.relative_to(directory).as_posix() for p in paths if p.is_file()} != (
        set(manifest["files"]) | {"manifest.json"}
    ):
        raise ValueError("Evidence inventory changed")
    for name, entry in manifest["files"].items():
        path = relative(directory, name)
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("Evidence checksum mismatch: " + name)
    return manifest


def _tree(directory):
    return arrays(directory), _type_tree(load_json(Path(directory) / "tree.json"))


def verify_collection(directory):
    directory = Path(directory)
    manifest = _files(directory)
    identity, result = (load_json(directory / name) for name in ("identity.json", "result.json"))
    if (
        identity["stage"] != "collect-repro"
        or result["stage"] != "collect-repro"
        or result["status"] != manifest["status"]
        or result.get("diagnostic_only") is not True
        or result.get("optimizer_steps") != 0
    ):
        raise ValueError("Reproducibility evidence scope changed")
    if result["status"] != "complete":
        if result["status"] != "incomplete" or result.get("task_success") is not None:
            raise ValueError("Invalid incomplete status")
        return {"status": "verified_incomplete", "stage": "collect-repro"}
    if (
        identity.get("schema_version") != 1
        or identity["identity"].get("kind") not in ("synthetic", "registered_model")
        or identity["pairing"].get("role") not in ("baseline_a", "baseline_b", "full_trace")
        or type(identity["pairing"].get("seed")) is not int
        or identity["pairing"]["seed"] not in range(1000, 1005)
        or not identity["pairing"].get("pair_id")
    ):
        raise ValueError("Invalid pairing identity")
    count = result["steps_saved"]
    if (
        type(count) is not int
        or not 0 < count <= identity["maximum_steps"]
        or result["predictions_saved"] != count
        or result.get("parameters_unchanged") is not True
        or result.get("physics_model_unchanged") is not True
    ):
        raise ValueError("Invalid completed rollout count or parameter check")
    expected = {f"{i:04d}" for i in range(count)}
    for name in ("steps", "predictions"):
        if {p.name for p in (directory / name).iterdir()} != expected:
            raise ValueError("Missing or extra rollout steps")
    reset, _ = _tree(directory / "reset")
    previous = reset
    traced_rows = None
    if identity["pairing"]["role"] == "full_trace":
        from .rollout_trace_verify import verify_trace

        verify_trace(directory / "trace")
        traced_rows = [
            json.loads(line) for line in (directory / "trace/trace.jsonl").read_text().splitlines()
        ]
        traced_rows = [row for row in traced_rows if row["event"] == "step"]
        if len(traced_rows) != count:
            raise ValueError("Trace and capture counts differ")
    elif (directory / "trace").exists():
        raise ValueError("Baseline unexpectedly contains full trace")
    for i in range(count):
        prediction, _ = _tree(directory / f"predictions/{i:04d}")
        before, _ = _tree(directory / f"steps/{i:04d}/before")
        after, _ = _tree(directory / f"steps/{i:04d}/after")
        if not set(BOUNDARIES) <= prediction.keys():
            raise ValueError("Missing captured model boundary")
        if difference(previous["physics"], before["physics"]) or difference(
            previous["observation"], prediction["observation"]
        ):
            raise ValueError("Discontinuous rollout input or integration state")
        physics = before["physics"]
        if (
            not isinstance(physics["integration"], np.ndarray)
            or physics["integration"].ndim != 1
            or physics["integration"].size == 0
            or physics["integration"].dtype != np.float64
            or physics["model_sha256"] != after["physics"]["model_sha256"]
            or physics["spec"] != after["physics"]["spec"]
        ):
            raise ValueError("Invalid integration-state capture")
        if difference(prediction["projected"][0], before["executed"]):
            raise ValueError("Executed action differs from captured first action")
        if traced_rows is not None and (
            traced_rows[i]["step"] != i
            or traced_rows[i]["executed_action"] != before["executed"].tolist()
            or traced_rows[i]["reward"] != after["reward"]
            or traced_rows[i]["done"] != after["done"]
        ):
            raise ValueError("Full trace disagrees with boundary capture")
        if any(type(after[k]) is not bool for k in ("done", "success", "terminated", "truncated")):
            raise ValueError("Missing terminal evidence")
        if after["done"] != (after["terminated"] or after["truncated"]):
            raise ValueError("Inconsistent termination evidence")
        if after["done"] and i != count - 1:
            raise ValueError("Actions executed after terminal state")
        previous = after
    if result["task_success"] is not previous["success"] or (
        count < identity["maximum_steps"] and not previous["done"]
    ):
        raise ValueError("Partial trajectory declared complete")
    load_json(directory / "runtime.json")
    load_json(directory / "runtime-final.json")
    return {"status": "verified_complete", "stage": "collect-repro"}


def _compare(left, right):
    a, b = [load_json(p / "identity.json") for p in (left, right)]
    kinds = [x["identity"].get("kind") for x in (a, b)]
    if kinds[0] != kinds[1] or kinds[0] not in ("synthetic", "registered_model"):
        return {"status": "incomparable", "reason": "provenance"}
    for key in ("pair_id", "seed"):
        if a["pairing"][key] != b["pairing"][key]:
            return {"status": "incomparable", "reason": key}
    for key in ("maximum_steps", "dimension_names", "chunk_length"):
        if a[key] != b[key]:
            return {"status": "incomparable", "reason": key}
    for key in (
        "checkpoint_files",
        "sources",
        "training_plan_sha256",
        "action_contract_sha256",
        "artifact_config_sha256",
    ):
        if a["identity"].get(key) != b["identity"].get(key) or (
            kinds[0] == "registered_model" and not a["identity"].get(key)
        ):
            return {"status": "incomparable", "reason": key}
    if a["execution_id"] == b["execution_id"] or (
        kinds[0] == "registered_model" and a["pid"] == b["pid"]
    ):
        return {"status": "incomparable", "reason": "independent_process_required"}
    for file in ("runtime.json", "runtime-final.json"):
        fa, fb = [load_json(p / file) for p in (left, right)]
        found = difference(fa, fb)
        if found:
            return {"status": "incomparable", "reason": file + found}
        if kinds[0] == "registered_model" and (
            not fa.get("driver")
            or not fa.get("gpu_uuid")
            or not fa.get("libraries")
            or not fa.get("settings")
            or not fa.get("packages")
        ):
            return {"status": "insufficient_evidence", "reason": "runtime_fingerprint"}
    ca, cb = [load_json(p / "result.json")["steps_saved"] for p in (left, right)]
    checks = [(None, "reset", "physics"), (None, "reset", "observation")]
    for i in range(min(ca, cb)):
        checks.append((i, f"steps/{i:04d}/before", "physics"))
        checks.extend((i, f"predictions/{i:04d}", k) for k in BOUNDARIES)
        checks.append((i, f"steps/{i:04d}/before", "executed"))
        checks.extend(
            (i, f"steps/{i:04d}/after", k)
            for k in (
                "physics",
                "observation",
                "reward",
                "done",
                "success",
                "terminated",
                "truncated",
            )
        )
    # Load each compressed bundle once even when it holds several model boundaries.
    cached_name, trees = None, None
    for step, name, field in checks:
        if name != cached_name:
            trees = [_tree(p / name) for p in (left, right)]
            cached_name = name
        values = [(v[field], t[field]) if field else (v, t) for v, t in trees]
        found = difference(values[0][1], values[1][1]) or difference(values[0][0], values[1][0])
        if found:
            return {
                "status": "diverged",
                "step": step,
                "boundary": field or name,
                "phase": name,
                "field": found,
                "unique_cause_proven": False,
            }
    if ca != cb:
        return {"status": "diverged", "step": min(ca, cb), "boundary": "rollout_length"}
    return {"status": "matched", "steps": ca, "scope": "observed_boundaries_bitwise"}


def compare_campaign(baseline, repeat, traced, output):
    directories = [Path(p) for p in (baseline, repeat, traced)]
    output = Path(output)
    if output.exists() or any(output.resolve().is_relative_to(p.resolve()) for p in directories):
        raise ValueError("Comparison output must be new and outside source bundles")
    if len({p.resolve() for p in directories}) != 3:
        raise ValueError("Three distinct source bundles required")
    checks = [verify_collection(p) for p in directories]
    roles = [load_json(p / "identity.json")["pairing"]["role"] for p in directories]
    if roles != ["baseline_a", "baseline_b", "full_trace"]:
        raise ValueError("Expected baseline_a, baseline_b, full_trace order")
    result = {
        "stage": "compare-repro",
        "diagnostic_only": True,
        "optimizer_steps": 0,
        "m2_complete": False,
        "uninstrumented_parity_verified": False,
        "physics_restore_verified": False,
        "source_manifests": [sha(p / "manifest.json") for p in directories],
    }
    kind = load_json(directories[0] / "identity.json")["identity"]["kind"]
    result["provenance_kind"] = kind
    if any(x["status"] != "verified_complete" for x in checks):
        result["status"] = "insufficient_evidence"
    else:
        result["repeat"] = _compare(directories[0], directories[1])
        result["trace_effect"] = _compare(directories[0], directories[2])
        statuses = {result[k]["status"] for k in ("repeat", "trace_effect")}
        result["status"] = next(
            (
                s
                for s in (
                    "incomparable",
                    "insufficient_evidence",
                    "diverged",
                )
                if s in statuses
            ),
            "matched",
        )
    result["real_rollout_parity_verified"] = (
        kind == "registered_model" and result["status"] == "matched"
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "identity.json", {"stage": "compare-repro", "sources": result["source_manifests"]}
    )
    seal(output, result)
    return result
