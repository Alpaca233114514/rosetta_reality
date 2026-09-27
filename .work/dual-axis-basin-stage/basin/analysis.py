"""Deterministic arithmetic. Findings describe observations, never causes."""

import math


def pointer(value, path):
    if not path:
        return value
    if not path.startswith("/"):
        raise ValueError("Use a JSON Pointer beginning with /")
    for part in path[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def leaves(value, path=""):
    if isinstance(value, dict):
        for key in sorted(value):
            escaped = key.replace("~", "~0").replace("/", "~1")
            yield from leaves(value[key], path + "/" + escaped)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, path + "/" + str(index))
    else:
        yield path, value


def differences(a, b, tolerance=0.0, path=""):
    """Keep missing/null and structural changes distinct; bool is not a number."""
    if type(a) in (int, float) and type(b) in (int, float):
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("Non-finite comparison")
        delta = b - a
        if abs(delta) > tolerance:
            yield {"path": path, "left": a, "right": b, "delta": delta}
    elif type(a) is not type(b):
        yield {"path": path, "left": a, "right": b, "kind": "type_changed"}
    elif isinstance(a, dict):
        for key in sorted(a.keys() | b.keys()):
            p = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in a or key not in b:
                yield {"path": p, "kind": "missing", "left_present": key in a,
                       "right_present": key in b}
            else:
                yield from differences(a[key], b[key], tolerance, p)
    elif isinstance(a, list):
        if len(a) != len(b):
            yield {"path": path, "kind": "shape_changed", "left_length": len(a), "right_length": len(b)}
        else:
            for index, (x, y) in enumerate(zip(a, b)):
                yield from differences(x, y, tolerance, path + "/" + str(index))
    elif a != b:
        yield {"path": path, "left": a, "right": b}


def analyze(store, run_id):
    record, events = store.get(run_id)
    series = {}
    for event in events:
        for path, value in leaves(event["values"]):
            if type(value) in (float, int):
                if not math.isfinite(value):
                    raise ValueError("Non-finite evidence")
                key = event["stage"] + path
                series.setdefault(key, []).append((event["step"], value))
    stats = {key: {"count": len(values), "minimum": min(v for _, v in values),
                   "maximum": max(v for _, v in values),
                   "mean": math.fsum(v / len(values) for _, v in values),
                   "first_step": min(s for s, _ in values), "last_step": max(s for s, _ in values)}
             for key, values in series.items()}
    execution = [e for e in events if e["stage"] == "execution"]
    rewards = [e["values"]["reward"] for e in execution if "reward" in e["values"]]
    successes = [e["values"]["success"] for e in execution if "success" in e["values"]]
    facts = {"recorded_events": len(events), "completed_execution_steps": len(execution),
             "maximum_reward": max(rewards) if rewards else None,
             "episode_success": any(successes) if successes else None,
             "gate": record["gate"]}
    return {"run_id": run_id, "evidence_kind": record["evidence_kind"], "status": record["status"],
            "facts": facts, "statistics": stats, "hypotheses": [],
            "limitations": record.get("limitations", []) + [
                "Statistics cover recorded values only; absent values are not zeros.",
                "Episode success is not Gate acceptance or M2 completion."]}


def compare(store, left_id, right_id, field="/action", tolerance=0.0):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and non-negative")
    left, a = store.get(left_id)
    right, b = store.get(right_id)
    reasons = []
    if left["evidence_kind"] != right["evidence_kind"]:
        reasons.append("different evidence kinds")
    if left["adapter"] != right["adapter"]:
        reasons.append("different adapters")
    if left.get("dimensions") != right.get("dimensions"):
        reasons.append("different dimension semantics")
    # For Rosetta, Action Contract identity must match before numeric comparison.
    if left["adapter"] == "rosetta.rollout_trace.v1":
        contract = "/identity/original_protocol/action_contract_sha256"
        try:
            lc, rc = pointer(left["parameters"], contract), pointer(right["parameters"], contract)
            if not lc or lc != rc:
                reasons.append("missing or different Action Contract identity")
        except (KeyError, TypeError, IndexError):
            reasons.append("missing Action Contract identity")
    parameter_diff = list(differences(left["parameters"], right["parameters"]))
    result = {"left": left_id, "right": right_id, "field": field, "absolute_tolerance": tolerance,
              "comparable": not reasons, "incompatibilities": reasons,
              "parameter_difference_count": len(parameter_diff), "parameter_differences": parameter_diff[:100],
              "gate_left": left["gate"], "gate_right": right["gate"],
              "hypotheses": [], "causal_conclusion": None,
              "limitations": ["Earliest recorded difference is not a failure cause.",
                              "Alignment uses (step, stage), not wall time or equivalent physical state.",
                              "Equal recorded fields do not establish full-run equivalence."]}
    if reasons:
        return result
    a = {(e["step"], e["stage"]): e for e in a}
    b = {(e["step"], e["stage"]): e for e in b}
    changed, compared, missing, max_delta, first = 0, 0, [], None, None
    for key in sorted(a.keys() | b.keys()):
        if key not in a or key not in b:
            missing.append({"step": key[0], "stage": key[1], "reason": "unmatched_event"})
            continue
        try:
            x, y = pointer(a[key]["values"], field), pointer(b[key]["values"], field)
        except (KeyError, IndexError, TypeError):
            missing.append({"step": key[0], "stage": key[1], "reason": "missing_field"})
            continue
        if x is None or y is None:
            missing.append({"step": key[0], "stage": key[1], "reason": "null_observation"})
            continue
        compared += 1
        changes = list(differences(x, y, tolerance, field))
        if changes:
            changed += 1
            if first is None:
                first = {"step": key[0], "stage": key[1], "changes": changes[:20],
                         "left_evidence": a[key]["evidence"], "right_evidence": b[key]["evidence"]}
        # Maximum observed delta includes sub-tolerance differences.
        if type(x) in (int, float) and type(y) in (int, float):
            max_delta = max_delta if max_delta is not None else 0.0
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            xl, yl = dict(leaves(x)), dict(leaves(y))
            if any(type(v) in (int, float) and type(yl.get(p)) in (int, float) for p, v in xl.items()):
                max_delta = max_delta if max_delta is not None else 0.0
        for change in differences(x, y, 0.0):
            if "delta" in change:
                max_delta = max(max_delta or 0.0, abs(change["delta"]))
    result.update({"compared_events": compared, "changed_events": changed,
                   "missing_count": len(missing), "missing": missing[:100],
                   "coverage_complete": bool(compared) and not missing,
                   "maximum_absolute_delta": max_delta, "first_recorded_difference": first})
    return result
