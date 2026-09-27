"""Import and summarize Basin's safe TorchLens JSON export without loading TorchLens.

TorchLens ModelHistory objects are executable Python objects, so this module only
accepts the JSON envelope emitted by ``capture_synthetic``. All calculations use
the saved activation values; graph and runtime transparency remain unverified.
"""

import math
import re
from pathlib import Path

from .io import loads, read_bytes

MAX_BYTES = 16 * 1024 * 1024
MAX_OPS = 4096
MAX_ELEMENTS = 4096
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _numbers(value, shape):
    if not shape:
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("Invalid or non-finite TorchLens activation")
        yield value
        return
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError("TorchLens activation does not match its shape")
    for item in value:
        yield from _numbers(item, shape[1:])


def validate_envelope(value):
    """Validate the bounded, data-only native envelope emitted by our collector."""
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported TorchLens envelope schema")
    params = value.get("parameters")
    if not isinstance(params, dict) or params.get("collector") != "basin.torchlens.v1":
        raise ValueError("Expected Basin TorchLens JSON export")
    if value.get("evidence_kind") != "synthetic" or value.get("status") not in ("complete", "incomplete"):
        raise ValueError("Unsupported TorchLens evidence kind or status")
    if not isinstance(value.get("source_run"), str) or not value["source_run"]:
        raise ValueError("Missing TorchLens source run")
    if not isinstance(value.get("events"), list) or len(value["events"]) > MAX_OPS:
        raise ValueError("TorchLens operation budget exceeded")
    if params.get("consistency") != "not_assessed" or params.get("live_integration_accepted") is not False:
        raise ValueError("TorchLens transparency claims are unsupported")
    if value["status"] == "incomplete":
        if value["events"] or not isinstance(params.get("error_type"), str):
            raise ValueError("Incomplete TorchLens export needs an error and no published events")
        return value
    for index, event in enumerate(value["events"]):
        if (not isinstance(event, dict) or type(event.get("step")) is not int
                or event["step"] != 0 or not isinstance(event.get("values"), dict)):
            raise ValueError("Invalid TorchLens operation event")
        op = event["values"]
        if (not isinstance(op.get("module"), str) or not op["module"]
                or type(op.get("call_index")) is not int or op["call_index"] < 0
                or not isinstance(op.get("label"), str) or not op["label"]
                or not isinstance(op.get("parents"), list)
                or not all(isinstance(p, str) for p in op["parents"])
                or not isinstance(op.get("dtype"), str) or not op["dtype"]
                or type(op.get("saved")) is not bool):
            raise ValueError("Invalid TorchLens operation metadata")
        expected_stage = f"module:{op['module']}/call:{op['call_index']}/op:{index}"
        if event.get("stage") != expected_stage:
            raise ValueError("TorchLens operation index or stage mismatch")
        shape = op.get("shape")
        if (not isinstance(shape, list) or len(shape) > 32
                or any(type(n) is not int or n < 0 for n in shape)):
            raise ValueError("Invalid TorchLens tensor shape")
        elements = math.prod(shape)
        if op["saved"]:
            if (elements > MAX_ELEMENTS or op.get("finite") is not True
                    or not isinstance(op.get("sha256"), str)
                    or not _DIGEST.fullmatch(op["sha256"])):
                raise ValueError("Invalid saved TorchLens activation")
            if sum(1 for _ in _numbers(op.get("activation"), shape)) != elements:
                raise ValueError("TorchLens activation element count mismatch")
        elif op.get("finite") is not None or "activation" in op or "sha256" in op:
            raise ValueError("Unsaved TorchLens operation contains activation data")
    return value


def import_torchlens(store, source, run_id):
    """Copy a validated JSON export and publish its indexed events create-only."""
    source = Path(source).resolve()
    if source.is_relative_to(store.root):
        raise ValueError("TorchLens input must be outside the history store")
    data = read_bytes(source, MAX_BYTES)
    try:
        value = validate_envelope(loads(data))
    except RecursionError as exc:
        raise ValueError("TorchLens JSON nesting exceeds supported depth") from exc
    events = [{**event, "evidence": {"artifact": "torchlens.json", "pointer": f"/events/{index}"}}
              for index, event in enumerate(value["events"])]
    record = {"schema_version": 1, "id": run_id, "adapter": "basin.torchlens.v1",
              "source_run": value["source_run"], "status": value["status"],
              "evidence_kind": "synthetic", "gate": value.get("gate", {}),
              "parameters": value["parameters"], "dimensions": value.get("dimensions", []),
              "verification": {"source_hashes": "computed_at_import_not_upstream_verified",
                               "activation_sha256": "collector_claim_not_independently_verified",
                               "graph_completeness": "not_independently_verified"},
              "limitations": ["Synthetic capture only; no live execution or RNG transparency claim.",
                              "Activation digests are collector claims; JSON values are analyzed directly."]}
    return store.add(record, events, {"torchlens.json": data})


class _Stats:
    def __init__(self):
        self.operations = 0
        self.saved_operations = 0
        self.elements = 0
        self.minimum = None
        self.maximum = None
        self.mean = 0.0
        self.scale = 0.0
        self.ssq = 1.0

    def add(self, op):
        self.operations += 1
        if not op["saved"]:
            return
        self.saved_operations += 1
        for number in _numbers(op["activation"], op["shape"]):
            self.elements += 1
            self.minimum = number if self.minimum is None else min(self.minimum, number)
            self.maximum = number if self.maximum is None else max(self.maximum, number)
            # Scale before summation: opposing large finite values can otherwise
            # overflow an intermediate even when their mean is representable.
            self.mean += (number / self.elements) - (self.mean / self.elements)
            magnitude = abs(number)
            if magnitude > self.scale:
                self.ssq = 1 + self.ssq * (self.scale / magnitude) ** 2
                self.scale = magnitude
            elif magnitude:
                self.ssq += (magnitude / self.scale) ** 2

    def result(self):
        norm = self.scale * math.sqrt(self.ssq) if self.elements else None
        if norm is not None and not math.isfinite(norm):
            norm = None  # A finite input may have a norm outside JSON's finite range.
        rms = (self.scale * math.sqrt(self.ssq / self.elements)) if self.elements else None
        if rms is not None and not math.isfinite(rms):
            rms = None
        return {"operations": self.operations, "saved_operations": self.saved_operations,
                "observed_elements": self.elements, "minimum": self.minimum,
                "maximum": self.maximum, "mean": self.mean if self.elements else None,
                "l2_norm": norm, "rms": rms}


def _graph(events):
    labels = {}
    for index, event in enumerate(events):
        labels.setdefault(event["values"]["label"], []).append(index)
    edges, issues = [], []
    children = [[] for _ in events]
    indegree = [0] * len(events)
    parent_references = 0
    for target, event in enumerate(events):
        for parent in event["values"]["parents"]:
            parent_references += 1
            candidates = labels.get(parent, [])
            if len(candidates) != 1:
                issues.append({"op": target, "parent_label": parent,
                               "kind": "unresolved" if not candidates else "ambiguous"})
                continue
            source = candidates[0]
            edges.append({"from_op": source, "to_op": target, "parent_label": parent})
            if source == target:
                issues.append({"op": target, "parent_label": parent, "kind": "self_reference"})
            children[source].append(target)
            indegree[target] += 1
    roots = [index for index, degree in enumerate(indegree) if degree == 0]
    depth = [0] * len(events)
    queue = list(roots)
    position = 0
    while position < len(queue):
        source = queue[position]
        position += 1
        for target in children[source]:
            depth[target] = max(depth[target], depth[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    cyclic = len(queue) != len(events)
    if cyclic:
        issues.extend({"op": index, "kind": "cycle_member_or_downstream"}
                      for index, degree in enumerate(indegree) if degree)
    topology_available = not issues
    return {"observed_parent_references": parent_references,
            "resolved_edges": len(edges), "unresolved_references": sum(
                issue["kind"] == "unresolved" for issue in issues),
            "ambiguous_references": sum(issue["kind"] == "ambiguous" for issue in issues),
            "cycle_detected": cyclic, "topology_available": topology_available,
            "observed_roots": len(roots) if topology_available else None,
            "longest_observed_path_edges": max(depth, default=0) if topology_available else None,
            "edges": edges, "issues": issues}


def analyze_torchlens(store, run_id):
    """Compute descriptive statistics from saved JSON values, grouped by module."""
    record, events = store.get(run_id)
    if record.get("adapter") != "basin.torchlens.v1":
        raise ValueError("Run was not imported with the TorchLens adapter")
    overall, modules = _Stats(), {}
    for event in events:
        op = event["values"]
        overall.add(op)
        modules.setdefault(op["module"], _Stats()).add(op)
    graph = _graph(events)
    return {"run_id": run_id, "status": record["status"], "evidence_kind": record["evidence_kind"],
            "summary": {**overall.result(), "module_count": len(modules)},
            "graph": {key: value for key, value in graph.items() if key not in ("edges", "issues")},
            "modules": [{"module": name, **stats.result()} for name, stats in sorted(modules.items())],
            "edges": graph["edges"], "issues": graph["issues"],
            "limitations": record["limitations"] + [
                "Only saved activations contribute to numeric statistics; unsaved values are unknown.",
                "Resolved edges describe exported references only; graph completeness is unverified."]}
