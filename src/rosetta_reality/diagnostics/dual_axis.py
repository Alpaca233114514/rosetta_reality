"""Bounded data-only dual-axis evidence; importing never loads ML dependencies."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from pathlib import Path

SCHEMA = "rosetta.dual_axis.v1"
BOUNDARIES = (
    "reset",
    "raw_input",
    "processed_input",
    "noise",
    "model_internal",
    "model_output",
    "decoded_action",
    "projected_action",
    "executed_action",
    "physics",
)
IDENTITY_KEYS = ("input", "noise", "processor", "action_contract", "runtime")


def encoded(value):
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def relative(root, name):
    root = Path(root).resolve()
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Invalid evidence relative path")
    path = Path(name)
    target = (root / path).resolve()
    if (
        path.is_absolute()
        or ".." in path.parts
        or target == root
        or not target.is_relative_to(root)
    ):
        raise ValueError("Evidence path escapes root")
    return target


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {name}")
    return value


def make_schedule(total, *, warmup=0, epoch_boundaries=(), checkpoints=(), branches=()):
    integer(total, "total", 1)
    integer(warmup, "warmup")
    nodes = {}

    def add(step, reason):
        if 0 <= step <= total:
            nodes.setdefault(step, set()).add(reason)

    for step in (0, 1, 2, 4, 8, 16, 32, 64, *range(128, total + 1, 128), total):
        add(step, "regular")
    for reason, points in (
        ("warmup", [warmup]),
        ("epoch", epoch_boundaries),
        ("checkpoint", checkpoints),
        ("branch", branches),
    ):
        for point in points:
            integer(point, reason)
            if point > total:
                raise ValueError("Scheduled boundary exceeds training length")
            for step in range(point - 2, point + 3):
                add(step, reason)
    for step in (0, total):
        add(step, "endpoint")
    return [
        {
            "update": step,
            "reasons": sorted(reasons),
            "panel": "extended" if "endpoint" in reasons else "small",
            "all_noises": reasons != {"regular"},
        }
        for step, reasons in sorted(nodes.items())
    ]


def make_panels(train, dev, hidden, noises):
    groups = [list(group) for group in (train, dev, hidden)]
    for group in groups:
        if len(group) != len(set(group)) or any(type(x) is not int or x < 0 for x in group):
            raise ValueError("Invalid episode identities")
    a, b, c = map(set, groups)
    if len(a) < 2 or len(b) < 2 or a & b or a & c or b & c:
        raise ValueError("Probe splits overlap or lack two episodes")
    if len(noises) != 4 or len({n["name"] for n in noises}) != 4:
        raise ValueError("Four distinct registered noise conditions required")
    gaussian = [n for n in noises if n["name"] not in ("zero", "zeros")]
    if not gaussian:
        raise ValueError("Missing registered Gaussian condition")
    rows = [
        {
            "sample_id": f"{split}-ep{episode}-frame{frame}",
            "split": split,
            "episode": episode,
            "frame": frame,
        }
        for split, ids in (("train", sorted(a)), ("dev", sorted(b)))
        for episode in ids
        for frame in (0, 125, 250, 499)
    ]
    small_ids = {("train", ep) for ep in sorted(a)[:2]} | {("dev", ep) for ep in sorted(b)[:2]}
    return {
        "extended": rows,
        "small": [r for r in rows if (r["split"], r["episode"]) in small_ids],
        "noise_conditions": noises,
        "regular_noise": gaussian[0],
        "hidden": sorted(c),
    }


class Sampling:
    def __init__(self, schedule, total):
        self.nodes = {n["update"]: n for n in schedule}
        self.total = total
        self.dense_until = -1
        self.emitted = set()

    def trigger(self, update):
        self.dense_until = max(self.dense_until, min(self.total, update + 8))

    def due(self, update):
        if update in self.emitted:
            return None
        node = self.nodes.get(update)
        if node is None and update > self.dense_until:
            return None
        node = dict(
            node or {"update": update, "reasons": [], "panel": "small", "all_noises": False}
        )
        if update <= self.dense_until:
            node["reasons"] = sorted(set(node["reasons"]) | {"anomaly"})
        self.emitted.add(update)
        return node


class Window:
    """Only detached serialized observations enter the ring; never retain a graph."""

    def __init__(self, steps=16, maximum_bytes=256 * 1024**2):
        self.steps = integer(steps, "ring steps", 1)
        self.maximum_bytes = integer(maximum_bytes, "ring bytes", 1)
        self.rows = deque()
        self.bytes = 0
        self.evictions = 0

    def add(self, update, value):
        raw = encoded(value)
        integer(update, "ring update")
        if self.rows and update < self.rows[-1][0]:
            raise ValueError("Ring updates must be ordered")
        if len(raw) > self.maximum_bytes:
            self.evictions += 1
            return False
        self.rows.append((update, raw))
        self.bytes += len(raw)
        while self.rows and (
            self.bytes > self.maximum_bytes or self.rows[0][0] <= update - self.steps
        ):
            self.bytes -= len(self.rows.popleft()[1])
            self.evictions += 1
        return True

    def snapshot(self):
        return {
            "records": [json.loads(raw) for _, raw in self.rows],
            "updates": sorted({u for u, _ in self.rows}),
            "evictions": self.evictions,
            "retained_bytes": self.bytes,
            "requested_steps": self.steps,
        }


class DeviationMonitor:
    """Train-only trigger engine. Basin independently recomputes saved evidence."""

    def __init__(self):
        self.previous = {}
        self.pending = {}

    def observe(self, key, update, error, floor, *, regular, split):
        if split != "train":
            return None
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (error, floor)
        ):
            raise ValueError("Nonfinite or negative error/floor")
        reference = self.pending.get(key) or self.previous.get(key)
        result = None
        if reference:
            threshold = max(0.2 * reference["error"], 5 * max(floor, reference["floor"]))
            if error - reference["error"] > threshold:
                result = {
                    "state": "persistent" if key in self.pending else "candidate",
                    "reference_update": reference["update"],
                    "observed_update": update,
                    "threshold": threshold,
                    "heuristic_not_gate": True,
                }
                self.pending[key] = reference
            elif key in self.pending:
                result = {
                    "state": "recovered",
                    "reference_update": reference["update"],
                    "observed_update": update,
                    "heuristic_not_gate": True,
                }
                del self.pending[key]
        if regular and key not in self.pending:
            self.previous[key] = {"error": error, "update": update, "floor": floor}
        return result


def validate_event(event):
    if event.get("schema") != SCHEMA:
        raise ValueError("Unknown dual-axis schema")
    integer(event.get("sequence"), "sequence")
    coords = event.get("coordinates", {})
    if coords.get("axis") not in ("training", "probe", "execution", "snapshot", "history"):
        raise ValueError("Unknown event axis")
    for name in (
        "attempt",
        "update",
        "epoch",
        "exposures",
        "episode",
        "frame",
        "rollout_step",
        "call",
        "denoise_step",
    ):
        if coords.get(name) is not None:
            integer(coords[name], name)
    if coords.get("boundary") is not None and coords["boundary"] not in BOUNDARIES:
        raise ValueError("Unknown causal boundary")
    for name in ("values", "identity"):
        if not isinstance(event.get(name), dict):
            raise ValueError(f"Missing {name}")
    if not isinstance(event.get("reasons"), list) or event.get("status") not in (
        "observed",
        "missing",
        "failed",
    ):
        raise ValueError("Invalid event observation status")
    encoded(event)


class EvidenceWriter:
    """Create-only chunks, manifest last; an interruption remains visibly incomplete."""

    def __init__(self, root, identity, *, maximum_bytes, chunk_bytes=8 * 1024**2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.maximum = integer(maximum_bytes, "evidence budget", 4096)
        self.chunk_bytes = min(integer(chunk_bytes, "chunk bytes", 256), 32 * 1024**2)
        self.files = {}
        self.used = self.sequence = 0
        self.closed = False
        self.chunk_index = 0
        self.chunk_used = 0
        self.stream = None
        self.identity = json.loads(encoded(identity))
        self.write("identity.json", {"schema": SCHEMA, **self.identity})

    def _room(self, count):
        reserve = 2048 + sum(len(name) + 256 for name in self.files)
        if self.closed or self.used + count > self.maximum - reserve:
            raise ValueError("Evidence budget exhausted or writer closed")

    def write(self, name, value, *, raw=False):
        data = value if raw else encoded(value)
        self._room(len(data))
        path = relative(self.root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        self.used += len(data)
        spec = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        self.files[name] = spec
        return {"path": name, **spec}

    def _close_chunk(self):
        if self.stream:
            self.stream.close()
            self.files[self.chunk_name] = {
                "sha256": self.chunk_hash.hexdigest(),
                "bytes": self.chunk_used,
            }
            self.stream = None

    def event(self, coordinates, values, *, identity=None, reasons=(), status="observed"):
        row = {
            "schema": SCHEMA,
            "sequence": self.sequence,
            "coordinates": coordinates,
            "values": values,
            "identity": identity or {},
            "reasons": list(reasons),
            "status": status,
        }
        validate_event(row)
        data = encoded(row)
        if len(data) > self.chunk_bytes:
            raise ValueError("Event exceeds chunk budget; externalize tensors")
        self._room(len(data))
        if self.stream is None or self.chunk_used + len(data) > self.chunk_bytes:
            self._close_chunk()
            self.chunk_name = f"events/{self.chunk_index:06d}.jsonl"
            path = relative(self.root, self.chunk_name)
            path.parent.mkdir(exist_ok=True)
            self.stream = path.open("xb")
            self.chunk_index += 1
            self.chunk_used = 0
            self.chunk_hash = hashlib.sha256()
        self.stream.write(data)
        self.stream.flush()
        self.chunk_hash.update(data)
        self.chunk_used += len(data)
        self.used += len(data)
        self.sequence += 1
        return row

    def seal(self, status="complete", *, error=None):
        if self.closed:
            raise ValueError("Writer already sealed")
        self._close_chunk()
        manifest = {
            "schema": SCHEMA,
            "status": status,
            "error": error,
            "events": self.sequence,
            "files": self.files,
            "bytes": self.used,
        }
        data = encoded(manifest)
        if self.used + len(data) > self.maximum:
            raise ValueError("No room for evidence manifest")
        with (self.root / "manifest.json").open("xb") as stream:
            stream.write(data)
        self.closed = True
        return manifest
