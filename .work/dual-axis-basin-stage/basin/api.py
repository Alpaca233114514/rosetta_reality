"""Model-facing tool API shared by the CLI and MCP stdio server."""

import math
import base64
import hashlib
from pathlib import Path

from .adapters import import_gate, import_native, import_trace
from .analysis import analyze, compare, pointer
from .identity import check_identity
from .io import child, dumps, loads, outside, read_bytes
from .store import Store
from .torchlens_data import analyze_torchlens, import_torchlens
from .dual_axis import branch_compare, import_dual_axis, locate_deviation, timeline


class ToolError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


STRING = {"type": "string"}
PAGE = {"offset": {"type": "integer", "minimum": 0, "default": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}}


def tool(name, description, properties, required=(), write=False):
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties,
                            "required": list(required), "additionalProperties": False},
            "annotations": {"readOnlyHint": not write, "destructiveHint": False,
                            "idempotentHint": True, "openWorldHint": False}}


READ_TOOLS = [
    tool("basin_timeline", "Read dual-axis events and sampling coverage; update and rollout_step remain separate.",
         dict(PAGE, run_id=STRING, axis=STRING, field=STRING, start={"type": "integer", "minimum": 0},
              end={"type": "integer", "minimum": 0}), ("run_id",)),
    tool("basin_locate_deviation", "Locate observed regression intervals, absolute failures, recovery and evidence gaps; no causal claim.",
         dict(PAGE, run_id=STRING, axis={"type": "string", "enum": ["probe", "execution", "training"]}, metric=STRING,
              section={"type": "string", "enum": ["deviations", "absolute_failures", "gaps", "trends"]}), ("run_id",)),
    tool("basin_branch_compare", "Compare explicitly aligned branch evidence and identities; research ancestry is not weight ancestry.",
         dict(PAGE, left=STRING, right=STRING), ("left", "right")),
    tool("basin_history", "List recorded run units and integrity. Filter by source_run or evidence_kind; paginated.",
         dict(PAGE, source_run=STRING, evidence_kind=STRING)),
    tool("basin_get_run", "Read run metadata. For parameters use JSON Pointer, e.g. /parameters/options. Default omits large parameters.",
         {"run_id": STRING, "pointer": STRING}, ("run_id",)),
    tool("basin_events", "Query recorded events by step/stage with original evidence pointers. Field selects within values, e.g. /action.",
         dict(PAGE, run_id=STRING, step={"type": "integer", "minimum": 0}, stage=STRING, field=STRING), ("run_id",)),
    tool("basin_read_artifact", "Read a verified original artifact. JSONL requires a 1-based line. Optional JSON Pointer narrows data.",
         {"run_id": STRING, "name": STRING, "line": {"type": "integer", "minimum": 1}, "pointer": STRING,
          "source": STRING, "byte_offset": {"type": "integer", "minimum": 0},
          "byte_limit": {"type": "integer", "minimum": 1, "maximum": 4096}},
         ("run_id", "name")),
    tool("basin_analyze", "Compute observed statistics, facts and limitations. Statistics are paginated; prefix filters keys such as execution/action/.",
         dict(PAGE, run_id=STRING, prefix=STRING), ("run_id",)),
    tool("basin_analyze_torchlens", "Compute saved-activation statistics and observed graph relationships. Page modules, edges or graph issues.",
         dict(PAGE, run_id=STRING, section={"type": "string", "enum": ["modules", "edges", "issues"]}),
         ("run_id",)),
    tool("basin_compare", "Compare two runs at matching step/stage; report missing coverage and first observed difference, never a causal claim.",
         {"left": STRING, "right": STRING, "field": STRING,
          "atol": {"type": "number", "minimum": 0, "default": 0}}, ("left", "right")),
    tool("basin_verify", "Verify all stored file hashes for one run; does not certify measurement truth or Gate acceptance.",
         {"run_id": STRING}, ("run_id",)),
    tool("basin_check_identity", "Compare a verified stored identity claim with an expected value and/or SHA256 of a file under a configured source alias. Missing claims fail closed.",
         {"run_id": STRING, "pointer": STRING, "expected": STRING,
          "source": STRING, "path": STRING}, ("run_id", "pointer")),
]

IMPORT_TOOLS = [
    tool("basin_import_dual_axis", "Import a sealed diagnostic bundle from an operator-configured source; stream-hash external tensors without deserialization.",
         {"source": STRING, "path": STRING, "run_id": STRING}, ("source", "path", "run_id"), write=True),
    tool("basin_import_torchlens", "Import an existing Basin TorchLens JSON export from a configured source alias. No Python object deserialization or model execution.",
         {"source": STRING, "path": STRING, "run_id": STRING}, ("source", "path", "run_id"), write=True),
    tool("basin_import_native", "Copy an existing collector JSON from a configured source alias into Basin. Never overwrites evidence.",
         {"source": STRING, "path": STRING, "run_id": STRING}, ("source", "path", "run_id"), write=True),
    tool("basin_import_rosetta", "Read existing Rosetta trace or Gate report from a configured source alias; writes only Basin history. Does not execute Rosetta.",
         {"source": STRING, "path": STRING, "run_id": STRING,
          "kind": {"type": "string", "enum": ["trace", "gate-report"], "default": "trace"},
          "gate": {"type": "string", "enum": ["diagnostic", "gate3", "gate4"], "default": "diagnostic"}},
         ("source", "path", "run_id"), write=True),
]


def validate(schema, arguments):
    """Validate the primitive object schemas published by this tool catalog."""
    if not isinstance(arguments, dict):
        raise ToolError("invalid_arguments", "Arguments must be an object")
    properties = schema["properties"]
    if set(arguments) - properties.keys() or set(schema["required"]) - arguments.keys():
        raise ToolError("invalid_arguments", "Unknown arguments or missing required arguments")
    for name, value in arguments.items():
        spec = properties[name]
        kind = spec["type"]
        valid = ((kind == "string" and isinstance(value, str))
                 or (kind == "integer" and type(value) is int)
                 or (kind == "number" and type(value) in (int, float)))
        if not valid:
            raise ToolError("invalid_arguments", f"{name} must be {kind}")
        if kind in ("integer", "number"):
            if not math.isfinite(value) or value < spec.get("minimum", -math.inf) or value > spec.get("maximum", math.inf):
                raise ToolError("invalid_arguments", f"{name} is outside its allowed range")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError("invalid_arguments", f"Unsupported {name}")


def page(rows, args):
    offset, limit = args.get("offset", 0), args.get("limit", 20)
    end = offset + limit
    return {"items": rows[offset:end], "total": len(rows), "offset": offset,
            "next_offset": end if end < len(rows) else None}


class BasinAPI:
    def __init__(self, store, sources=None, max_result_bytes=128 * 1024):
        self.store = Store(store)
        self.sources = {name: Path(path).resolve(strict=True) for name, path in (sources or {}).items()}
        for source in self.sources.values():
            if not source.is_dir():
                raise ValueError("Source must be a directory")
            outside(self.store.root, source)
        self.max_result_bytes = max_result_bytes

    def tools(self):
        return READ_TOOLS + (IMPORT_TOOLS if self.sources else [])

    def call(self, name, arguments=None):
        specs = {item["name"]: item for item in self.tools()}
        if name not in specs:
            raise ToolError("unknown_tool", f"Unknown or disabled tool: {name}")
        args = {} if arguments is None else arguments
        validate(specs[name]["inputSchema"], args)
        try:
            data = self._dispatch(name, args)
            result = {"schema_version": 1, "ok": True, "tool": name, "data": data}
            if len(dumps(result).encode("utf-8")) > self.max_result_bytes:
                raise ToolError("result_too_large", "Use a narrower pointer/field/prefix or a smaller limit")
            return result
        except ToolError:
            raise
        except (ValueError, OSError, KeyError, TypeError, IndexError, OverflowError) as exc:
            raise ToolError("evidence_error", str(exc)) from exc

    def _dispatch(self, name, args):
        run_id = args.get("run_id")
        if name == "basin_timeline":
            result = timeline(self.store, run_id, **{k: args[k] for k in ("axis", "start", "end", "offset", "limit") if k in args})
            if "field" in args:
                for event in result["events"]["items"]:
                    event["values"] = pointer(event["values"], args["field"])
            return result
        if name == "basin_locate_deviation":
            result = locate_deviation(self.store, run_id, axis=args.get("axis", "probe"), metric=args.get("metric"))
            section = args.get("section", "deviations")
            rows = result[section]
            for key in ("deviations", "absolute_failures", "gaps", "trends"):
                result[key + "_count"] = len(result.pop(key))
            result[section] = page(rows, args)
            return result
        if name == "basin_branch_compare":
            result = branch_compare(self.store, args["left"], args["right"])
            result["pairs"] = page(result["pairs"], args)
            result["missing"] = page(result["missing"], args)
            return result
        if name == "basin_history":
            rows = self.store.history()
            for key in ("source_run", "evidence_kind"):
                if key in args:
                    rows = [row for row in rows if row.get(key) == args[key]]
            return page(rows, args)
        if name == "basin_get_run":
            record = self.store.get(run_id)[0]
            return pointer(record, args["pointer"]) if "pointer" in args else {
                k: v for k, v in record.items() if k != "parameters"}
        if name == "basin_events":
            events = self.store.get(run_id)[1]
            selected = [e for e in events if ("step" not in args or e["step"] == args["step"])
                        and ("stage" not in args or e["stage"] == args["stage"])]
            result = page(selected, args)
            if "field" in args:
                narrowed = []
                for event in result["items"]:
                    row = {key: event[key] for key in ("step", "stage", "evidence")}
                    try:
                        row.update(value=pointer(event["values"], args["field"]), present=True)
                    except (KeyError, IndexError, TypeError):
                        row.update(value=None, present=False)
                    narrowed.append(row)
                result["items"] = narrowed
            return result
        if name == "basin_read_artifact":
            if "source" in args:
                if "line" in args or "pointer" in args:
                    raise ValueError("External binary reads use byte ranges only")
                record = self.store.get(run_id)[0]
                spec = record.get("external_artifacts", {}).get(args["name"])
                if spec is None or args["source"] not in self.sources:
                    raise ValueError("External artifact or source alias is not registered")
                path = child(child(self.sources[args["source"]], record["source_path"]), args["name"])
                offset, limit = args.get("byte_offset", 0), args.get("byte_limit", 4096)
                h, count, selected = hashlib.sha256(), 0, bytearray()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        end = count + len(block)
                        if end > offset and count < offset + limit:
                            selected.extend(block[max(0, offset-count):min(len(block), offset+limit-count)])
                        h.update(block)
                        count = end
                        if count > 4 * 1024**3:
                            raise ValueError("External artifact exceeds streaming bound")
                if count != spec["bytes"] or h.hexdigest() != spec["sha256"]:
                    raise ValueError("External artifact bytes changed")
                if offset >= count:
                    raise ValueError("Byte offset outside artifact")
                return {"encoding": "base64_raw_bytes", "data": base64.b64encode(selected).decode(),
                        "byte_offset": offset, "returned_bytes": len(selected), "total_bytes": count,
                        "sha256": h.hexdigest(), "verification": "source_bytes_hashed"}
            if "byte_offset" in args or "byte_limit" in args:
                raise ValueError("Binary ranges require a configured source")
            manifest = self.store.verify(run_id)
            name = "artifacts/" + args["name"]
            if name not in manifest["files"]:
                raise ValueError("Artifact not in manifest")
            if args["name"].endswith(".jsonl") and "line" not in args:
                raise ToolError("invalid_arguments", "JSONL artifacts require a 1-based line")
            raw = read_bytes(child(self.store.path(run_id), name))
            if "line" in args:
                raw = raw.splitlines()[args["line"] - 1]
            return {"value": pointer(loads(raw), args.get("pointer", "")),
                    "evidence": {"run_id": run_id, "artifact": args["name"],
                                 "line": args.get("line"), "pointer": args.get("pointer", ""),
                                 "sha256": manifest["files"][name]}}
        if name == "basin_analyze":
            result = analyze(self.store, run_id)
            stats = result.pop("statistics")
            result["statistics"] = page([{"field": key, **value} for key, value in sorted(stats.items())
                                         if key.startswith(args.get("prefix", ""))], args)
            return result
        if name == "basin_analyze_torchlens":
            result = analyze_torchlens(self.store, run_id)
            section = args.get("section", "modules")
            rows = result.pop(section)
            for other in ("modules", "edges", "issues"):
                result.pop(other, None)
            result["section"] = section
            result[section] = page(rows, args)
            return result
        if name == "basin_compare":
            return compare(self.store, args["left"], args["right"], args.get("field", "/action"), args.get("atol", 0.0))
        if name == "basin_verify":
            manifest = self.store.verify(run_id)
            return {"run_id": run_id, "integrity": "verified", "files": manifest["files"]}
        if name == "basin_check_identity":
            return check_identity(self.store, self.sources, run_id, args["pointer"],
                                  expected=args.get("expected"), source=args.get("source"),
                                  path=args.get("path"))
        if args["source"] not in self.sources:
            raise ToolError("invalid_arguments", "Source alias was not configured by the operator")
        source = self.sources[args["source"]]
        if name == "basin_import_dual_axis":
            return import_dual_axis(self.store, source, args["path"], run_id)
        if name == "basin_import_torchlens":
            return import_torchlens(self.store, child(source, args["path"]), run_id)
        if name == "basin_import_native":
            return import_native(self.store, child(source, args["path"]), run_id)
        if args.get("kind", "trace") == "gate-report":
            return import_gate(self.store, source, args["path"], run_id)
        return import_trace(self.store, source, args["path"], run_id, gate=args.get("gate", "diagnostic"))


def error_result(name, exc):
    return {"schema_version": 1, "ok": False, "tool": name,
            "error": {"code": exc.code, "message": str(exc)}}


def parse_sources(values):
    result = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path or name in result:
            raise ValueError("Use unique --source alias=directory entries")
        result[name] = path
    return result
