"""Data-only dual-axis import and deterministic localization; never infer causality."""

import hashlib
import math
import shutil

from .analysis import differences
from .io import child, dumps, loads, outside, read_bytes, sha, write_new

SCHEMA = "rosetta.dual_axis.v1"
IDENTITIES = ("input", "noise", "processor", "action_contract", "runtime")
BOUNDARIES = ("reset", "raw_input", "processed_input", "noise", "model_internal",
              "model_output", "decoded_action", "projected_action", "executed_action", "physics")


def validate_event(row, sequence):
    if row.get("schema") != SCHEMA or type(row.get("sequence")) is not int or row["sequence"] != sequence:
        raise ValueError("Non-contiguous diagnostic sequence")
    c = row.get("coordinates", {})
    if c.get("axis") not in ("training", "probe", "execution", "snapshot", "history"):
        raise ValueError("Invalid diagnostic axis")
    for key in ("attempt", "update", "epoch", "exposures", "episode", "frame", "rollout_step", "call", "denoise_step"):
        value = c.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Invalid diagnostic coordinate")
    if c.get("boundary") is not None and c["boundary"] not in BOUNDARIES:
        raise ValueError("Invalid boundary")
    if not isinstance(row.get("values"), dict) or not isinstance(row.get("identity"), dict):
        raise ValueError("Invalid diagnostic payload")
    if row.get("status") not in ("observed", "missing", "failed") or not isinstance(row.get("reasons"), list):
        raise ValueError("Invalid diagnostic status")
    return row


def import_dual_axis(store, source, relative, run_id):
    outside(store.root, source)
    root = child(source, relative)
    raw_manifest = read_bytes(child(root, "manifest.json"))
    manifest = loads(raw_manifest)
    if manifest.get("schema") != SCHEMA or manifest.get("status") not in ("complete", "incomplete"):
        raise ValueError("Unpublished or invalid diagnostic bundle")
    artifacts = {"manifest.json": raw_manifest}
    events = 0
    summary_files = {}
    # Small summary bundles only. Large binary payloads remain in a registered source.
    # Every referenced payload is streamed and verified at import, not deserialized.
    for name, spec in sorted(manifest["files"].items()):
        path = child(root, name)
        h, count = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                count += len(block)
                if count > 4 * 1024**3:
                    raise ValueError("Source artifact exceeds 4 GiB")
                h.update(block)
        if count != spec["bytes"] or h.hexdigest() != spec["sha256"]:
            raise ValueError("Diagnostic source hash/size mismatch")
        if name == "identity.json" or name.startswith(("events/", "sources/")):
            raw = read_bytes(path, 32 * 1024**2)
            if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
                raise ValueError("Summary changed while importing")
            summary_files[name] = path
            if name == "identity.json" or name.startswith("sources/"):
                artifacts[name] = raw
            else:
                for line in raw.splitlines():
                    validate_event(loads(line), events)
                    events += 1
    identity = loads(artifacts["identity.json"])
    if identity.get("schema") != SCHEMA or not isinstance(identity.get("source_run"), str):
        raise ValueError("Missing diagnostic run identity")
    if events != manifest["events"]:
        raise ValueError("Diagnostic event count mismatch")
    record = {"schema_version": 1, "id": run_id, "adapter": SCHEMA,
              "source_run": identity["source_run"], "source_path": relative,
              "evidence_kind": identity.get("evidence_kind", "historical_observation"),
              "status": manifest["status"], "parameters": identity, "dimensions": [],
              "gate": {"acceptance": "not_inferred"},
              "event_shards": sorted(k for k in summary_files if k.startswith("events/")),
              "event_count": events,
              "external_artifacts": {k: v for k, v in manifest["files"].items() if k not in summary_files},
              "verification": {"source_bytes": "hashed_at_import", "external_current_bytes": "not_rechecked"},
              "limitations": ["Summary imports are derived evidence, not new model captures.",
                              "External tensor availability is not guaranteed by history verification."]}
    destination = store.path(run_id)
    payloads = {"run.json": dumps(record).encode(), "events.jsonl": b"",
                **{"artifacts/" + k: v for k, v in artifacts.items()}}
    hashes = {k: sha(v) for k, v in payloads.items()}
    hashes.update({"artifacts/" + k: manifest["files"][k]["sha256"] for k in record["event_shards"]})
    if destination.exists():
        if store.verify(run_id)["files"] != hashes:
            raise ValueError("Run id exists with different evidence")
        return {"id": run_id, "status": "already_present"}
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        write_new(child(destination, name), raw)
    for name in record["event_shards"]:
        target = child(destination, "artifacts/" + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with summary_files[name].open("rb") as source_stream, target.open("xb") as out:
            shutil.copyfileobj(source_stream, out, 1024 * 1024)
        if sha(read_bytes(target, 32 * 1024**2)) != manifest["files"][name]["sha256"]:
            raise ValueError("Source shard changed while copying")
    write_new(destination / "manifest.json", dumps({"schema_version": 1, "files": hashes}))
    return {"id": run_id, "status": "imported"}


def read(store, run_id):
    manifest = store.verify(run_id)
    record_raw = read_bytes(child(store.path(run_id), "run.json"))
    if sha(record_raw) != manifest["files"]["run.json"]:
        raise ValueError("Run metadata changed while reading")
    record = loads(record_raw)
    if record["adapter"] != SCHEMA:
        raise ValueError("This operation requires a dual-axis diagnostic bundle")
    def iterate():
        if "event_shards" not in record:
            events = store.get(run_id)[1]
            for e in events:
                yield {**e["values"], "evidence": e["evidence"]}
            return
        sequence = 0
        for name in record["event_shards"]:
            data = read_bytes(child(store.path(run_id), "artifacts/" + name), 32 * 1024**2)
            if sha(data) != manifest["files"]["artifacts/" + name]:
                raise ValueError("Stored shard changed during read")
            for line, raw in enumerate(data.splitlines(), 1):
                row = validate_event(loads(raw), sequence)
                sequence += 1
                yield {**row, "evidence": {"artifact": name, "line": line,
                                           "sha256": manifest["files"]["artifacts/" + name]}}
    return record, iterate()


def timeline(store, run_id, *, axis=None, start=None, end=None, offset=0, limit=20):
    record, rows = read(store, run_id)
    selected, observed, matched = [], set(), 0
    for row in rows:
        c = row["coordinates"]
        if c["axis"] == "probe" and c.get("update") is not None:
            observed.add(c["update"])
        coordinate = c.get("rollout_step") if c["axis"] == "execution" else c.get("update")
        if axis and c["axis"] != axis:
            continue
        if start is not None and (coordinate is None or coordinate < start):
            continue
        if end is not None and (coordinate is None or coordinate > end):
            continue
        if offset <= matched < offset + limit:
            selected.append(row)
        matched += 1
    scheduled = [n["update"] for n in record["parameters"].get("schedule", [])]
    return {"run_id": run_id, "status": record["status"],
            "events": {"items": selected, "total": matched, "offset": offset,
                       "next_offset": offset + limit if offset + limit < matched else None},
            "coverage": {"scheduled_probe_updates": scheduled, "observed_probe_updates": sorted(observed),
                         "missing_scheduled_updates": sorted(set(scheduled) - set(observed)),
                         "complete_sample_coverage": "inspect per-sample events and missing records"},
            "step_semantics": "update for training/probe; rollout_step for execution"}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _compatible(left, right):
    return [key for key in IDENTITIES if not left.get(key) or left.get(key) != right.get(key)]


def locate_deviation(store, run_id, *, axis="probe", metric=None):
    record, rows = read(store, run_id)
    tracks, gaps, absolute = {}, [], []
    coordinate_name = "rollout_step" if axis == "execution" else "update"
    for row in rows:
        c = row["coordinates"]
        if c["axis"] != axis:
            continue
        x = c.get(coordinate_name)
        if x is None or row["status"] != "observed":
            gaps.append({"coordinates": c, "reason": "missing_observation", "evidence": row["evidence"]})
            continue
        for m in row["values"].get("metrics", []):
            if metric and m.get("name") != metric:
                continue
            if not _finite(m.get("value")) or m["value"] < 0:
                raise ValueError("Error metrics must be finite and nonnegative")
            key = (c.get("sample_id"), c.get("episode"), c.get("noise_id"),
                   c.get("boundary"), m["name"], m.get("group"), m.get("window"))
            item = {"coordinate": x, "coordinates": c, "metric": m,
                    "identity": row["identity"], "regular": "regular" in row["reasons"],
                    "evidence": row["evidence"]}
            tracks.setdefault(key, []).append(item)
            limit = m.get("absolute_limit")
            if limit is not None:
                if not _finite(limit) or limit < 0 or not m.get("threshold_identity"):
                    raise ValueError("Absolute thresholds need a sealed identity")
                if m["value"] > limit:
                    absolute.append(item)
    episodes, trends = [], []
    compatible_pairs = 0
    for key, points in tracks.items():
        points.sort(key=lambda p: p["coordinate"])
        if len({p["coordinate"] for p in points}) != len(points):
            raise ValueError("Ambiguous duplicate metric coordinates")
        if len(points) >= 2 and not _compatible(points[0]["identity"], points[-1]["identity"]):
            trends.append({"coordinates": points[-1]["coordinates"], "metric": points[-1]["metric"]["name"],
                           "first_value": points[0]["metric"]["value"], "last_value": points[-1]["metric"]["value"],
                           "change": points[-1]["metric"]["value"] - points[0]["metric"]["value"],
                           "meaning": "descriptive cumulative change, not an adjacent-point alarm"})
        previous = pending = None
        for point in points:
            m = point["metric"]
            reference = pending["reference"] if pending else previous
            floor = m.get("aa_floor")
            if m.get("category") != "behavior":
                continue
            if reference:
                mismatch = _compatible(reference["identity"], point["identity"])
                ref_floor = reference["metric"].get("aa_floor")
                if mismatch or not all(_finite(f) and f >= 0 for f in (floor, ref_floor)):
                    gaps.append({"coordinates": point["coordinates"],
                                 "reason": "identity_or_AA_floor_missing", "identity_fields": mismatch,
                                 "evidence": point["evidence"]})
                    pending = previous = None
                else:
                    compatible_pairs += 1
                    threshold = max(0.2 * reference["metric"]["value"], 5 * max(floor, ref_floor))
                    if m["value"] - reference["metric"]["value"] > threshold:
                        if pending is None:
                            pending = {"reference": reference, "first_observed": point,
                                       "possible_start_interval": [reference["coordinate"], point["coordinate"]],
                                       "left_endpoint_excluded": True, "state": "candidate",
                                       "threshold": threshold, "confirmation": None, "recovery": None}
                            episodes.append(pending)
                        else:
                            pending["state"] = "persistent"
                            if pending["confirmation"] is None:
                                pending["confirmation"] = point
                    elif pending:
                        pending["recovery"] = point
                        pending["state"] = "recovered_after_persistent" if pending["confirmation"] else "transient"
                        pending = None
            if point["regular"] and pending is None:
                previous = point
    rank = lambda p: (p["coordinate"], BOUNDARIES.index(p["coordinates"]["boundary"])
                      if p["coordinates"].get("boundary") in BOUNDARIES else len(BOUNDARIES))
    episodes.sort(key=lambda e: rank(e["first_observed"]))
    absolute.sort(key=rank)
    return {"run_id": run_id, "axis": axis, "deviations": episodes,
            "first_observed": episodes[0]["first_observed"] if episodes else None,
            "absolute_failures": absolute, "gaps": gaps, "trends": trends,
            "result": "observed_deviation" if episodes else "insufficient_evidence" if gaps or not compatible_pairs else "no_observed_deviation",
            "causal_conclusion": None,
            "limitations": ["20% / 5x AA-floor is a diagnostic heuristic, not significance or Gate acceptance.",
                            "Earliest observation bounds sampled evidence only; unsampled regressions may recover.",
                            "Absolute failure can precede training; base is not a correct-action oracle."]}


def branch_compare(store, left_id, right_id):
    left, a = read(store, left_id)
    right, b = read(store, right_id)
    def key(row):
        c = row["coordinates"]
        return tuple(c.get(k) for k in ("axis", "update", "sample_id", "noise_id", "episode", "rollout_step", "boundary", "module", "call", "denoise_step"))
    maps = []
    for rows in (a, b):
        mapping = {}
        for row in rows:
            if row["coordinates"]["axis"] not in ("probe", "execution"):
                continue
            # Compare bounded scalar summaries and payload identities, never materialize binary tensors.
            if key(row) in mapping:
                raise ValueError("Ambiguous branch coordinates")
            mapping[key(row)] = row
        maps.append(mapping)
    paired, missing, first_boundary_mismatch = [], [], None
    def order(k):
        axis, update, sample, noise, episode, step, boundary, module, call, denoise = k
        return (update if update is not None else -1, step if step is not None else -1,
                BOUNDARIES.index(boundary) if boundary in BOUNDARIES else len(BOUNDARIES),
                str(axis), str(sample), str(noise), episode if episode is not None else -1,
                str(module), call if call is not None else -1, denoise if denoise is not None else -1)
    for k in sorted(maps[0].keys() | maps[1].keys(), key=order):
        x, y = maps[0].get(k), maps[1].get(k)
        if x is None or y is None:
            missing.append({"coordinates": (x or y)["coordinates"], "reason": "unmatched_event"})
            continue
        mismatch = _compatible(x["identity"], y["identity"])
        raw_differences = list(differences(x["values"], y["values"]))
        if raw_differences and first_boundary_mismatch is None:
            first_boundary_mismatch = {"coordinates": x["coordinates"],
                                       "identity_incompatibilities": mismatch,
                                       "left_evidence": x["evidence"], "right_evidence": y["evidence"],
                                       "meaning": "first recorded evidence difference, not causal attribution"}
        paired.append({"coordinates": x["coordinates"], "comparable": not mismatch,
                       "incompatibilities": mismatch,
                       "differences": [] if mismatch else raw_differences[:50],
                       "left_evidence": x["evidence"], "right_evidence": y["evidence"]})
    return {"left": left_id, "right": right_id, "pairs": paired, "missing": missing,
            "first_boundary_mismatch": first_boundary_mismatch,
            "branch_left": left["parameters"].get("lineage"), "branch_right": right["parameters"].get("lineage"),
            "parameter_differences": list(differences(left["parameters"], right["parameters"]))[:100],
            "causal_conclusion": None, "limitations": ["Research ancestry is not weight ancestry.",
                "Matching update counts do not establish equal exposure budgets or matched physical states."]}
