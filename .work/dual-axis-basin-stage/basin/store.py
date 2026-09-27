"""Immutable imports; source bytes are copied so history survives source moves."""

import re
from datetime import datetime, timezone
from pathlib import Path

from .io import child, dumps, loads, read_bytes, sha, write_new


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def path(self, run_id):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}", run_id):
            raise ValueError("Invalid run id")
        return child(self.root, run_id)

    def add(self, record, events, artifacts):
        run_id = record["id"]
        path = self.path(run_id)
        payloads = {"run.json": dumps(record).encode("utf-8"),
                    "events.jsonl": b"".join((dumps(e).replace("\n", "") + "\n").encode("utf-8")
                                              for e in events)}
        for name, data in artifacts.items():
            child(path, "artifacts/" + name)
            payloads["artifacts/" + name] = data
        hashes = {name: sha(data) for name, data in payloads.items()}
        if path.exists():
            existing = self.verify(run_id)
            if existing["files"] != hashes:
                raise ValueError("Run id exists with different evidence; choose a new id")
            return {"id": run_id, "status": "already_present"}
        path.mkdir(parents=True, exist_ok=False)
        for name, data in payloads.items():
            write_new(child(path, name), data)
        # A failed write leaves an unpublished directory, never a successful run.
        write_new(path / "manifest.json", dumps({"schema_version": 1, "files": hashes,
                  "imported_at": datetime.now(timezone.utc).isoformat()}))
        return {"id": run_id, "status": "imported"}

    def verify(self, run_id):
        path = self.path(run_id)
        manifest = loads(read_bytes(child(path, "manifest.json")))
        files = manifest["files"]
        if manifest.get("schema_version") != 1 or not {"run.json", "events.jsonl"} <= files.keys():
            raise ValueError("Invalid Basin manifest")
        for name, expected in files.items():
            if sha(read_bytes(child(path, name))) != expected:
                raise ValueError(f"Evidence digest mismatch: {run_id}/{name}")
        return manifest

    def get(self, run_id):
        self.verify(run_id)
        path = self.path(run_id)
        record = loads(read_bytes(child(path, "run.json")))
        events = [loads(line) for line in read_bytes(child(path, "events.jsonl")).splitlines()]
        return record, events

    def history(self):
        rows = []
        if self.root.exists():
            for path in sorted(self.root.iterdir()):
                if not path.is_dir():
                    continue
                try:
                    record, events = self.get(path.name)
                    rows.append({key: record.get(key) for key in
                                 ("id", "source_run", "adapter", "status", "evidence_kind", "gate")}
                                | {"events": record.get("event_count", len(events)), "integrity": "verified"})
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    rows.append({"id": path.name, "integrity": "invalid", "error": str(exc)})
        return rows
