"""Strict JSON and bounded, create-only local evidence storage helpers."""

import hashlib
import json
from pathlib import Path


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def loads(data):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(data, parse_constant=invalid, object_pairs_hook=unique)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def child(root, name):
    """Manifest names must remain in the designated root, including symlinks."""
    root = Path(root).resolve()
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in str(name) or ":" in str(name):
        raise ValueError(f"Unsafe relative path: {name}")
    result = (root / path).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError(f"Path escapes root: {name}")
    return result


def read_bytes(path, limit=64 * 1024 * 1024):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Evidence exceeds {limit} bytes: {Path(path).name}")
    return data


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data if isinstance(data, bytes) else data.encode("utf-8"))


def outside(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Basin output and source roots must not overlap")
