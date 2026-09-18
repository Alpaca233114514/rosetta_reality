"""Create-only, pickle-free tensor trees for bounded diagnostic evidence."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def relative(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Repository-relative POSIX path required")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Path escapes evidence root")
    result = Path(root) / path
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Resolved path escapes evidence root")
    return result


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")


def seal(directory, result):
    directory = Path(directory)
    write_json(directory / "result.json", result)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Evidence links are forbidden")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = {
                "sha256": sha(path),
                "bytes": path.stat().st_size,
            }
    write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "status": result["status"],
            "files": files,
        },
    )


def write_tree(directory, value, *, budget_root, maximum_bytes, reserve_bytes=1024 * 1024):
    """Persist exact tensor bytes, including BF16, with an explicit JSON type tree."""
    import numpy as np
    import torch

    arrays = {}

    def encode(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous().clone()
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("Nonfinite evidence tensor")
            name = "tensor_" + str(len(arrays))
            arrays[name] = tensor.reshape(-1).view(torch.uint8).numpy()
            return {
                "type": "tensor",
                "array": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("Only string tensor-tree keys are supported")
            return {"type": "dict", "items": {key: encode(v) for key, v in item.items()}}
        if isinstance(item, (tuple, list)):
            return {
                "type": "tuple" if isinstance(item, tuple) else "list",
                "items": [encode(v) for v in item],
            }
        if item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Nonfinite evidence scalar")
            return {"type": "scalar", "value": item}
        raise TypeError("Unsupported evidence value: " + type(item).__name__)

    tree = encode(value)
    metadata = (json.dumps(tree, allow_nan=False, sort_keys=True) + "\n").encode()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    content = buffer.getvalue()
    used = sum(p.stat().st_size for p in Path(budget_root).rglob("*") if p.is_file())
    if used + len(metadata) + len(content) + reserve_bytes > maximum_bytes:
        raise RuntimeError("Evidence byte budget exhausted")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "tree.json").open("xb") as stream:
        stream.write(metadata)
    with (directory / "arrays.npz").open("xb") as stream:
        stream.write(content)


def read_tree(directory, *, device="cpu"):
    import numpy as np
    import torch

    directory = Path(directory)
    with np.load(directory / "arrays.npz", allow_pickle=False) as archive:

        def decode(node):
            kind = node["type"]
            if kind == "tensor":
                dtype = getattr(torch, node["dtype"].removeprefix("torch."), None)
                if not isinstance(dtype, torch.dtype):
                    raise ValueError("Unknown tensor dtype")
                raw = archive[node["array"]]
                if raw.dtype != np.uint8 or raw.ndim != 1:
                    raise ValueError("Invalid byte tensor")
                return torch.from_numpy(raw.copy()).view(dtype).reshape(node["shape"]).to(device)
            if kind == "dict":
                return {key: decode(v) for key, v in node["items"].items()}
            if kind in ("list", "tuple"):
                values = [decode(v) for v in node["items"]]
                return tuple(values) if kind == "tuple" else values
            if kind == "scalar":
                return node["value"]
            raise ValueError("Unknown tensor-tree node")

        return decode(load_json(directory / "tree.json"))
