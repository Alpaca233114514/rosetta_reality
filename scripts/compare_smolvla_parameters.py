"""Read-only, bounded checkpoint tensor algebra. Never constructs a policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import ExitStack
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def group(name):
    if ".lm_expert.layers." in name:
        prefix, rest = name.split(".lm_expert.layers.", 1)
        layer, module = rest.split(".", 1)
        return f"expert.layer.{layer}." + module.rsplit(".", 1)[0]
    return name.rsplit(".", 1)[0]


def tensor_stats(left, right, block_size=262144):
    """Exact payload hashes; chunked float64 arithmetic without full double copies."""
    import numpy as np
    import torch

    if left.shape != right.shape:
        raise ValueError("Tensor shape mismatch")
    if not left.is_floating_point() or not right.is_floating_point():
        raise ValueError("Nonfloating tensor requires a separately registered comparison")
    if left.numel() == 0 or block_size < 1:
        raise ValueError("Nonempty tensors and positive block size required")
    a, b = left.reshape(-1), right.reshape(-1)
    hashes = [hashlib.sha256(), hashlib.sha256()]
    sums, squares = [0.0, 0.0], [0.0, 0.0]
    zeros = [0, 0]
    delta_square = dot = maximum = 0.0
    changed = 0
    for start in range(0, a.numel(), block_size):
        parts = [v[start : start + block_size] for v in (a, b)]
        arrays = []
        for i, part in enumerate(parts):
            hashes[i].update(part.contiguous().view(torch.uint8).numpy().tobytes())
            array = part.double().numpy()
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite tensor")
            arrays.append(array)
            sums[i] += float(array.sum())
            squares[i] += float(np.dot(array, array))
            zeros[i] += int(np.count_nonzero(array == 0))
        x, y = arrays
        delta = y - x
        delta_square += float(np.dot(delta, delta))
        dot += float(np.dot(x, y))
        maximum = max(maximum, float(np.max(np.abs(delta))))
        changed += int(np.count_nonzero(x != y))
    count = a.numel()
    norms = [math.sqrt(v) for v in squares]
    payload_hashes = [h.hexdigest() for h in hashes]
    stats = []
    for i in range(2):
        mean = sums[i] / count
        stats.append({"mean": mean, "std_population": math.sqrt(
            max(0.0, squares[i] / count - mean * mean)),
            "rms": math.sqrt(squares[i] / count), "l2": norms[i],
            "zero_fraction": zeros[i] / count, "payload_sha256": payload_hashes[i]})
    return {
        "shape": list(left.shape), "numel": count,
        "dtype": [str(left.dtype), str(right.dtype)], "finite": True,
        "left": stats[0], "right": stats[1],
        "byte_equal": left.dtype == right.dtype and payload_hashes[0] == payload_hashes[1],
        "numeric_equal": changed == 0, "changed_elements": changed,
        "changed_fraction": changed / count, "delta_l2": math.sqrt(delta_square),
        "delta_rms": math.sqrt(delta_square / count), "delta_max_abs": maximum,
        "relative_l2": math.sqrt(delta_square) / norms[0] if norms[0] else None,
        "cosine": min(1.0, max(-1.0, dot / (norms[0] * norms[1])))
        if norms[0] and norms[1] else None,
    }


def compare_files(left, right):
    from safetensors import safe_open

    with ExitStack() as stack:
        a = stack.enter_context(safe_open(left, framework="pt", device="cpu"))
        b = stack.enter_context(safe_open(right, framework="pt", device="cpu"))
        if set(a.keys()) != set(b.keys()):
            raise ValueError(json.dumps({"missing_right": sorted(set(a.keys()) - set(b.keys())),
                                         "extra_right": sorted(set(b.keys()) - set(a.keys()))}))
        for name in sorted(a.keys()):
            try:
                row = tensor_stats(a.get_tensor(name), b.get_tensor(name))
            except ValueError as error:
                raise ValueError(f"{name}: {error}") from error
            yield {"name": name, "group": group(name), **row}


def summarize(rows):
    result = {}
    for row in rows:
        bucket = result.setdefault(row["group"], {
            "tensors": 0, "numel": 0, "byte_changed_tensors": 0,
            "numeric_changed_tensors": 0, "changed_elements": 0,
            "left_l2_squared": 0.0, "delta_l2_squared": 0.0})
        bucket["tensors"] += 1
        bucket["numel"] += row["numel"]
        bucket["byte_changed_tensors"] += int(not row["byte_equal"])
        bucket["numeric_changed_tensors"] += int(not row["numeric_equal"])
        bucket["changed_elements"] += row["changed_elements"]
        bucket["left_l2_squared"] += row["left"]["l2"] ** 2
        bucket["delta_l2_squared"] += row["delta_l2"] ** 2
    for bucket in result.values():
        base = bucket["left_l2_squared"]
        bucket["relative_l2"] = math.sqrt(bucket["delta_l2_squared"] / base) if base else None
    return result


def run(plan_path, output):
    from scripts.root_analysis_resources import verify_resource_envelope

    plan = json.loads(plan_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if plan["scope"] != "saved_weight_algebra_only" or plan["optimizer_steps"] != 0:
            raise ValueError("Only zero-update saved weight analysis is registered")
        if os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["image"]:
            raise ValueError("Pinned container identity required")
        resources = verify_resource_envelope(plan)
        if Path('/sys/fs/cgroup/memory.swap.max').read_text().strip() != '0':
            raise ValueError("Additional swap is forbidden")
        for path, expected in plan["files"].items():
            if sha(path) != expected:
                raise ValueError(f"Input/source hash mismatch: {path}")
        summaries = {}
        for left, right in plan["pairs"]:
            rows = []
            pair = f"{left}-to-{right}"
            with (output / f"{pair}.jsonl").open("x") as stream:
                for row in compare_files(plan["models"][left], plan["models"][right]):
                    if time.monotonic() - started >= plan["resources"]["wall_seconds"]:
                        raise ValueError("Registered deadline exceeded")
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
            summaries[pair] = {
                "tensors": len(rows), "numel": sum(r["numel"] for r in rows),
                "byte_equal": sum(r["byte_equal"] for r in rows),
                "numeric_equal": sum(r["numeric_equal"] for r in rows),
                "groups": summarize(rows), "rows_sha256": sha(output / f"{pair}.jsonl")}
        for path, expected in plan["files"].items():
            if sha(path) != expected:
                raise ValueError(f"Post-run input/source drift: {path}")
        write_json(output / "result.json", {
            "id": plan["id"], "status": "complete", "plan_sha256": sha(plan_path),
            "resources": resources, "pairs": summaries, "seconds": time.monotonic() - started,
            "policy_constructed": False, "optimizer_steps": 0, "gate_measured": False,
            "proof_scope": "serialized tensors only; not historical step-zero or causal proof"})
        print(json.dumps({k: {x: v[x] for x in ("tensors", "numel", "byte_equal", "numeric_equal")}
                          for k, v in summaries.items()}))
    except Exception as error:
        write_json(output / "failure.json", {"status": "incomplete", "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.output)
