"""Independent torch reductions and payload verification of saved parameter rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"Independent arithmetic mismatch: {actual} != {expected}")


def verify(plan_path, result_dir, output):
    import torch
    from safetensors import safe_open

    from scripts.root_analysis_resources import verify_resource_envelope

    plan = json.loads(plan_path.read_text())
    resources = verify_resource_envelope(plan)
    result = json.loads((result_dir / "result.json").read_text())
    if result["status"] != "complete" or result["plan_sha256"] != digest(plan_path):
        raise ValueError("Completed plan-bound result required")
    for path, expected in plan["files"].items():
        if digest(path) != expected:
            raise ValueError("Source/input drift")
    verified = {}
    for left, right in plan["pairs"]:
        pair = f"{left}-to-{right}"
        path = result_dir / f"{pair}.jsonl"
        summary = result["pairs"][pair]
        if digest(path) != summary["rows_sha256"]:
            raise ValueError("Row file changed")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        with safe_open(plan["models"][left], framework="pt") as a, safe_open(
            plan["models"][right], framework="pt"
        ) as b:
            names = [r["name"] for r in rows]
            if len(set(names)) != len(names) or set(names) != set(a.keys()):
                raise ValueError("Tensor coverage differs")
            if set(names) != set(b.keys()):
                raise ValueError("Right tensor coverage differs")
            for key, expected in {
                "tensors": len(rows), "numel": sum(r["numel"] for r in rows),
                "byte_equal": sum(r["byte_equal"] for r in rows),
                "numeric_equal": sum(r["numeric_equal"] for r in rows),
            }.items():
                if summary[key] != expected:
                    raise ValueError("Summary count differs")
            # Every serialized tensor; different reduction library and block boundaries.
            for row in rows:
                originals = [f.get_tensor(row["name"]) for f in (a, b)]
                if (
                    list(originals[0].shape) != row["shape"]
                    or originals[0].shape != originals[1].shape
                ):
                    raise ValueError("Shape differs")
                hashes = [hashlib.sha256(), hashlib.sha256()]
                sums, squares, zeros = [0., 0.], [0., 0.], [0, 0]
                delta_square = dot = max_abs = 0.0
                changed = 0
                count = originals[0].numel()
                for start in range(0, count, 131072):
                    parts = [t.reshape(-1)[start:start + 131072] for t in originals]
                    x, y = [t.double() for t in parts]
                    for i, t in enumerate((x, y)):
                        if not torch.isfinite(t).all():
                            raise ValueError("Nonfinite")
                        hashes[i].update(parts[i].view(torch.uint8).numpy().tobytes())
                        sums[i] += t.sum().item()
                        squares[i] += t.square().sum().item()
                        zeros[i] += int((t == 0).sum())
                    delta = y - x
                    delta_square += delta.square().sum().item()
                    dot += (x * y).sum().item()
                    max_abs = max(max_abs, delta.abs().max().item())
                    changed += int((x != y).sum())
                if row["numel"] != count or row["changed_elements"] != changed:
                    raise ValueError("Element counts differ")
                same_bytes = (originals[0].dtype == originals[1].dtype
                              and hashes[0].hexdigest() == hashes[1].hexdigest())
                if row["byte_equal"] != same_bytes or row["numeric_equal"] != (changed == 0):
                    raise ValueError("Equality flags differ")
                if row["dtype"] != [str(t.dtype) for t in originals]:
                    raise ValueError("Dtypes differ")
                for i, side in enumerate(("left", "right")):
                    if hashes[i].hexdigest() != row[side]["payload_sha256"]:
                        raise ValueError("Payload hash differs")
                    close(sums[i] / count, row[side]["mean"])
                    close(math.sqrt(squares[i]), row[side]["l2"])
                    close(math.sqrt(squares[i] / count), row[side]["rms"])
                    close(math.sqrt(max(0., squares[i] / count - (sums[i] / count) ** 2)),
                          row[side]["std_population"])
                    close(zeros[i] / count, row[side]["zero_fraction"])
                close(math.sqrt(delta_square), row["delta_l2"])
                close(math.sqrt(delta_square / count), row["delta_rms"])
                close(max_abs, row["delta_max_abs"])
                close(changed / count, row["changed_fraction"])
                if squares[0]:
                    close(math.sqrt(delta_square / squares[0]), row["relative_l2"])
                elif row["relative_l2"] is not None:
                    raise ValueError("Undefined relative norm fabricated")
                if squares[0] and squares[1]:
                    close(dot / math.sqrt(squares[0] * squares[1]), row["cosine"])
                elif row["cosine"] is not None:
                    raise ValueError("Undefined cosine fabricated")
        verified[pair] = len(rows)
    for path, expected in plan["files"].items():
        if digest(path) != expected:
            raise ValueError("Post-verification source/input drift")
    with output.open("x") as stream:
        json.dump({"status": "passed", "verified_tensor_pairs": verified,
                   "resources": resources, "verifier_sha256": digest(__file__),
                   "plan_sha256": digest(plan_path),
                   "result_sha256": digest(result_dir / "result.json"),
                   "independent_reduction": "torch float64; block size 131072",
                   "optimizer_steps": 0, "model_execution": False}, stream, indent=2)
        stream.write("\n")
    print(json.dumps(verified))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.plan, args.result_dir, args.output)
