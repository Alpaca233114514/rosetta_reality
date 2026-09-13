"""Create-only CUDA arithmetic audit of the 320 sealed Q-replay snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(tensor):
    import torch

    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        str((tuple(tensor.shape), tensor.dtype)).encode()
        + tensor.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def audit(root):
    import numpy as np
    import torch

    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Registered CUDA 4090 runtime required")
    records = []
    for base in (640, 1280):
        events = {}
        with (root / f"base{base}/attention.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["mode"] == "native" and event["noise"] == 0 and event["row"] in (0, 40):
                    key = (event["row"], event["step"], event["layer"])
                    if key in events:
                        raise ValueError("Duplicate sentinel event")
                    events[key] = event
        expected = {(r, s, layer) for r in (0, 40) for s in range(10) for layer in range(1, 16, 2)}
        if set(events) != expected:
            raise ValueError("Sentinel event coverage changed")
        for (row, step, layer), event in sorted(events.items()):
            path = root / f"base{base}/sentinel-{row}-{step}-{layer}.npz"
            with np.load(path, allow_pickle=False) as arrays:
                values = {}
                for name, dtype in zip(
                    ("q", "k", "v", "probs", "output"), arrays["dtypes"], strict=True
                ):
                    if str(dtype) not in ("torch.float32", "torch.bfloat16"):
                        raise ValueError("Unexpected recorded dtype")
                    values[name] = torch.from_numpy(arrays[name].copy()).to(
                        getattr(torch, str(dtype).split(".")[-1])
                    )
                values["mask"] = torch.from_numpy(arrays["mask"].copy())
            for name, tensor in values.items():
                if digest(tensor) != event[name]:
                    raise ValueError("Snapshot does not match original CUDA event")
            q, k, v, mask = (values[n].cuda() for n in ("q", "k", "v", "mask"))
            b, nq, nh, hd = q.shape
            nk = k.shape[2]
            # Independently express the pinned eager operations. Autocast rounds
            # GEMM outputs; the in-place scale is rounded before FP32 softmax.
            kh = k.repeat_interleave(nh // nk, dim=2).transpose(1, 2)
            vh = v.repeat_interleave(nh // nk, dim=2).transpose(1, 2)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                score = torch.matmul(q.float().transpose(1, 2), kh.float().transpose(-2, -1))
                score.mul_(hd**-0.5)
                score32 = score.float()
                score32 = score32.masked_fill(~mask[:, None], torch.finfo(torch.float32).min)
                probability = torch.softmax(score32, dim=-1).to(v.dtype)
                result = (probability @ vh).transpose(1, 2).reshape(b, nq, nh * hd)
            p_ok = digest(probability) == event["probs"]
            o_ok = digest(result) == event["output"]
            records.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "probability_exact": p_ok,
                    "output_exact": o_ok,
                    "probability_max_abs": float(
                        (probability.cpu().float() - values["probs"].float()).abs().max()
                    ),
                    "output_max_abs": float(
                        (result.cpu().float() - values["output"].float()).abs().max()
                    ),
                }
            )
    return {
        "id": "iris-attention-backend-20260913-001",
        "status": "passed"
        if all(r["probability_exact"] and r["output_exact"] for r in records)
        else "failed",
        "records": records,
        "snapshot_tensor_hash_checks": len(records) * 6,
        "original_float64_one_ulp_check": "failed_not_relaxed",
        "interpretation": "same_backend_native_order_reproduction_only",
        "new_policy_forwards": 0,
        "optimizer_steps": 0,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "nested_docker_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output is create-only")
    report = audit(args.root)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
