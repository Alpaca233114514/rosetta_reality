"""BF16-aware projection-energy proxy diagnostic on existing native prefix records only."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.diagnose_hestia_checkpoint_localization import file_hash, read_json, require


def energy_parts(delta, train_count):
    a = np.asarray(delta, dtype=np.float64)
    require(a.ndim == 3 and np.isfinite(a).all(), "Scene/token/channel array required")
    token_mean = a[:train_count].mean(0, keepdims=True)
    global_mean = token_mean.mean(1, keepdims=True)
    token_pattern = token_mean - global_mean
    residual = a - token_mean
    output = {}
    for split, rows in (("train40", slice(0, train_count)), ("dev5", slice(train_count, None))):
        total = float(np.square(a[rows]).mean())
        constant = float(np.square(global_mean).mean())
        pattern = float(np.square(token_pattern).mean())
        varying = float(np.square(residual[rows]).mean())
        cross = float(2 * (token_mean * residual[rows]).mean())
        require(
            np.isclose(total, constant + pattern + varying + cross, atol=1e-12, rtol=1e-12),
            "Energy identity failed",
        )
        output[split] = {
            "total": total,
            "global_offset": constant,
            "shared_token_pattern": pattern,
            "scene_residual": varying,
            "cross": cross,
            "global_fraction": constant / total if total else None,
            "shared_token_fraction": pattern / total if total else None,
            "scene_fraction": varying / total if total else None,
        }
    return output


def main():
    plan = read_json("reports/training/m2-smolvla-hestia-value-energy-plan-002-2026-09-12.json")
    require(
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") == plan["image"], "Pinned container required"
    )
    for name, sha in plan["sha256"].items():
        require(file_hash(name) == sha, "Input/source drift")
    import torch
    from safetensors import safe_open

    root = Path(plan["prefix_root"])
    observer = read_json(root / "prefix-observer.json")
    audit = read_json(plan["tensor_audit"])
    require(
        observer["native_full_arrays_exact"] and len(observer["records"]) == 45,
        "Native prefix identity differs",
    )
    source, projected = {}, {}
    for layer in (1, 15):
        source[layer], projected[layer] = [], []
    for row, record in enumerate(observer["records"]):
        require(record["row"] == row, "Row order drift")
        p = root / record["file"]
        require(file_hash(p) == record["sha256"], "Prefix file drift")
        with np.load(p, allow_pickle=False) as a:
            for layer in (1, 15):
                source[layer].append(a[f"layer{layer}_v_input"][0])
                projected[layer].append(a[f"layer{layer}_v_projected"][0])
    matrices = {}
    for step in (640, 1280):
        with safe_open(plan["checkpoints"][str(step)], framework="pt", device="cpu") as archive:
            for layer in (1, 15):
                name = f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.v_proj.weight"
                value = archive.get_tensor(name).numpy()
                require(
                    value.dtype == np.float32 and value.shape == (320, 320),
                    "Expected F32 projection",
                )
                require(
                    hashlib.sha256(value.tobytes()).hexdigest() == audit[str(step)][name]["sha256"],
                    "Selected tensor drift",
                )
                matrices[step, layer] = value.astype(np.float64)
    result = {}
    for layer in (1, 15):
        x, actual = (
            np.asarray(source[layer], dtype=np.float64),
            np.asarray(projected[layer], dtype=np.float64),
        )
        w0, w1 = matrices[640, layer], matrices[1280, layer]
        q0 = torch.from_numpy(w0).to(torch.bfloat16).float().numpy().astype(np.float64)
        q1 = torch.from_numpy(w1).to(torch.bfloat16).float().numpy().astype(np.float64)
        qx = torch.from_numpy(x).to(torch.bfloat16).float().numpy().astype(np.float64)
        raw0, raw1 = qx @ q0.T, qx @ q1.T
        p0 = torch.from_numpy(raw0).to(torch.bfloat16).float().numpy().astype(np.float64)
        p1 = torch.from_numpy(raw1).to(torch.bfloat16).float().numpy().astype(np.float64)
        delta = p1 - p0
        result[str(layer)] = {
            "energy": energy_parts(delta, 40),
            "native_reconstruction_max_abs": float(np.abs(p1-actual).max()),
            "native_reconstruction_rmse": float(np.sqrt(np.square(p1-actual).mean())),
            "native_reconstruction_exact_fraction": float(np.mean(p1==actual)),
            "input_bf16_rounding_changes": int(np.count_nonzero(qx!=x)),
            "f32_weight_changed_elements": int(np.count_nonzero(w1!=w0)),
            "bf16_weight_changed_elements": int(np.count_nonzero(q1!=q0)),
            "weight_delta_rms_f32": float(np.sqrt(np.square(w1-w0).mean())),
            "weight_delta_rms_bf16": float(np.sqrt(np.square(q1-q0).mean())),
            "delta_rms": float(np.sqrt(np.square(delta).mean())),
        }
    out = Path(plan["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(
            {
                "results": result,
                "arithmetic": "BF16-rounded operands and output with float64 accumulation; checked against recorded native autocast output, not a CUDA replay",
                "model_forwards": 0,
                "causal_action_claim": False,
                "hidden_loaded": False,
            },
            f,
            indent=2,
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
