"""Observe native CUDA visual K/V before/after expert projection with exact parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from scripts import diagnose_hestia_cuda_checkpoints as frozen  # noqa: E402


def validate_permit(template, permit, step, now):
    if step != 1280 or permit.get("allowed_steps") != [1280]:
        raise ValueError("Only the fixed final checkpoint is registered")
    if (
        permit.get("model_execution_authorized") is not True
        or permit.get("training_authorized") is not False
        or permit.get("shutdown_authorized") is not True
    ):
        raise ValueError("Explicit inference-only and shutdown authorization required")
    if permit.get("template_sha256") != template or permit.get("watchdog_active") is not True:
        raise ValueError("Template/watchdog identity differs")
    start, end = permit.get("started_unix", 0), permit.get("deadline_unix", 0)
    if not 0 < end - start <= 600 or not start <= now < end:
        raise ValueError("Missing or expired shared deadline")


class PrefixObserver:
    """Read-only hooks, bounded to real-camera tokens; no policy tensor replacement."""

    def __init__(self, output, *, camera_tokens=64, steps=10, rows=45, maximum_bytes=48 * 1024**2):
        self.output, self.tokens, self.steps, self.rows = Path(output), camera_tokens, steps, rows
        self.maximum_bytes = maximum_bytes
        self.references, self.counts, self.saved, self.records = {}, {}, {}, []
        self.total_bytes = 0
        self.index = None

    def begin(self, index):
        if self.index is not None or not 0 <= index < self.rows * 4:
            raise ValueError("Unexpected collector call order")
        self.index, self.references, self.counts = index, {}, {}

    def record(self, name, value):
        import torch

        if (
            self.index is None
            or value.ndim != 3
            or value.shape[0] != 1
            or value.shape[1] < self.tokens
        ):
            raise ValueError("Native visual prefix layout differs")
        if not 0 < value.shape[2] <= 1024:
            raise ValueError("Unregistered prefix width")
        real = value[:, : self.tokens].detach()
        if not torch.isfinite(real).all():
            raise ValueError("Nonfinite prefix")
        if name in self.references:
            if not torch.equal(real, self.references[name]):
                raise ValueError("Prefix changed during denoising")
        else:
            self.references[name] = real.clone()
        self.counts[name] = self.counts.get(name, 0) + 1

    def pre_hook(self, name):
        def hook(_module, inputs):
            self.record(name, inputs[0])
            return None

        return hook

    def post_hook(self, name):
        def hook(_module, _inputs, output):
            self.record(name, output)
            return None

        return hook

    def finish(self):
        expected = {
            f"layer{layer}_{kind}_{stage}"
            for layer in (1, 15)
            for kind in ("k", "v")
            for stage in ("input", "projected")
        }
        if set(self.counts) != expected or any(n != self.steps for n in self.counts.values()):
            raise ValueError("Missing native projection calls")
        arrays = {k: v.cpu().float().numpy().copy() for k, v in self.references.items()}
        hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}
        row = self.index % self.rows
        if self.index < self.rows:
            self.total_bytes += sum(v.nbytes for v in arrays.values())
            if self.total_bytes > self.maximum_bytes:
                raise ValueError("Prefix evidence exceeds bound")
            path = self.output / f"prefix-row-{row:03d}.npz"
            with path.open("xb") as f:
                np.savez_compressed(f, **arrays)
            with np.load(path, allow_pickle=False) as a:
                if set(a.files) != expected or any(
                    not np.array_equal(a[k], v) for k, v in arrays.items()
                ):
                    raise ValueError("Saved prefix reload differs")
            self.saved[row] = hashes
            self.records.append(
                {
                    "row": row,
                    "file": path.name,
                    "sha256": frozen.file_hash(path),
                    "shapes": {k: list(v.shape) for k, v in arrays.items()},
                }
            )
        elif hashes != self.saved.get(row):
            raise ValueError("Prefix differs across fixed inference noises")
        self.index, self.references, self.counts = None, {}, {}


def observed_collect(plan, output):
    import evaluate_visual_native_small as native

    observer = PrefixObserver(output)
    original_factory = native.make_policy
    handles, policies, counters = [], [], []

    def factory(*args, **kwargs):
        policy = original_factory(*args, **kwargs)
        core = policy.model.vlm_with_expert
        if (
            policy.config.add_image_special_tokens
            or policy.config.attention_mode != "cross_attn"
            or core.num_vlm_layers != 16
            or core.num_expert_layers != 16
            or core.self_attn_every_n_layers != 2
        ):
            raise ValueError("Native cross-attention layout differs")
        if any(p.requires_grad for p in core.vlm.parameters()):
            raise ValueError("Frozen VLM required")
        for layer in (1, 15):
            for kind in ("k", "v"):
                module = getattr(core.lm_expert.layers[layer].self_attn, kind + "_proj")
                handles.append(
                    module.register_forward_pre_hook(
                        observer.pre_hook(f"layer{layer}_{kind}_input")
                    )
                )
                handles.append(
                    module.register_forward_hook(
                        observer.post_hook(f"layer{layer}_{kind}_projected")
                    )
                )
        original_predict = policy.predict_action_chunk
        counter = [0]

        def predict(batch, *args, **kwargs):
            images, masks = policy.prepare_images(batch)
            vision = core.get_vlm_model().vision_model
            scale = core.get_vlm_model().connector.scale_factor
            patch = vision.config.patch_size
            if (images[0].shape[-2] // patch // scale, images[0].shape[-1] // patch // scale) != (
                8,
                8,
            ):
                raise ValueError("Registered 8 by 8 real-camera token grid differs")
            if [bool(m.all()) for m in masks] != [True, False, False]:
                raise ValueError("Camera masks differ")
            observer.begin(counter[0])
            prediction = original_predict(batch, *args, **kwargs)
            observer.finish()
            counter[0] += 1
            return prediction

        policies.append((policy, original_predict))
        counters.append(counter)
        policy.predict_action_chunk = predict
        return policy

    try:
        native.make_policy = factory
        ORIGINAL_COLLECT(plan, output)
        if len(counters) != 1 or counters[0][0] != 180 or len(observer.records) != 45:
            raise ValueError("Incomplete fixed native collection")
        # The frozen collector verifies full normalized and standard arrays exactly.
        control = json.loads((output / "result.json").read_text())["same_device_control"]
        if (
            not control["passed"]
            or control["normalized_predictions"]["max_abs"] != 0
            or control["standard_predictions"]["max_abs"] != 0
        ):
            raise ValueError("Observer does not reproduce original CUDA exactly")
        with (output / "prefix-observer.json").open("x") as f:
            json.dump(
                {
                    "status": "passed",
                    "records": observer.records,
                    "real_camera_tokens": 64,
                    "unpooled_tokens_retained": True,
                    "within_denoising_exact": True,
                    "across_four_noises_exact": True,
                    "native_full_arrays_exact": True,
                    "raw_array_bytes": observer.total_bytes,
                    "policy_forwards": 180,
                    "optimizer_steps": 0,
                },
                f,
                indent=2,
            )
    finally:
        native.make_policy = original_factory
        for policy, original in policies:
            policy.predict_action_chunk = original
        for handle in handles:
            handle.remove()


ORIGINAL_COLLECT = frozen.collect


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--template", type=Path, required=True)
    args, _ = parser.parse_known_args()
    plan = json.loads(args.template.read_text(encoding="utf-8-sig"))
    if (
        plan.get("id") != "hestia-prefix-path-20260912-001"
        or plan.get("camera_tokens") != 64
        or plan.get("prefix_layers") != [1, 15]
    ):
        raise ValueError("Unregistered observer protocol")
    old_collect, old_validator = frozen.collect, frozen.validate_permit
    try:
        frozen.collect, frozen.validate_permit = observed_collect, validate_permit
        frozen.main()
    finally:
        frozen.collect, frozen.validate_permit = old_collect, old_validator


if __name__ == "__main__":
    main()
