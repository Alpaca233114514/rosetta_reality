"""Fresh-base train40 calibration and exact zero-coefficient CUDA control."""

from __future__ import annotations

from scripts.iris_runtime import budget, load_native, observations, save, sha


def calibrate(plan_path, output, deadline):
    import torch
    from lerobot.policies.smolvla import modeling_smolvla

    from rosetta_reality.vla.image_key_regularization import (
        LAYERS,
        install,
        restore,
        scene_variance,
        validate_scales,
    )
    from scripts.diagnose_zen_noise_transfer import parameter_digests

    context = load_native(plan_path, split="train", deadline=deadline)
    policy = context.policy
    before = parameter_digests(policy)
    frozen = {name: before[name] for name, p in policy.named_parameters() if not p.requires_grad}
    if len(before) != 500 or len(frozen) != 345:
        raise ValueError("Registered base parameter/freeze layout differs")
    modules = dict(policy.named_modules())
    banks = {layer: [] for layer in LAYERS}
    active, counts, handles = {}, {}, []

    def hook(layer):
        def capture(_module, inputs, value):
            if (
                len(inputs) != 1
                or tuple(value.shape) != (1, 241, 320)
                or value.dtype != torch.bfloat16
            ):
                raise ValueError("Native full-prefix BF16 K layout differs")
            image = value[:, :64].detach().cpu().clone()
            counts[layer] = counts.get(layer, 0) + 1
            if layer not in active:
                active[layer] = image
            elif not torch.equal(active[layer], image):
                raise ValueError("Prefix K changed across denoising calls")

        return capture

    try:
        for layer in LAYERS:
            key = f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.k_proj"
            handles.append(modules[key].register_forward_hook(hook(layer)))
        for batch in context.batches:
            budget(deadline)
            active.clear()
            counts.clear()
            policy.reset()
            cfg = policy.config
            noise = torch.zeros((1, cfg.chunk_size, cfg.max_action_dim), device="cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                policy.predict_action_chunk(observations(batch), noise=noise)
            if counts != dict.fromkeys(LAYERS, 10):
                raise ValueError("Calibration denoising/layer coverage changed")
            for layer in LAYERS:
                banks[layer].append(active[layer])
    finally:
        for handle in handles:
            handle.remove()
    scales = {layer: float(scene_variance(torch.cat(banks[layer]).double())) for layer in LAYERS}
    validate_scales(scales)
    # Reset both RNG domains to compare the real native loss and gradients.
    policy.train()
    batch = {
        key: torch.cat([b[key] for b in context.batches[:4]])
        for key, value in context.batches[0].items()
        if isinstance(value, torch.Tensor)
    }
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()

    def forward():
        policy.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy({key: value.clone() for key, value in batch.items()})
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite native zero-control loss")
        loss.backward()
        gradients = {
            name: p.grad.detach().cpu().clone()
            for name, p in policy.named_parameters()
            if p.grad is not None
        }
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients.values()):
            raise FloatingPointError("Missing/nonfinite zero-control gradients")
        return loss.detach().cpu(), gradients, torch.get_rng_state(), torch.cuda.get_rng_state_all()

    native = forward()
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    original = modeling_smolvla.SmolVLAPolicy.forward
    install(modeling_smolvla, scales, 0)
    try:
        if modeling_smolvla.SmolVLAPolicy.forward is not original:
            raise ValueError("Zero coefficient did not bypass the wrapper")
        control = forward()
    finally:
        restore(modeling_smolvla)
    if (
        not torch.equal(native[0], control[0])
        or set(native[1]) != set(control[1])
        or any(not torch.equal(value, control[1][name]) for name, value in native[1].items())
        or not torch.equal(native[2], control[2])
        or any(not torch.equal(a, b) for a, b in zip(native[3], control[3], strict=True))
    ):
        raise ValueError("Native zero-coefficient loss/gradient/RNG parity failed")
    if parameter_digests(policy) != before:
        raise ValueError("Calibration changed parameters")
    budget(deadline)
    plan = context.plan
    report = {
        "status": "passed",
        "source": "fresh_pinned_base_train40",
        "labels_used": False,
        "labels_loaded_by_shared_input_reader": True,
        "dev_loaded": False,
        "hidden_loaded": False,
        "optimizer_steps": 0,
        "policy_forwards": 42,
        "zero_coefficient_exact": True,
        "zero_control_gradient_tensors": len(native[1]),
        "model_revision": context.experiment["model"]["revision"],
        "dataset_revision": context.experiment["dataset"]["revision"],
        "normalization_sha256": plan["normalization"]["report_sha256"],
        "dataset_view_manifest_sha256": plan["normalization"]["dataset_view_manifest_sha256"],
        "episodes": context.episodes,
        "frame_offset": 0,
        "layer_variance": {str(layer): value for layer, value in scales.items()},
        "frozen_parameter_sha256": frozen,
        "base_parameter_sha256": before,
        "base_weight_sha256": context.source_sha256,
        "plan_sha256": sha(plan_path),
        "image_sha256": context.image_sha256,
    }
    save(output, report)
