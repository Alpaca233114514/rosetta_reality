"""Single-layer V reciprocal interventions; no module import side effects."""

CONDITIONS = (
    "base1280",
    "base640",
    "base1280_v1_640",
    "base640_v1_1280",
    "base1280_v3_640",
    "base640_v3_1280",
    "base1280_v5_640",
    "base640_v5_1280",
    "base1280_v7_640",
    "base640_v7_1280",
    "base1280_v9_640",
    "base640_v9_1280",
    "base1280_v11_640",
    "base640_v11_1280",
    "base1280_v13_640",
    "base640_v13_1280",
    "base1280_v15_640",
    "base640_v15_1280",
)


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("Unregistered V-layer condition")
    base = 1280 if name.startswith("base1280") else 640
    return base, None if name in CONDITIONS[:2] else 1920 - base


def keys_for(name):
    condition_spec(name)
    if name in CONDITIONS[:2]:
        raise ValueError("An endpoint must not receive a parameter intervention")
    layer = int(name.split("_")[1][1:])
    return (f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.v_proj.weight",)


def substitute(parameters, donor, condition):
    import torch

    allowed = keys_for(condition)
    if set(donor) != set(allowed) or not set(allowed) <= set(parameters):
        raise ValueError("Exact registered one-key substitution required")
    for name in allowed:
        p, v = parameters[name], donor[name]
        if p.shape != v.shape or tuple(p.shape) != (320, 320) or p.dtype != v.dtype:
            raise ValueError("Invalid donor schema")
        if not torch.isfinite(v).all():
            raise ValueError("Nonfinite donor")
    with torch.no_grad():
        for name in allowed:
            parameters[name].copy_(donor[name])


def patch_policy(policy, donor_file, condition):
    from safetensors import safe_open

    from scripts.diagnose_zen_noise_transfer import parameter_digests

    before = parameter_digests(policy)
    parameters = dict(policy.named_parameters())
    allowed = keys_for(condition)
    with safe_open(str(donor_file), framework="pt", device="cpu") as f:
        donor = {k: f.get_tensor(k) for k in allowed}
    substitute(parameters, donor, condition)
    after = parameter_digests(policy)
    changed = sorted(k for k in before if before[k] != after[k])
    if set(changed) != set(allowed):
        raise ValueError("Intervention must change exactly the registered tensor")
    if any(not parameters[k].detach().cpu().equal(v) for k, v in donor.items()):
        raise ValueError("Donor copy was not exact")
    return {
        "condition": condition,
        "allowed_keys": list(allowed),
        "changed_keys": changed,
        "other_parameters_exact": True,
        "donor_copy_exact": True,
        "before": before,
        "after": after,
    }
