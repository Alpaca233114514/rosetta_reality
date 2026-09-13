"""Native cross-attention value routing intervention; no model import side effects."""

import hashlib
import json

LAYERS = tuple(range(1, 16, 2))
ROUTES = {"image": (0, 64), "language": (192, 240), "state": (240, 241)}
CONDITIONS = ("base1280", "base640") + tuple(
    f"base{base}_v{mode}{1920 - base}" for mode in ("full", *ROUTES) for base in (1280, 640)
)


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("Unregistered value route condition")
    base = 1280 if name.startswith("base1280") else 640
    return base, 1920 - base


def component_mode(name):
    base, donor = condition_spec(name)
    return "native" if name == f"base{base}" else name.split("_v")[1].removesuffix(str(donor))


def digest(tensor):
    a = tensor.detach().cpu().float().numpy().copy()
    a[a == 0] = 0
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def route_output(native, donor, mode):
    if native.shape != donor.shape or tuple(native.shape) != (1, 241, 320):
        raise ValueError("Full native prefix shape differs")
    if mode == "native":
        return native
    if mode == "full":
        return donor
    if mode not in ROUTES:
        raise ValueError("Unregistered token route")
    start, stop = ROUTES[mode]
    result = native.clone()
    result[:, start:stop] = donor[:, start:stop]
    return result


class ValueRoutes:
    """Swap exact donor values at selected prefix positions, keeping every weight unchanged."""

    def __init__(self, policy, donor_file, condition, job):
        import torch
        from safetensors import safe_open

        self.condition = condition
        self.base, _ = condition_spec(condition)
        self.mode = component_mode(condition)
        self.row = None
        self.handles, self.cache, self.calls = [], {}, {}
        self.layout = None
        self.reference = None
        self.reference_sha = None
        if self.mode != "native":
            path = job / f"base{self.base}" / "intervention.json"
            native_result = json.loads((path.parent / "result.json").read_text())
            self.reference = json.loads(path.read_text())
            if self.reference != native_result["value_route"]:
                raise ValueError("Native identity seal differs")
            self.reference_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        modules = dict(policy.named_modules())
        self.donors = {}
        with safe_open(str(donor_file), framework="pt", device="cpu") as saved:
            for layer in LAYERS:
                name = f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.v_proj"
                module = modules[name]
                donor = saved.get_tensor(name + ".weight")
                if (
                    tuple(donor.shape) != (320, 320)
                    or donor.dtype != module.weight.dtype
                    or module.bias is not None
                    or not torch.isfinite(donor).all()
                ):
                    raise ValueError("V donor schema differs")
                self.donors[layer] = donor.to(module.weight.device)
        try:
            for layer in LAYERS:
                name = f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.v_proj"
                self.handles.append(modules[name].register_forward_hook(self.hook(layer)))
        except Exception:
            self.close()
            raise

    def set_layout(self, mask):
        import torch

        if (
            tuple(mask.shape) != (1, 241)
            or mask.dtype != torch.bool
            or not mask[:, :64].all()
            or mask[:, 64:192].any()
            or not mask[:, 240:].all()
        ):
            raise ValueError("Native real/empty camera and state mask differs")
        identity = {
            "shape": [1, 241, 320],
            "mask_sha256": digest(mask),
            "attended_tokens": int(mask.sum()),
        }
        if self.layout is not None and self.layout != identity:
            raise ValueError("Prefix mask varies across registered rows")
        if self.reference is not None and self.reference["layout"] != identity:
            raise ValueError("Prefix layout differs from same-base endpoint")
        self.layout = identity

    def hook(self, layer):
        def apply(module, inputs, output):
            import torch
            import torch.nn.functional as functional

            if self.row not in range(45) or len(inputs) != 1 or self.layout is None:
                raise ValueError("Missing registered row and prefix layout")
            x = inputs[0]
            if tuple(x.shape) != (1, 241, 320) or output.shape != x.shape:
                raise ValueError(
                    f"Native V prefix shape differs: {tuple(x.shape)}, {tuple(output.shape)}"
                )
            if output.dtype != torch.bfloat16 or not torch.is_autocast_enabled("cuda"):
                raise ValueError("Native CUDA BF16 autocast required")
            key = (self.row, layer)
            self.calls[key] = self.calls.get(key, 0) + 1
            if key not in self.cache:
                donor = functional.linear(x, self.donors[layer])
                if donor.dtype != output.dtype or not torch.isfinite(donor).all():
                    raise ValueError("Donor projection invalid")
                identity = {"input": digest(x), "native": digest(output), "donor": digest(donor)}
                identity["input_routes"] = {
                    name: digest(x[:, start:stop]) for name, (start, stop) in ROUTES.items()
                }
                if self.reference is not None:
                    if identity != self.reference["rows"][f"{self.row}:{layer}"]:
                        raise ValueError("Same-base native input or V output drift")
                self.cache[key] = (x.detach().clone(), output.detach().clone(), donor, identity)
            saved_x, saved_y, donor, _ = self.cache[key]
            if not torch.equal(x, saved_x) or not torch.equal(output, saved_y):
                raise ValueError("Prefix depends on noise, denoise iteration, or intervention")
            return route_output(output, donor, self.mode)

        return apply

    def finish(self, output):
        expected = {(row, layer) for row in range(45) for layer in LAYERS}
        if set(self.cache) != expected or any(self.calls.get(k) != 40 for k in expected):
            raise ValueError("Missing row/layer/noise/denoise calls")
        report = {
            "condition": self.condition,
            "mode": self.mode,
            "parameter_intervention": False,
            "activation_intervention": self.mode != "native",
            "projected_donor_evaluations": len(self.cache),
            "hook_calls": sum(self.calls.values()),
            "native_inputs_and_unmodified_outputs_repeat_exactly": True,
            "same_base_identity_matched": self.reference is not None,
            "native_reference_sha256": self.reference_sha,
            "layout": self.layout,
            "routes": ROUTES,
            "rows": {
                f"{row}:{layer}": self.cache[(row, layer)][3] for row, layer in sorted(expected)
            },
            "labels_used": False,
            "cross_checkpoint_state_prefix_claimed_identical": False,
            "attention_weights_claimed_fixed": False,
        }
        with (output / "intervention.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        return report

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
