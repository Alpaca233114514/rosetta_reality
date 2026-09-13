"""Isolate individual KV heads in the preregistered first expert cross-attention layer."""

from scripts.hestia_k_direction_layer import DirectionLayers
from scripts.hestia_k_factor import factors
from scripts.hestia_scene_kv_mean import SceneKVMean

LAYER = 1
CONDITIONS = ("base640_native", "base640_krestore", "base640_kdirection") + tuple(
    f"base640_kh{head}" for head in range(5)
)


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("Unregistered direction-head condition")
    return 640, None


def component_mode(name):
    condition_spec(name)
    return name.split("_")[1]


class DirectionHeads(DirectionLayers):
    def __init__(
        self, policy, checkpoint_sha256, condition, calibration_dir, episodes, train_episodes
    ):
        mode = component_mode(condition)
        self.selected_heads = (
            ()
            if mode == "native"
            else tuple(range(5))
            if mode in {"krestore", "kdirection"}
            else (int(mode[-1]),)
        )
        surrogate = condition if mode in {"native", "krestore"} else f"base640_kd{LAYER:02d}"
        super().__init__(
            policy, checkpoint_sha256, surrogate, calibration_dir, episodes, train_episodes
        )
        self.condition = condition
        self.selected = () if mode == "native" else (LAYER,)

    def hook(self, key):
        observer = SceneKVMean.hook(self, key)

        def apply(module, inputs, native):
            import torch

            observed = observer(module, inputs, native)
            if key.endswith("v") or int(key.split("_")[0]) not in self.selected:
                return observed
            pair = self.row, key
            if pair not in self.transformed:
                full = factors(
                    native, self.means[key], self.norms[key], self.head_dim, self.factor_mode
                )
                output = native.clone()
                selected = output[:, :64].reshape(1, 64, 5, 64)
                replacement = full[:, :64].reshape_as(selected)
                selected[:, :, self.selected_heads] = replacement[:, :, self.selected_heads]
                rest = [head for head in range(5) if head not in self.selected_heads]
                if not torch.equal(
                    selected[:, :, rest], native[:, :64].reshape_as(selected)[:, :, rest]
                ):
                    raise ValueError("Unselected KV head changed")
                self.transformed[pair] = output
            return self.transformed[pair]

        return apply

    def finish(self, output):
        import json

        with (output / "head-selection.json").open("x") as stream:
            json.dump(
                {
                    "layer": LAYER,
                    "heads": list(self.selected_heads),
                    "condition": self.condition,
                    "head_dim": 64,
                    "query_heads_per_kv_head": 3,
                },
                stream,
                indent=2,
            )
        return super().finish(output)
