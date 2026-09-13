"""Isolate the preregistered joint and complementary projected K head subsets."""

from scripts.hestia_k_direction_head import LAYER as LAYER
from scripts.hestia_k_direction_head import DirectionHeads as SingleHeads

CONDITIONS = (
    "base640_native",
    "base640_krestore",
    "base640_kdirection",
    "base640_kh1",
    "base640_kh2",
    "base640_kh12",
    "base640_kh034",
)
HEADS = {"kh1": (1,), "kh2": (2,), "kh12": (1, 2), "kh034": (0, 3, 4)}


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("Unregistered head interaction condition")
    return 640, None


def component_mode(name):
    condition_spec(name)
    return name.split("_")[1]


def reference_for(plan, condition):
    mode = component_mode(condition)
    if mode in ("native", "krestore"):
        return plan["reference_endpoints"]["640"]
    if mode == "kdirection":
        return plan["reference_direction"]
    if mode in ("kh1", "kh2"):
        return plan["reference_heads"][mode]
    return None


class DirectionHeads(SingleHeads):
    def __init__(
        self, policy, checkpoint_sha256, condition, calibration_dir, episodes, train_episodes
    ):
        mode = component_mode(condition)
        surrogate = "base640_kh1" if mode in ("kh12", "kh034") else condition
        super().__init__(
            policy, checkpoint_sha256, surrogate, calibration_dir, episodes, train_episodes
        )
        self.condition = condition
        if mode in HEADS:
            self.selected_heads = HEADS[mode]
