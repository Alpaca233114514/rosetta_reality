"""Camera mask changes must never silently turn a skipped image into a valid key."""

from types import SimpleNamespace

import pytest
import torch

from rosetta_reality.vla.training.masked_camera import (
    install_masked_camera_encoder_skip,
    restore_masked_camera_encoder_skip,
)


def test_changed_placeholder_mask_rejected_on_second_forward():
    class Flow:
        def embed_prefix(self, *args):
            raise AssertionError("Native stub should be wrapped")

    module = SimpleNamespace(VLAFlowMatching=Flow)
    flow = Flow()
    flow.config = SimpleNamespace(empty_cameras=1)
    flow.add_image_special_tokens = False
    flow.prefix_length = 0
    flow.vlm_with_expert = SimpleNamespace(
        embed_image=lambda image: torch.ones((1, 2, 4)),
        embed_language_tokens=lambda tokens: torch.ones((1, 1, 4)),
    )
    flow.state_proj = lambda state: state
    args = [
        [torch.zeros((1, 3, 8, 8))] * 2,
        [torch.tensor([True]), torch.tensor([False])],
        torch.zeros((1, 1), dtype=torch.long),
        torch.tensor([[True]]),
        torch.zeros((1, 4)),
    ]
    install_masked_camera_encoder_skip(module)
    try:
        flow.embed_prefix(*args)
        args[1][1] = torch.tensor([True])
        with pytest.raises((RuntimeError, AssertionError), match="placeholder|masked"):
            flow.embed_prefix(*args)
    finally:
        restore_masked_camera_encoder_skip(module)


def test_smolvla_gradient_groups_do_not_collapse_into_model():
    from rosetta_reality.vla.training.features import GradientClipDiagnosticsFeature

    expert, state = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    expert.grad, state.grad = torch.tensor([3.0]), torch.tensor([4.0])
    result = GradientClipDiagnosticsFeature._norms(
        [expert, state],
        {
            id(expert): "model.vlm_with_expert.lm_expert.layers.1.weight",
            id(state): "model.state_proj.weight",
        },
    )
    assert result["global"] == 5
    assert len(set(result) - {"global"}) == 2
