"""Generic composition of backbone, robot state, and action prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rosetta_reality.models.action_head import ContinuousActionHead
from rosetta_reality.models.backbones.base import BackboneBatch, VLABackbone
from rosetta_reality.models.state_encoder import StateEncoder


class VLAPolicy(nn.Module):
    """Fuse a replaceable backbone representation with encoded robot state."""

    def __init__(
        self,
        backbone: VLABackbone,
        state_encoder: StateEncoder,
        action_head: ContinuousActionHead,
        *,
        state_to_action_scale: Tensor | None = None,
        state_to_action_offset: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.state_encoder = state_encoder
        self.action_head = action_head
        if (state_to_action_scale is None) != (state_to_action_offset is None):
            raise ValueError("Residual action scale and offset must be provided together.")
        self.residual_from_current_state = state_to_action_scale is not None
        if state_to_action_scale is not None and state_to_action_offset is not None:
            expected = (action_head.action_dim,)
            if (
                tuple(state_to_action_scale.shape) != expected
                or tuple(state_to_action_offset.shape) != expected
                or state_encoder.state_dim != action_head.action_dim
            ):
                raise ValueError(
                    "Residual action affine tensors and robot state must match action_dim."
                )
            self.register_buffer(
                "state_to_action_scale",
                state_to_action_scale.detach().to(torch.float32).clone(),
            )
            self.register_buffer(
                "state_to_action_offset",
                state_to_action_offset.detach().to(torch.float32).clone(),
            )
        fusion_input_dim = backbone.hidden_size + state_encoder.output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, action_head.input_dim),
            nn.GELU(),
            nn.LayerNorm(action_head.input_dim),
        )

    def forward(self, observations: BackboneBatch, robot_state: Tensor) -> Tensor:
        """Predict an action chunk for each observation and robot-state pair."""

        backbone_hidden = self.backbone(observations)
        state_hidden = self.state_encoder(robot_state)
        if backbone_hidden.shape[0] != state_hidden.shape[0]:
            raise ValueError(
                "Backbone observations and robot_state must have the same batch size, "
                f"but received {backbone_hidden.shape[0]} and {state_hidden.shape[0]}."
            )
        fused = self.fusion(torch.cat((backbone_hidden, state_hidden), dim=-1))
        actions = self.action_head(fused)
        if self.residual_from_current_state:
            baseline = (
                robot_state * self.state_to_action_scale + self.state_to_action_offset
            )
            actions = actions + baseline.unsqueeze(1)
        return actions
