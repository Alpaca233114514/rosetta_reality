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
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.state_encoder = state_encoder
        self.action_head = action_head
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
        return self.action_head(fused)
