from __future__ import annotations

import pytest
import torch

from rosetta_reality.models.backbones import CachedBackbone


def test_cached_backbone_preserves_precomputed_features() -> None:
    backbone = CachedBackbone(hidden_size=8)
    features = torch.randn(3, 8)

    assert torch.equal(backbone({"features": features}), features)
    assert not tuple(backbone.parameters())


def test_cached_backbone_rejects_identity_mismatch() -> None:
    backbone = CachedBackbone(hidden_size=8)

    with pytest.raises(ValueError, match="shape"):
        backbone({"features": torch.randn(3, 7)})
