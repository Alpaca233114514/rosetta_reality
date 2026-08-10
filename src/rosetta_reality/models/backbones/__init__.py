"""Replaceable vision-language backbone adapters."""

from rosetta_reality.models.backbones.base import BackboneBatch, VLABackbone
from rosetta_reality.models.backbones.cached import CachedBackbone
from rosetta_reality.models.backbones.dummy import DummyBackbone

__all__ = ["BackboneBatch", "CachedBackbone", "DummyBackbone", "VLABackbone"]
