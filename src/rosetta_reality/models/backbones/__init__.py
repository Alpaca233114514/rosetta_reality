"""Replaceable vision-language backbone adapters."""

from rosetta_reality.models.backbones.base import BackboneBatch, VLABackbone
from rosetta_reality.models.backbones.dummy import DummyBackbone

__all__ = ["BackboneBatch", "DummyBackbone", "VLABackbone"]

