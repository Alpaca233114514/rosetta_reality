"""Composable sample transforms."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from rosetta_reality.data.schema import RosettaSample

SampleTransform = Callable[[RosettaSample], RosettaSample]


class Compose:
    """Apply sample transforms in declaration order."""

    def __init__(self, transforms: Iterable[SampleTransform]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, sample: RosettaSample) -> RosettaSample:
        """Return a sample after all transforms have run."""

        for transform in self.transforms:
            sample = transform(sample)
        return sample

