"""Plan-driven composition layer for the pinned LeRobot SmolVLA trainer.

This package is the version-2 replacement for the historical per-experiment
``train_smolvla_*`` / ``run_smolvla_*`` script stack.  It keeps the pinned
upstream trainer as the only training loop and turns every local extension
(logging, statistics, projections, samplers, loss profiles and memory
handling) into an explicitly declared, installable and restorable feature
bound to a hash-checked plan.  The historical scripts remain frozen as
provenance for the completed Faust, Aster and Way runs.

The rewrite is engineering restructuring only: it does not change any
learning semantics, does not authorize a new furnace by itself and does not
resolve the open Gate 4 research failure.
"""

from rosetta_reality.vla.training.context import TrainingContext
from rosetta_reality.vla.training.features import (
    FEATURE_FACTORIES,
    FeatureStack,
    feature_stack_from_plan,
)
from rosetta_reality.vla.training.launch import (
    build_training_arguments,
    compose_runtime_experiment,
    optimizer_arguments,
)
from rosetta_reality.vla.training.plan import (
    load_v2_plan,
    validate_optimizer_contract,
    validate_plan_structure,
)

__all__ = [
    "FEATURE_FACTORIES",
    "FeatureStack",
    "TrainingContext",
    "build_training_arguments",
    "compose_runtime_experiment",
    "feature_stack_from_plan",
    "load_v2_plan",
    "optimizer_arguments",
    "validate_optimizer_contract",
    "validate_plan_structure",
]
