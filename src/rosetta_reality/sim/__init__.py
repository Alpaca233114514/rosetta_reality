"""Simulation adapter interfaces and physical action contracts."""

from rosetta_reality.sim.action_contract import (
    ActionContract,
    ActionDimension,
    load_action_contract,
)
from rosetta_reality.sim.env import SimulationEnvironment
from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment
from rosetta_reality.sim.recovery_oracle import (
    OracleDecision,
    OracleOutOfDistributionError,
    OracleReferenceTrajectory,
    StateConditionedTrajectoryOracle,
)

__all__ = [
    "ActionContract",
    "ActionDimension",
    "GymAlohaEnvironment",
    "OracleDecision",
    "OracleOutOfDistributionError",
    "OracleReferenceTrajectory",
    "SimulationEnvironment",
    "StateConditionedTrajectoryOracle",
    "load_action_contract",
]
