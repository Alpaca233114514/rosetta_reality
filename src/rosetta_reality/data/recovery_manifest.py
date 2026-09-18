"""Create-only identity contract for future state-conditioned recovery data."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosetta_reality.data.manifest import require_commit_sha

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_FIELDS = frozenset(
    {
        "rollout_id",
        "transition_index",
        "robot_state",
        "images",
        "instruction",
        "oracle_action",
        "source_episode",
        "source_reference_index",
        "oracle_state_distance",
    }
)


def _sha256(value: str, name: str) -> str:
    normalized = value.lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return normalized


def _unique_non_negative(values: tuple[int, ...], name: str) -> set[int]:
    result = set(values)
    if len(result) != len(values) or any(value < 0 for value in values):
        raise ValueError(f"{name} must contain unique non-negative integers.")
    return result


@dataclass(frozen=True, slots=True)
class RecoveryDatasetManifest:
    """Identity and isolation boundary for recovery-labelled trajectories."""

    dataset_id: str
    source_repo_id: str
    source_revision: str
    source_manifest_sha256: str
    action_contract_sha256: str
    oracle_implementation_sha256: str
    oracle_evaluation_report_sha256: str
    oracle_protocol: str
    authorized_train_episodes: tuple[int, ...]
    source_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]
    hidden_test_episodes: tuple[int, ...]
    collection_simulator_seeds: tuple[int, ...]
    oracle_evaluation_seeds: tuple[int, ...]
    policy_gate4_seeds: tuple[int, ...]
    sample_count: int
    state_dimension: int
    action_dimension: int
    records_sha256: str
    fields: dict[str, str]
    perturbation_contract: dict[str, Any]
    oracle_gate_status: str = "passed"
    state_conditioned: bool = True
    time_indexed_reference: bool = False
    target_action_noise: bool = False
    hidden_test_loaded: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"Unsupported recovery manifest version: {self.version!r}.")
        if not self.dataset_id or not self.source_repo_id or not self.oracle_protocol:
            raise ValueError("Recovery dataset identity and oracle protocol must be non-empty.")
        require_commit_sha(self.source_revision)
        for name in (
            "source_manifest_sha256",
            "action_contract_sha256",
            "oracle_implementation_sha256",
            "oracle_evaluation_report_sha256",
            "records_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if (
            self.oracle_gate_status != "passed"
            or self.state_conditioned is not True
            or self.time_indexed_reference is not False
            or self.target_action_noise is not False
            or self.hidden_test_loaded is not False
        ):
            raise ValueError(
                "Recovery data requires a passed state-conditioned oracle gate, clean targets, "
                "and a sealed hidden test."
            )
        if self.sample_count <= 0 or self.state_dimension <= 0 or self.action_dimension <= 0:
            raise ValueError("Recovery sample count and tensor dimensions must be positive.")

        train = _unique_non_negative(self.authorized_train_episodes, "Train episodes")
        source = _unique_non_negative(self.source_episodes, "Source episodes")
        validation = _unique_non_negative(self.validation_episodes, "Validation episodes")
        hidden = _unique_non_negative(self.hidden_test_episodes, "Hidden-test episodes")
        if not train or not source or not source <= train:
            raise ValueError(
                "Recovery source episodes must be a non-empty subset of train episodes."
            )
        if train & validation or train & hidden or validation & hidden:
            raise ValueError(
                "Recovery train, validation, and hidden-test episodes must be disjoint."
            )

        collection = _unique_non_negative(
            self.collection_simulator_seeds, "Collection simulator seeds"
        )
        evaluation = _unique_non_negative(
            self.oracle_evaluation_seeds, "Oracle evaluation seeds"
        )
        policy_gate = _unique_non_negative(self.policy_gate4_seeds, "Policy Gate 4 seeds")
        if not collection or not evaluation or not policy_gate:
            raise ValueError(
                "Recovery collection, oracle evaluation, and policy Gate seeds are required."
            )
        if collection & evaluation or collection & policy_gate or evaluation & policy_gate:
            raise ValueError(
                "Recovery collection, oracle evaluation, and policy Gate seeds must be disjoint."
            )

        if set(self.fields) != REQUIRED_FIELDS or any(not value for value in self.fields.values()):
            raise ValueError("Recovery field mapping does not match the version-one schema.")
        if not self.perturbation_contract:
            raise ValueError("Recovery data requires an explicit perturbation contract.")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible payload."""

        payload = asdict(self)
        for name in (
            "authorized_train_episodes",
            "source_episodes",
            "validation_episodes",
            "hidden_test_episodes",
            "collection_simulator_seeds",
            "oracle_evaluation_seeds",
            "policy_gate4_seeds",
        ):
            payload[name] = list(payload[name])
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RecoveryDatasetManifest:
        """Restore and fully validate a version-one manifest."""

        payload = dict(value)
        for name in (
            "authorized_train_episodes",
            "source_episodes",
            "validation_episodes",
            "hidden_test_episodes",
            "collection_simulator_seeds",
            "oracle_evaluation_seeds",
            "policy_gate4_seeds",
        ):
            payload[name] = tuple(int(item) for item in payload[name])
        return cls(**payload)


def save_recovery_manifest(root: Path, manifest: RecoveryDatasetManifest) -> Path:
    """Create a recovery manifest once or validate an identical existing file."""

    path = root / "manifest.json"
    if path.exists():
        existing = RecoveryDatasetManifest.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if existing != manifest:
            raise FileExistsError(f"Refusing to overwrite a different recovery manifest at {path}.")
        return path
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Refusing to manage non-empty recovery directory without a manifest: {root}."
        )
    root.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(manifest.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def load_recovery_manifest(path: Path) -> RecoveryDatasetManifest:
    """Read and validate a recovery manifest without mutating its directory."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Recovery manifest must contain a JSON object.")
    return RecoveryDatasetManifest.from_dict(value)
