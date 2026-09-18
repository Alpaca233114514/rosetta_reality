"""Pure file/protocol checks shared by the canonical post-training consumers."""

import hashlib
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(root, expected):
    root = Path(root).resolve(strict=True)
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Missing complete file seal")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Nested artifact links are forbidden")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected):
        raise ValueError("Artifact file set differs from the recovered seal")
    for name, digest in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root) or sha(path) != digest:
            raise ValueError("Artifact file SHA differs from recovered seal: " + name)


def validate_gate_protocol(actual, registered):
    for gate in ("gate3", "gate4"):
        expected = dict(registered[gate])
        if gate == "gate3":
            expected["seed"] = expected.pop("environment_seed")
        expected["report_suffix"] = "471"
        if actual.get(gate) != expected:
            raise ValueError("Executed Gate protocol differs: " + gate)
    if actual.get("inference") != registered["executed_inference"]:
        raise ValueError("Executed inference protocol differs")
    if actual.get("resources") != registered["executed_gate_resources"]:
        raise ValueError("Executed Gate resources differ")


def train_action_table(parquet, path, episodes):
    # Filtering precedes materialization, including Python conversion of action rows.
    return parquet.read_table(
        path,
        columns=["episode_index", "action"],
        filters=[("episode_index", "in", list(episodes))],
    )


def internal_grippers(action, dimensions):
    indices = [i for i, dim in enumerate(dimensions) if dim.encoding is not None]
    if len(indices) != 2 or action.shape[-1] != len(dimensions):
        raise ValueError("Explicit dual gripper action contract required")
    return action[..., indices]
