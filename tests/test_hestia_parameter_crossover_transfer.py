import pytest

from scripts.hestia_parameter_crossover import CONDITIONS
from scripts.verify_hestia_parameter_crossover_results import validate_manifest


def manifest():
    return {
        "files": {
            name: {"bytes": 1, "sha256": "0" * 64}
            for name in ("registration.json", "permit.json", "worker-exited.json")
        },
        "total_bytes": 3,
    }


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/absolute",
        "base640/../../outside",
        "base640/arrays.npz:stream",
        "best/arrays.npz",
        "base640/model.safetensors",
    ],
)
def test_unregistered_paths_rejected(name):
    v = manifest()
    v["files"][name] = {"bytes": 0, "sha256": "0" * 64}
    with pytest.raises(ValueError):
        validate_manifest(v)


def test_exact_four_conditions_only():
    v = manifest()
    for name in CONDITIONS:
        for file in ("arrays.npz", "result.json", "intervention.json"):
            v["files"][name + "/" + file] = {"bytes": 0, "sha256": "0" * 64}
    validate_manifest(v)
    del v["files"]["worker-exited.json"]
    v["total_bytes"] = 2
    with pytest.raises(ValueError):
        validate_manifest(v)


def test_size_identity_rejected():
    v = manifest()
    v["total_bytes"] = 4
    with pytest.raises(ValueError):
        validate_manifest(v)
