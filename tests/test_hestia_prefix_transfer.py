import pytest

from scripts.verify_hestia_prefix_results import validate_manifest


def manifest():
    files = {
        name: {"bytes": 1, "sha256": "0" * 64}
        for name in ("registration.json", "permit.json", "worker-exited.json")
    }
    return {"files": files, "total_bytes": 3}


@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "001280/../../outside", "001280/arrays.npz:stream"]
)
def test_escape_and_windows_stream_paths_rejected(name):
    value = manifest()
    value["files"][name] = {"bytes": 0, "sha256": "0" * 64}
    with pytest.raises(ValueError):
        validate_manifest(value)


def test_wrong_total_and_missing_exit_receipt_rejected():
    value = manifest()
    value["total_bytes"] = 4
    with pytest.raises(ValueError):
        validate_manifest(value)


def test_only_fixed_step_prefix_files_are_accepted():
    value = manifest()
    value["files"]["001280/prefix-row-044.npz"] = {"bytes": 2, "sha256": "a" * 64}
    value["files"]["001280/prefix-observer.json"] = {"bytes": 2, "sha256": "b" * 64}
    value["total_bytes"] = 7
    validate_manifest(value)
    value["files"]["000640/arrays.npz"] = {"bytes": 0, "sha256": "c" * 64}
    with pytest.raises(ValueError):
        validate_manifest(value)
    value = manifest()
    del value["files"]["worker-exited.json"]
    value["total_bytes"] = 2
    with pytest.raises(ValueError):
        validate_manifest(value)
