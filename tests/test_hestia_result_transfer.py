import pytest

from scripts.verify_hestia_checkpoint_results import validate_manifest


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
    value = manifest()
    del value["files"]["worker-exited.json"]
    value["total_bytes"] = 2
    with pytest.raises(ValueError):
        validate_manifest(value)
