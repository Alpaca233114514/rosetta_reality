import json

import pytest
import torch

from scripts.diagnose_hestia_kv_split import endpoint_controls, validate_permit
from scripts.hestia_kv_split import CONDITIONS, keys_for, substitute
from scripts.hestia_parameter_crossover import kv_keys
from scripts.verify_hestia_kv_split_results import validate_manifest


def test_disjoint_partition_and_reciprocal_identity():
    k, v = set(keys_for(CONDITIONS[2])), set(keys_for(CONDITIONS[4]))
    assert len(k) == len(v) == 8 and not k & v
    assert k | v == set(kv_keys())
    assert k == set(keys_for(CONDITIONS[3]))
    assert v == set(keys_for(CONDITIONS[5]))


@pytest.mark.parametrize("condition", CONDITIONS[2:])
def test_only_registered_half_changes(condition):
    parameters = {key: torch.zeros(320, 320) for key in kv_keys()}
    donor = {key: torch.ones(320, 320) for key in keys_for(condition)}
    substitute(parameters, donor, condition)
    for key, value in parameters.items():
        assert torch.equal(
            value, torch.ones_like(value) if key in donor else torch.zeros_like(value)
        )


@pytest.mark.parametrize("bad", ["missing", "extra", "nan", "shape", "dtype"])
def test_invalid_donor_is_atomic(bad):
    condition = CONDITIONS[2]
    parameters = {key: torch.zeros(320, 320) for key in kv_keys()}
    donor = {key: torch.ones(320, 320) for key in keys_for(condition)}
    key = next(iter(donor))
    if bad == "missing":
        del donor[key]
    elif bad == "extra":
        donor[keys_for(CONDITIONS[4])[0]] = torch.ones(320, 320)
    elif bad == "nan":
        donor[key][0, 0] = float("nan")
    elif bad == "shape":
        donor[key] = torch.ones(320, 319)
    else:
        donor[key] = donor[key].double()
    with pytest.raises(ValueError):
        substitute(parameters, donor, condition)
    assert all(torch.count_nonzero(v) == 0 for v in parameters.values())


@pytest.mark.parametrize("condition", CONDITIONS[2:])
def test_both_fresh_endpoints_still_required(tmp_path, condition):
    with pytest.raises(FileNotFoundError):
        endpoint_controls(tmp_path, condition)
    folder = tmp_path / "base1280"
    folder.mkdir()
    (folder / "result.json").write_text(
        json.dumps(
            {
                "condition": "base1280",
                "intervention_applied": False,
                "same_device_control": {"passed": False},
            }
        )
    )
    with pytest.raises(ValueError, match="endpoint"):
        endpoint_controls(tmp_path, condition)


def test_six_conditions_and_twenty_minute_work_limit():
    permit = {
        "model_execution_authorized": True,
        "training_authorized": False,
        "shutdown_authorized": True,
        "template_sha256": "abc",
        "watchdog_active": True,
        "allowed_conditions": list(CONDITIONS),
        "started_unix": 100,
        "deadline_unix": 1300,
    }
    validate_permit("abc", permit, CONDITIONS[-1], 101)
    permit["deadline_unix"] = 1301
    with pytest.raises(ValueError, match="deadline"):
        validate_permit("abc", permit, CONDITIONS[-1], 101)


def test_transfer_accepts_registered_six_arms_only():
    files = {
        name: {"bytes": 1, "sha256": "0" * 64}
        for name in ["registration.json", "permit.json", "worker-exited.json"]
        + [f"{c}/result.json" for c in CONDITIONS]
    }
    validate_manifest({"files": files, "total_bytes": len(files)})
    files["base1280_kv640/result.json"] = {"bytes": 1, "sha256": "0" * 64}
    with pytest.raises(ValueError):
        validate_manifest({"files": files, "total_bytes": len(files)})
