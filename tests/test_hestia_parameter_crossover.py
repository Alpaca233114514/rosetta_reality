"""Counterexamples for the exact sixteen-tensor intervention boundary."""

import json

import pytest
import torch

from scripts.hestia_parameter_crossover import (
    CONDITIONS,
    compare_checkpoints,
    condition_spec,
    kv_keys,
    read_header,
    substitute,
    tensor_file_digests,
)


def test_only_registered_reciprocal_conditions():
    assert [condition_spec(k) for k in CONDITIONS] == [
        (1280, None),
        (640, None),
        (1280, 640),
        (640, 1280),
    ]
    with pytest.raises(ValueError):
        condition_spec("best_dev")
    assert len(set(kv_keys())) == 16
    assert all(".layers.0." not in k and ".q_proj." not in k for k in kv_keys())


def test_exact_substitution_preserves_others():
    p = {k: torch.zeros(320, 320) for k in kv_keys()}
    p["other"] = torch.ones(2)
    donor = {k: torch.full((320, 320), i + 1.0) for i, k in enumerate(kv_keys())}
    substitute(p, donor)
    assert p["other"].tolist() == [1, 1]
    for k in donor:
        assert torch.equal(p[k], donor[k])


@pytest.mark.parametrize("problem", ["extra", "missing", "nonfinite", "shape", "dtype", "keys"])
def test_invalid_donor_causes_no_partial_write(problem):
    p = {k: torch.zeros(320, 320) for k in kv_keys()}
    d = {k: torch.ones(320, 320) for k in kv_keys()}
    args = {}
    if problem == "extra":
        d["other"] = torch.ones(1)
    if problem == "missing":
        d.pop(kv_keys()[-1])
    if problem == "nonfinite":
        d[kv_keys()[-1]][0, 0] = float("nan")
    if problem == "shape":
        d[kv_keys()[-1]] = torch.ones(2)
    if problem == "dtype":
        d[kv_keys()[-1]] = d[kv_keys()[-1]].double()
    if problem == "keys":
        args["keys"] = kv_keys()[:-1]
    with pytest.raises(ValueError):
        substitute(p, d, **args)
    assert all(torch.count_nonzero(v) == 0 for v in p.values())


def test_byte_audit_rejects_schema_and_frozen_drift():
    a = {k: {"dtype": "F32", "shape": [320, 320], "sha256": "a"} for k in kv_keys()}
    frozen = "model.vlm_with_expert.vlm.example"
    a[frozen] = {"dtype": "F32", "shape": [2], "sha256": "a"}
    b = json.loads(json.dumps(a))
    b[kv_keys()[0]]["sha256"] = "b"
    r = compare_checkpoints(a, b)
    assert r["changed_cross_kv"] == [kv_keys()[0]] and r["frozen_vlm_exact"]
    b[frozen]["sha256"] = "b"
    with pytest.raises(ValueError):
        compare_checkpoints(a, b)
    b = json.loads(json.dumps(a))
    b[kv_keys()[0]]["shape"] = [1]
    with pytest.raises(ValueError):
        compare_checkpoints(a, b)


def test_streamed_header_ranges(tmp_path):
    import hashlib

    p = tmp_path / "sample.safetensors"
    header = {"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode()
    p.write_bytes(len(raw).to_bytes(8, "little") + raw + b"abcd")
    assert tensor_file_digests(p)["a"]["sha256"] == hashlib.sha256(b"abcd").hexdigest()
    header["a"]["data_offsets"] = [1, 5]
    raw = json.dumps(header).encode()
    p.write_bytes(len(raw).to_bytes(8, "little") + raw + b"abcde")
    with pytest.raises(ValueError):
        read_header(p)
