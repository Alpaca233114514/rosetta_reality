"""Prove full-array comparison rejects equal-metric and same-process substitutes."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rosetta_reality.vla.reload_evidence import (
    HASH_IDENTITIES,
    compare_bundles,
    write_bundle,
)

IDENTITY = {
    **{key: "a" * 64 for key in HASH_IDENTITIES},
    "sample_count": 2,
    "chunk_size": 3,
    "normalized_action_dim": 2,
    "standard_action_dim": 2,
}


def arrays():
    return {
        "normalized_actions": np.arange(12, dtype=np.float32).reshape(2, 3, 2),
        "standard_actions": np.arange(12, dtype=np.float32).reshape(2, 3, 2),
        "noise": np.zeros((2, 3, 2), dtype=np.float32),
        "sample_identities": np.array([[1, 0], [1, 9]], dtype=np.int64),
        "valid_mask": np.array([[True, True, True], [True, False, False]]),
    }


def test_native_internal_noise_can_be_wider_than_output_actions(tmp_path):
    value = arrays()
    value["noise"] = np.zeros((2, 3, 32), dtype=np.float32)
    identity = {**IDENTITY, "noise_action_dim": 32}
    write_bundle(tmp_path / "native", value, identity)
    with np.load(tmp_path / "native/arrays.npz") as saved:
        assert saved["noise"].shape == (2, 3, 32)
    value["noise"] = value["noise"][:, :, :2]
    with pytest.raises(ValueError, match="registered full"):
        write_bundle(tmp_path / "truncated", value, identity)


@pytest.mark.parametrize("dimension", [True, 0, -1, 32.0, "32"])
def test_noise_dimension_must_be_an_explicit_positive_integer(tmp_path, dimension):
    with pytest.raises(ValueError, match="noise dimension"):
        write_bundle(
            tmp_path / "invalid", arrays(), {**IDENTITY, "noise_action_dim": dimension}
        )


def collect(path, *, changed=False):
    code = """
import sys
sys.path.insert(0, sys.argv[3])
import numpy as np
from rosetta_reality.vla.reload_evidence import HASH_IDENTITIES, write_bundle
a = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
if sys.argv[2] == 'changed':
    a[0, -1] = a[0, -1, ::-1].copy()
write_bundle(sys.argv[1], {
 'normalized_actions': a, 'standard_actions': a.copy(),
 'noise': np.zeros((2, 3, 2), dtype=np.float32),
 'sample_identities': np.array([[1, 0], [1, 9]], dtype=np.int64),
 'valid_mask': np.array([[True, True, True], [True, False, False]])
}, {**{key:'a'*64 for key in HASH_IDENTITIES}, 'sample_count':2, 'chunk_size':3,
    'normalized_action_dim':2, 'standard_action_dim':2})
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(path),
            "changed" if changed else "same",
            str(Path(__file__).resolve().parents[1] / "src"),
        ],
        check=True,
        timeout=30,
    )


def test_independent_full_chunks_pass_but_do_not_claim_model_resume(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    collect(a)
    collect(b)
    proof = compare_bundles(a, b, expected_identity=IDENTITY)
    assert proof["exact_tensor_equality"] is True
    assert proof["model_execution_proven_by_this_comparison"] is False
    assert proof["formal_resume_verified"] is False


def test_equal_means_and_first_actions_cannot_hide_last_slot_change(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    collect(a)
    collect(b, changed=True)
    with np.load(a / "arrays.npz") as x, np.load(b / "arrays.npz") as y:
        assert x["normalized_actions"].mean() == y["normalized_actions"].mean()
        assert np.array_equal(
            x["normalized_actions"][:, 0], y["normalized_actions"][:, 0]
        )
    proof = compare_bundles(a, b, expected_identity=IDENTITY)
    assert proof["status"] == "tensor_mismatch"
    assert proof["exact_tensor_equality"] is False


def test_same_process_is_not_independent_collection(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_bundle(a, arrays(), IDENTITY)
    write_bundle(b, arrays(), IDENTITY)
    with pytest.raises(ValueError, match="one process"):
        compare_bundles(a, b, expected_identity=IDENTITY)


def test_metric_only_manifest_is_not_tensor_evidence(tmp_path):
    for name in ("a", "b"):
        path = tmp_path / name
        path.mkdir()
        (path / "manifest.json").write_text(
            json.dumps({"metrics": {"mae": 0.1}, "exact_tensor_equality": True})
        )
    with pytest.raises(ValueError, match="Metrics-only"):
        compare_bundles(tmp_path / "a", tmp_path / "b", expected_identity=IDENTITY)


def test_array_checksum_and_expected_processor_identity_are_enforced(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    collect(a)
    collect(b)
    with pytest.raises(ValueError, match="identity"):
        compare_bundles(
            a, b, expected_identity={**IDENTITY, "processor_sha256": "b" * 64}
        )
    with (b / "arrays.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        compare_bundles(a, b, expected_identity=IDENTITY)


def test_partial_and_nonfinite_chunks_are_rejected(tmp_path):
    a = arrays()
    a.pop("noise")
    with pytest.raises(ValueError, match="Complete"):
        write_bundle(tmp_path / "missing", a, IDENTITY)
    a = arrays()
    a["normalized_actions"][0, -1, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        write_bundle(tmp_path / "nan", a, IDENTITY)
    a = arrays()
    for key in ("normalized_actions", "standard_actions", "noise", "valid_mask"):
        a[key] = a[key][:, :1]
    with pytest.raises(ValueError, match="registered full"):
        write_bundle(tmp_path / "first-only", a, IDENTITY)


def test_cli_preserves_negative_proof_and_returns_nonzero(tmp_path, monkeypatch):
    from rosetta_reality.experiment import file_sha256
    from scripts import verify_smolvla_reload_v2 as cli

    a, b = tmp_path / "a", tmp_path / "b"
    collect(a)
    collect(b, changed=True)
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(IDENTITY))
    output = tmp_path / "proof.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
            "--first",
            str(a),
            "--second",
            str(b),
            "--identity",
            str(identity),
            "--identity-sha256",
            file_sha256(identity),
            "--output",
            str(output),
        ],
    )
    assert cli.main() == 4
    assert json.loads(output.read_text())["exact_tensor_equality"] is False
    with pytest.raises(FileExistsError):
        cli.main()
