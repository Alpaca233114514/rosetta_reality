from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from rosetta_reality.eval.diagnostics import (
    action_error_summary,
    cross_episode_shuffle_indices,
    nearest_cross_episode_indices,
    paired_cosine_distances,
    paired_l2_distances,
    pairwise_cosine_summary,
    pairwise_l2_summary,
    pearson_correlation,
    phase_labels,
    scalar_summary,
)
from rosetta_reality.experiment import file_sha256, stable_hash
from scripts.diagnose_m2 import (
    _alignment_metrics,
    _load_initial_images,
    _replacement_batch,
    _seed_map,
)


def test_pairwise_summaries_use_unique_pairs() -> None:
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    l2 = pairwise_l2_summary(values)
    cosine = pairwise_cosine_summary(values)

    assert l2["pairs"] == 3
    assert l2["maximum"] == pytest.approx(2**0.5)
    assert cosine["pairs"] == 3
    assert cosine["maximum"] == pytest.approx(1.0)


def test_action_error_summary_separates_first_action_from_chunk() -> None:
    target = torch.zeros(1, 2, 1)
    predicted = torch.tensor([[[1.0], [3.0]]])

    result = action_error_summary(predicted, target)

    assert result["first_action_mae"] == pytest.approx(1.0)
    assert result["chunk_mae"] == pytest.approx(2.0)


def test_phase_labels_use_half_open_boundaries() -> None:
    frames = torch.tensor([0, 99, 100, 499, 500])

    assert phase_labels(frames, (100, 200, 300, 400, 500)) == [
        "000-099",
        "000-099",
        "100-199",
        "400-499",
        "500+",
    ]


@pytest.mark.parametrize("boundaries", [(), (0,), (100, 100), (200, 100)])
def test_phase_labels_reject_invalid_boundaries(boundaries: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        phase_labels(torch.tensor([0]), boundaries)


def test_image_alignment_prefers_identical_images() -> None:
    reference = torch.zeros(3, 8, 8)
    changed = reference.clone()
    changed[:, 2:6, 2:6] = 1.0

    identical = _alignment_metrics(reference, reference)
    different = _alignment_metrics(reference, changed)

    assert identical == {"pixel_mae": 0.0, "pixel_rmse": 0.0, "pooled_4x4_mae": 0.0}
    assert different["pixel_mae"] > identical["pixel_mae"]
    assert different["pooled_4x4_mae"] > identical["pooled_4x4_mae"]


def test_seed_map_parses_and_rejects_duplicates() -> None:
    assert _seed_map(["2:10", "3:11"]) == {2: 10, 3: 11}
    with pytest.raises(argparse.ArgumentTypeError, match="Duplicate seed-map episode"):
        _seed_map(["2:10", "2:12"])


def test_initial_image_loader_rejects_undeclared_file_before_tensor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "initial-images"
    artifact.mkdir()
    validation_scope = {
        "experiment_id": "experiment",
        "experiment_config_sha256": "a" * 64,
        "split": "validation",
        "episodes": [7],
        "test_split_opened": False,
    }
    identity = {
        "schema_version": 2,
        "dataset_repo_id": "dataset",
        "dataset_revision": "b" * 40,
        "dataset_manifest_sha256": "c" * 64,
        "episodes": [7],
        "camera": "top",
        "decoder": "decoder",
        "validation_scope": validation_scope,
    }
    identity_hash = stable_hash(identity)
    shard = artifact / "episode-000007.pt"
    torch.save(
        {
            "schema_version": 1,
            "identity_hash": identity_hash,
            "episode": 7,
            "image": torch.zeros(3, 2, 2),
        },
        shard,
    )
    manifest = {
        **identity,
        "identity_hash": identity_hash,
        "files": {
            "7": {
                "path": shard.name,
                "sha256": file_sha256(shard),
                "shape": [3, 2, 2],
            }
        },
    }
    (artifact / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (artifact / "episode-000031.pt").write_bytes(b"undeclared hidden-test shard")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("tensor load occurred before file audit"),
    )

    with pytest.raises(ValueError, match="undeclared files"):
        _load_initial_images(
            artifact,
            expected_manifest_sha256="c" * 64,
            expected_revision="b" * 40,
            episodes=(7,),
        )


def test_cross_episode_shuffle_preserves_frame_and_is_deterministic() -> None:
    episodes = torch.tensor([0, 1, 2, 0, 1, 2])
    frames = torch.tensor([0, 0, 0, 5, 5, 5])

    first = cross_episode_shuffle_indices(
        episodes,
        frame_indices=frames,
        seed=20260809,
    )
    second = cross_episode_shuffle_indices(
        episodes,
        frame_indices=frames,
        seed=20260809,
    )

    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(6))
    assert episodes.ne(episodes[first]).all()
    assert frames.eq(frames[first]).all()


def test_cross_episode_shuffle_rejects_singleton_frame_group() -> None:
    with pytest.raises(ValueError, match="fewer than two samples"):
        cross_episode_shuffle_indices(
            torch.tensor([0, 1, 2]),
            frame_indices=torch.tensor([0, 0, 5]),
        )


def test_nearest_cross_episode_indices_excludes_same_episode() -> None:
    indices, distances = nearest_cross_episode_indices(
        torch.tensor([[0.0], [9.0]]),
        torch.tensor([0, 1]),
        torch.tensor([[0.1], [1.0], [8.5]]),
        torch.tensor([0, 2, 2]),
        batch_size=1,
    )

    assert indices.tolist() == [1, 2]
    assert distances.tolist() == pytest.approx([1.0, 0.5])


def test_paired_and_scalar_diagnostics_are_aligned() -> None:
    left = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    right = torch.tensor([[0.0, 1.0], [0.0, 1.0], [2.0, 2.0]])

    l2 = paired_l2_distances(left, right)
    cosine = paired_cosine_distances(left, right)
    summary = scalar_summary(l2)

    assert l2.tolist() == pytest.approx([2**0.5, 0.0, 2**0.5])
    assert cosine.tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert summary["samples"] == 3
    assert summary["maximum"] == pytest.approx(2**0.5)
    assert pearson_correlation(l2, l2) == pytest.approx(1.0)
    assert pearson_correlation(torch.ones(3), l2) is None


def test_replacement_batch_supports_constant_and_sample_aligned_inputs() -> None:
    original = torch.zeros(2, 3)
    constant = torch.tensor([[1.0, 2.0, 3.0]])
    aligned = torch.arange(15, dtype=torch.float32).reshape(5, 3)

    assert torch.equal(
        _replacement_batch(
            constant,
            original=original,
            start=1,
            stop=3,
            total=5,
            name="feature",
        ),
        constant.expand(2, -1),
    )
    assert torch.equal(
        _replacement_batch(
            aligned,
            original=original,
            start=1,
            stop=3,
            total=5,
            name="feature",
        ),
        aligned[1:3],
    )
    with pytest.raises(ValueError, match="one or 5 samples"):
        _replacement_batch(
            torch.zeros(2, 3),
            original=original,
            start=1,
            stop=3,
            total=5,
            name="feature",
        )
