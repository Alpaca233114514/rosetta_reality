"""Synthetic evidence for intervention identity and camera ingress failures."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow
import pyarrow.parquet as parquet
import pytest
import torch

from rosetta_reality.vla.visual_grounding import (
    FRAME_OFFSETS,
    TRAIN_EPISODES,
    intervention_sample,
    read_temporal_rows,
    summarize_offset,
    validate_image_ingress,
)


def test_intervention_preserves_destination_state_and_has_no_action_label():
    destination = {
        "rgb": torch.zeros(3, 4, 4, dtype=torch.uint8),
        "observation.state": torch.tensor([1.0, 2.0]),
        "task": "destination",
        "action": torch.tensor([99.0]),
    }
    source = {
        "rgb": torch.full((3, 4, 4), 255, dtype=torch.uint8),
        "observation.state": torch.tensor([8.0, 9.0]),
        "task": "source",
    }
    result = intervention_sample(destination, source, "rgb")
    assert result["task"] == "destination"
    assert "action" not in result
    assert torch.equal(result["observation.state"], destination["observation.state"])
    assert torch.equal(result["rgb"], torch.ones(3, 4, 4))
    result["rgb"].zero_()
    result["observation.state"].zero_()
    assert source["rgb"].min() == 255
    assert destination["observation.state"].min() == 1


@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.1])
def test_invalid_rgb_cannot_become_evidence(bad):
    sample = {
        "rgb": torch.full((3, 4, 4), bad),
        "observation.state": torch.zeros(2),
        "task": "insert",
    }
    with pytest.raises(ValueError, match="RGB"):
        intervention_sample(sample, sample, "rgb")


def test_ingress_rechecks_masks_after_a_valid_first_call():
    masks = [torch.tensor([True]), torch.tensor([False]), torch.tensor([False])]
    policy = SimpleNamespace(
        config=SimpleNamespace(empty_cameras=2),
        prepare_images=lambda batch: ([torch.zeros(1, 3, 4, 4)] * 3, masks),
    )
    assert validate_image_ingress(policy, {})["camera_masks"] == [True, False, False]
    masks[0][0] = False
    with pytest.raises(ValueError, match="real camera is masked"):
        validate_image_ingress(policy, {})
    masks[0][0], masks[2][0] = True, True
    with pytest.raises(ValueError, match="placeholder is unmasked"):
        validate_image_ingress(policy, {})


def test_temporal_scan_filters_hidden_and_unregistered_frames_before_materialization(
    tmp_path, monkeypatch
):
    fields = SimpleNamespace(
        episode_index="episode_index", frame_index="frame_index", state="state", action="action"
    )
    config = SimpleNamespace(fields=fields, episodes=(*TRAIN_EPISODES, 31))
    records = [
        {"episode_index": ep, "frame_index": offset, "state": [0.0], "action": [0.0]}
        for ep in TRAIN_EPISODES
        for offset in FRAME_OFFSETS
    ]
    records.extend(
        [
            {
                "episode_index": 31,
                "frame_index": 0,
                "state": [float("nan")],
                "action": [float("nan")],
            },
            {
                "episode_index": 49,
                "frame_index": 1,
                "state": [float("nan")],
                "action": [float("nan")],
            },
        ]
    )
    (tmp_path / "data").mkdir()
    parquet.write_table(pa.Table.from_pylist(records), tmp_path / "data/rows.parquet")
    original = arrow.dataset
    scanned = []

    class Scanner:
        def to_table(self, **kwargs):
            table = original(tmp_path / "data", format="parquet").to_table(**kwargs)
            assert table.num_rows == 20
            assert 31 not in table["episode_index"].to_pylist()
            assert 1 not in table["frame_index"].to_pylist()
            scanned.append(True)
            return table

    monkeypatch.setattr(arrow, "dataset", lambda *args, **kwargs: Scanner())
    assert len(read_temporal_rows(tmp_path, config, list(TRAIN_EPISODES), [31])) == 20
    assert scanned == [True]
    with pytest.raises(ValueError, match="train-only"):
        read_temporal_rows(tmp_path, config, list(TRAIN_EPISODES), [49])
    assert scanned == [True]


def test_image_sensitive_but_wrong_policy_does_not_pass_alignment():
    truth = np.arange(10.0).reshape(5, 2)
    dims = [{"unit": "radian"}, {"unit": "normalized"}]
    result = summarize_offset(truth[::-1], truth, truth, truth, dims)
    assert result["image_output_shift"] > 0
    assert result["paired_mae_gain"] < 0
    assert result["state_persistence_mae"] == 0
    assert set(result["by_unit"]) == {"radian", "normalized"}
    assert result["state_shuffle_degenerate"] is False


def test_constant_state_is_reported_as_degenerate_without_dividing_by_zero():
    truth = np.zeros((5, 2))
    result = summarize_offset(truth, truth, truth, truth, [{"unit": "radian"}] * 2)
    assert result["paired_mae_gain"] == 0
    assert result["state_shuffle_degenerate"] is True
    assert result["output_target_correlation_per_dimension"] == [None, None]
