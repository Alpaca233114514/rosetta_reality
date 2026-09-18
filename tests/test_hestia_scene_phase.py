"""Counterexamples for geometry-only neighbors, censoring and bounded label reads."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow
import pyarrow.parquet as pq
import pytest

from scripts.diagnose_hestia_scene_phase import (
    event_scores,
    first_upcross,
    geometry_neighbors,
    read_phase_actions,
    trajectory_scores,
)


def test_geometry_ties_use_episode_id_and_never_development_donors():
    _, nearest, allowed = geometry_neighbors(
        np.zeros((5, 4)), [30, 10, 20, 50, 40], [0, 1, 2, 3], 2
    )
    assert nearest[0].tolist() == [1, 2]
    assert nearest[4].tolist() == [1, 2]
    assert all(4 not in donors for donors in allowed)
    assert all(i not in nearest[i] for i in range(4))


def test_development_labels_cannot_change_donors_or_any_prediction():
    rng = np.random.default_rng(2)
    coordinates = rng.uniform(0, 1, (7, 4))
    _, nearest, allowed = geometry_neighbors(coordinates, list(range(7)), list(range(6)), 3)
    y = rng.normal(size=(7, 10, 4))
    groups, rows, windows = (
        {"joint": [1, 3]},
        {"train": list(range(6)), "dev": [6]},
        {"all": [0, 10]},
    )
    before, pred = trajectory_scores(y, nearest, allowed, groups, rows, windows)
    y[6] += 100
    after, new = trajectory_scores(y, nearest, allowed, groups, rows, windows)
    np.testing.assert_array_equal(new, pred)
    assert before["train"] == after["train"]
    assert after["dev"]["all"]["joint"]["neighbor_mae"] > 90


def test_predictable_geometry_has_better_events_than_constant():
    coordinates = np.zeros((6, 4))
    coordinates[:, 0] = [0, 1, 2, 100, 101, 102]
    _, nearest, allowed = geometry_neighbors(coordinates, list(range(6)), list(range(6)), 2)
    result = event_scores([3, 4, 5, None, None, None], nearest, allowed, list(range(6)))
    assert result["neighbor_brier"] == 0
    assert result["constant_brier"] > 0
    assert result["neighbor_confusion"] == {"tp": 3, "fp": 0, "fn": 0, "tn": 3}


def test_onset_censoring_does_not_invent_end_frame_or_timing_coverage():
    assert first_upcross([[0, 0.3, 0.5, 0.1], [0, 0, 0, 0]], 0.5) == [2, None]
    with pytest.raises(ValueError, match="initially open"):
        first_upcross([[0.5, 0.2]], 0.5)
    result = event_scores(
        [None, None, 9],
        np.array([[1], [0], [1]]),
        [np.array([1, 2]), np.array([0, 2]), np.array([0, 1])],
        [2],
    )
    assert result["conditional_onset_mae"]["common_count"] == 0
    assert result["conditional_onset_mae"]["neighbor_frames"] is None
    assert result["observed_event_count"] == 1


def test_prefix_filter_excludes_hidden_and_future_state_before_materialization(
    tmp_path, monkeypatch
):
    (tmp_path / "data").mkdir()
    pq.write_table(
        pa.table(
            {
                "ep": [8, 8, 8, 31],
                "frame": [0, 1, 2, 0],
                "time": [0.0, 0.02, 0.04, 0.0],
                "action": [[0.1], [0.2], [99.0], [float("nan")]],
                "state": [[float("nan")]] * 4,
            }
        ),
        tmp_path / "data/rows.parquet",
    )
    cfg = SimpleNamespace(
        fields=SimpleNamespace(
            episode_index="ep", frame_index="frame", timestamp="time", action="action"
        )
    )
    real_dataset = arrow.dataset
    called = []

    class Checked:
        def to_table(self, **kwargs):
            assert kwargs["columns"] == ["ep", "frame", "time", "action"]
            assert "filter" in kwargs
            table = real_dataset(tmp_path / "data", format="parquet").to_table(**kwargs)
            assert table["ep"].to_pylist() == [8, 8]
            called.append(True)
            return table

    monkeypatch.setattr(arrow, "dataset", lambda *a, **kw: Checked())
    values, count = read_phase_actions(tmp_path, cfg, [8], [31], 2)
    np.testing.assert_allclose(values, [[[0.1], [0.2]]])
    assert count == 2 and called == [True]
    with pytest.raises(ValueError, match="hidden"):
        read_phase_actions(tmp_path, cfg, [31], [31], 2)
    assert called == [True]


def test_missing_or_timestamp_misaligned_prefix_fails(tmp_path):
    (tmp_path / "data").mkdir()
    cfg = SimpleNamespace(
        fields=SimpleNamespace(
            episode_index="ep", frame_index="frame", timestamp="time", action="action"
        )
    )
    pq.write_table(
        pa.table({"ep": [8, 8], "frame": [0, 1], "time": [0.0, 0.04], "action": [[0.1], [0.2]]}),
        tmp_path / "data/rows.parquet",
    )
    with pytest.raises(ValueError, match="Missing"):
        read_phase_actions(tmp_path, cfg, [8], [], 3)
    with pytest.raises(ValueError, match="Timestamp"):
        read_phase_actions(tmp_path, cfg, [8], [], 2)
