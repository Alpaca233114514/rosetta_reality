from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.diagnose_socket_action_horizon import position_basis, read_registered_rows


def test_fixed_polynomial_has_no_dataset_fitted_parameters():
    np.testing.assert_array_equal(position_basis([[320, 240]]), [[0.5, 0.5, 0.25, 0.25, 0.25]])


def test_arrow_filters_hidden_and_other_offsets(tmp_path):
    (tmp_path / "data").mkdir()
    pq.write_table(
        pa.table(
            {
                "ep": [49, 49, 49, 31],
                "frame": [0, 49, 1, 0],
                "time": [0.0, 0.98, 0.02, 0.0],
                "act": [[0.0], [1.0], [99.0], [float("nan")]],
                "state": [[0.0], [1.0], [99.0], [float("nan")]],
            }
        ),
        tmp_path / "data" / "test.parquet",
    )
    cfg = SimpleNamespace(
        fields=SimpleNamespace(
            episode_index="ep", frame_index="frame", timestamp="time", action="act", state="state"
        )
    )
    rows = read_registered_rows(tmp_path, cfg, [49], [31], [0, 49])
    assert set(rows) == {(49, 0), (49, 49)}
    with pytest.raises(ValueError, match="hidden"):
        read_registered_rows(tmp_path, cfg, [31], [31], [0, 49])


def test_timestamp_error_rejected(tmp_path):
    (tmp_path / "data").mkdir()
    pq.write_table(
        pa.table({"ep": [49], "frame": [49], "time": [1.0], "act": [[0.0]], "state": [[0.0]]}),
        tmp_path / "data" / "test.parquet",
    )
    cfg = SimpleNamespace(
        fields=SimpleNamespace(
            episode_index="ep", frame_index="frame", timestamp="time", action="act", state="state"
        )
    )
    with pytest.raises(ValueError, match="timestamp"):
        read_registered_rows(tmp_path, cfg, [49], [31], [49])
