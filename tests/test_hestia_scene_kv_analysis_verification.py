import numpy as np
import pytest

from scripts.verify_hestia_scene_kv_analysis import check_summary


def test_independent_summary_includes_scene_removal_and_rejects_corruption():
    record = {
        "mean": 4.0,
        "by_noise": [2.0, 6.0],
        "by_episode": [3.0, 5.0],
        "by_noise_episode": [[1.0, 3.0], [5.0, 7.0]],
        "leave_one_episode_out_by_noise": [[3.0, 1.0], [7.0, 5.0]],
    }
    assert check_summary(record, np.array([[1, 3], [5, 7]])) == 13
    record["leave_one_episode_out_by_noise"][0][0] = 1.0
    with pytest.raises(AssertionError):
        check_summary(record, np.array([[1, 3], [5, 7]]))
