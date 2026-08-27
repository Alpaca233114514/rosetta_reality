"""Bounded first-deviation trace helpers."""

from scripts.diagnose_smolvla_aster_trajectory import _new_crossings


def test_new_crossings_records_each_threshold_once() -> None:
    crossings = {key: None for key in ("0.005", "0.01", "0.025", "0.05", "0.1")}

    assert not _new_crossings(0.004, 0, crossings)
    assert _new_crossings(0.026, 1, crossings)
    assert crossings == {
        "0.005": 1,
        "0.01": 1,
        "0.025": 1,
        "0.05": None,
        "0.1": None,
    }
    assert not _new_crossings(0.02, 2, crossings)
    assert _new_crossings(0.2, 3, crossings)
    assert crossings["0.05"] == 3
    assert crossings["0.1"] == 3
