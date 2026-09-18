import numpy as np

from scripts.diagnose_chunk_time_shift import shift_chunk
from scripts.diagnose_hestia_timing_bound import oracle_align


def test_oracle_recovers_opposite_row_shifts_but_not_a_shared_predictor():
    x = np.tile(np.arange(8, dtype=float)[None, :, None], (2, 1, 1))
    y = np.concatenate([shift_chunk(x[:1], -2), shift_chunk(x[1:], 2)])
    aligned, lags, _ = oracle_align(x, y, list(range(-3, 4)))
    np.testing.assert_array_equal(aligned, y)
    np.testing.assert_array_equal(lags, [-2, 2])


def test_constant_sequence_tie_chooses_zero_and_preserves_all_slots():
    x = np.ones((2, 8, 3))
    aligned, lags, scores = oracle_align(x, x + 2, [-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(lags, [0, 0])
    np.testing.assert_array_equal(aligned, x)
    np.testing.assert_array_equal(scores, np.full((5, 2), 2.0))
