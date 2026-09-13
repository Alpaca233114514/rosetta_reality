"""Independent numeric factor-analysis contracts."""

import numpy as np
import torch

from scripts.analyze_hestia_k_factor import paired_effect
from scripts.analyze_hestia_k_geometry import bf16_nearest


def test_paired_loss_identity_includes_nonlinear_movement_term():
    base = np.ones((4, 5, 50, 6)) * 3
    target = np.ones((5, 50, 6))
    treatment = base - 1
    result = paired_effect(base, treatment, target)
    assert result["mse_delta_by_noise"] == [-3.0] * 4
    assert result["movement_squared_by_noise"] == [1.0] * 4
    assert result["minus_twice_residual_alignment_by_noise"] == [-4.0] * 4


def test_numpy_bf16_rounding_covers_ties_signs_and_double_rounding():
    x = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1.00390625,
            1.01171875,
            1.00390625 - 1e-10,
            1.00390625 + 1e-10,
            -1.00390625 + 1e-10,
            -1.00390625 - 1e-10,
        ]
    )
    expected = np.array(
        [0.0, -0.0, 1.0, -1.0, 1.0, 1.015625, 1.0, 1.0078125, -1.0, -1.0078125], dtype=np.float32
    )
    actual = bf16_nearest(x)
    assert np.array_equal(actual, expected)
    assert np.array_equal(np.signbit(actual), np.signbit(expected))
    assert actual[5] == 1.015625 and actual[7] == 1.0078125
    # This local PyTorch cast passes through float32. It must not define the
    # independent float64 rounding oracle at values just above a halfway point.
    via_float32 = torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).float().numpy()
    assert via_float32[7] == 1.0 and actual[7] != via_float32[7]
