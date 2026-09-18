"""Read-only, opt-in validation of the registered frame-zero cache boundary."""

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.data
def test_registered_nonhidden_frame_zero_rows_share_one_verified_cache():
    from rosetta_reality.vla.vision_diagnostics import load_frame_zero_context

    context = load_frame_zero_context(Path(__file__).resolve().parents[1], "non_hidden")
    assert len(context["train"]) == 40
    assert len(context["validation"]) == 5
    assert context["episodes"] == context["train"] + context["validation"]
    assert set(context["episodes"]).isdisjoint({31, 6, 1, 24, 5})
    assert context["actions"].shape == context["states"].shape == (45, 14)
    np.testing.assert_array_equal(
        context["states"], np.broadcast_to(context["states"][0], context["states"].shape)
    )
