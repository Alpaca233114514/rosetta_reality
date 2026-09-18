import numpy as np

from scripts.diagnose_hestia_value_energy import energy_parts


def test_orthogonal_global_token_and_scene_components():
    scene = np.array([-1.0, 1.0, -1.0, 1.0, 2.0])[:, None, None]
    token = np.array([-2.0, 2.0])[None, :, None]
    result = energy_parts(3 + scene + token, 4)
    assert result["train40"]["global_offset"] == 9
    assert result["train40"]["shared_token_pattern"] == 4
    assert result["train40"]["scene_residual"] == 1
    assert result["train40"]["cross"] == 0
    assert result["dev5"]["cross"] == 12
