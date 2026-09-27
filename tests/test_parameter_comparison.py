import math

import pytest
import torch
from safetensors.torch import save_file

from scripts.compare_smolvla_parameters import compare_files, summarize, tensor_stats


def test_known_delta_chunk_boundaries():
    a = torch.tensor([0., 1., 2., -3.])
    b = torch.tensor([1., 1., 4., -3.])
    r = tensor_stats(a, b, block_size=2)
    assert r["changed_elements"] == 2
    assert r["delta_l2"] == pytest.approx(math.sqrt(5))
    assert r["relative_l2"] == pytest.approx(math.sqrt(5 / 14))
    assert r["cosine"] == pytest.approx(18 / math.sqrt(14 * 27))
    assert r["left"]["std_population"] == pytest.approx(math.sqrt(3.5))
    assert r == tensor_stats(a, b, block_size=100)


def test_zero_dtype_and_signed_zero():
    r = tensor_stats(torch.zeros(3), torch.ones(3))
    assert r["relative_l2"] is None and r["cosine"] is None
    r = tensor_stats(torch.ones(3), torch.ones(3, dtype=torch.bfloat16))
    assert r["numeric_equal"] and not r["byte_equal"]
    r = tensor_stats(torch.tensor([0.]), torch.tensor([-0.]))
    assert r["numeric_equal"] and not r["byte_equal"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite(value):
    with pytest.raises(ValueError, match="Nonfinite"):
        tensor_stats(torch.ones(1), torch.tensor([value]))


def test_schema_failure_and_all_keys(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    save_file({"x.weight": torch.ones(3), "x.bias": torch.zeros(1)}, a)
    save_file({"x.weight": torch.ones(3), "x.bias": torch.ones(1)}, b)
    rows = list(compare_files(a, b))
    summary = summarize(rows)["x"]
    assert len(rows) == 2 and summary["changed_elements"] == 1
    assert summary["relative_l2"] == pytest.approx(1 / math.sqrt(3))
    save_file({"y": torch.ones(3)}, b)
    with pytest.raises(ValueError, match="missing_right"):
        list(compare_files(a, b))
    with pytest.raises(ValueError, match="shape mismatch"):
        tensor_stats(torch.ones(2), torch.ones(3))
