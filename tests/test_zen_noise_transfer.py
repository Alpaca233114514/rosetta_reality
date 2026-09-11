import torch

from scripts.diagnose_zen_noise_transfer import make_noise


def test_fixed_noise_does_not_consume_global_rng():
    before = torch.get_rng_state().clone()
    a = make_noise((1, 50, 32), 20260905)
    b = make_noise((1, 50, 32), 20260905)
    assert torch.equal(a, b)
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.equal(a, make_noise((1, 50, 32), 20260906))


def test_zero_noise_shape_and_independent_storage():
    a = make_noise((1, 50, 32), None)
    assert a.shape == (1, 50, 32) and a.dtype == torch.float32
    assert not torch.count_nonzero(a)
    a.add_(1)
    assert not torch.count_nonzero(make_noise((1, 50, 32), None))
