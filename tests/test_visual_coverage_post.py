"""Distinguish consumed samples from the one batch prefetched by Accelerate."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "coverage_post", ROOT / "scripts/run_visual_coverage_post.py"
)
POST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POST)


def test_accepts_exact_consumed_and_prefetched_order():
    assert POST.verify_sample_observation(
        list(range(1028)), list(range(1040)), consumed=1024, batch_size=4
    ) == {"consumed": 1024, "prefetched_unused": 4, "exact_order": True}


@pytest.mark.parametrize("count", [1020, 1024, 1027, 1029, 1032])
def test_rejects_missing_or_excess_samples(count):
    with pytest.raises(ValueError, match="Sampler observation differs"):
        POST.verify_sample_observation(
            list(range(count)), list(range(1040)), consumed=1024, batch_size=4
        )


@pytest.mark.parametrize("index", [0, 511, 1023, 1024, 1027])
def test_rejects_wrong_order_including_unused_prefetch(index):
    observed = list(range(1028))
    observed[index] = -1
    with pytest.raises(ValueError, match="Sampler observation differs"):
        POST.verify_sample_observation(
            observed, list(range(1040)), consumed=1024, batch_size=4
        )


def test_real_accelerate_prefetch_with_epoch_boundaries():
    from accelerate.data_loader import DataLoaderShard

    emitted = []

    class Sampler:
        def __iter__(self):
            for index in range(40):
                emitted.append(index)
                yield index

        def __len__(self):
            return 40

    loader = DataLoaderShard(list(range(40)), batch_size=4, sampler=Sampler(), num_workers=0)

    def batches():
        while True:
            yield from loader

    iterator = batches()
    consumed = []
    for _ in range(256):
        consumed.extend(next(iterator).tolist())
    expected = list(range(40)) * 26
    assert consumed == expected[:1024]
    assert POST.verify_sample_observation(
        emitted, expected, consumed=1024, batch_size=4
    )["prefetched_unused"] == 4
