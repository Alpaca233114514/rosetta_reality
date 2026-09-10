"""Observer wiring must cover native feature teardown and incomplete launches."""

import json
from types import SimpleNamespace

import pytest
from rosetta_reality.vla.training.observed_launch import run_observed_launch


def fixture():
    def cycle(loader):
        yield from loader

    optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}], step=lambda: None)

    def update(_metrics, _policy, _batch, optimizer):
        optimizer.step()

    return SimpleNamespace(cycle=cycle, update_policy=update), optimizer


@pytest.mark.parametrize("failure", [None, "incomplete", "teardown", "exit"])
def test_guarded_launcher_result_and_original_error_are_preserved(tmp_path, failure):
    module, optimizer = fixture()
    originals = module.cycle, module.update_policy

    def launch():
        observed_update = module.update_policy
        module.update_policy = lambda *args: observed_update(*args)
        try:
            if failure != "incomplete":
                batch = next(module.cycle([{"episode_index": [3, 5], "frame_index": [0, 0]}]))
                module.update_policy(None, None, batch, optimizer)
            if failure == "teardown":
                raise ValueError("original feature teardown error")
            return 1 if failure == "exit" else 0
        finally:
            module.update_policy = observed_update

    def run():
        return run_observed_launch(
            module,
            launch,
            expected_samples=[(3, 0), (5, 0)],
            batch_size=2,
            output=tmp_path / "evidence",
        )

    if failure:
        expected = ValueError if failure == "teardown" else RuntimeError
        with pytest.raises(expected):
            run()
    else:
        assert run() == 0
    report = json.loads((tmp_path / "evidence/result.json").read_text())
    assert report["status"] == ("failed" if failure else "passed")
    assert report["observation"]["completed_updates"] == int(failure != "incomplete")
    assert (module.cycle, module.update_policy) == originals
    with pytest.raises(FileExistsError):
        run()
