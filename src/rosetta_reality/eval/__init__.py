"""Evaluation metrics and artifact checks."""

__all__ = ["action_metrics"]


def __getattr__(name):
    # File-only diagnostic validation must not initialize torch or a simulator.
    if name == "action_metrics":
        from rosetta_reality.eval.metrics import action_metrics

        return action_metrics
    raise AttributeError(name)
