"""Explicit optional integration worker; neither project's core imports the other."""

from rosetta_reality.diagnostics.torchlens_capture import FORWARD_SEED, SELECTED, execute_arm


def collect(case, run_id, output, result):
    # Only called in the collector's disposable child, never by the trainer or
    # Basin's read-only MCP service. Data returned to the parent stays JSON.
    from basin.torchlens_backend import capture_synthetic

    def capture(model, inputs):
        return capture_synthetic(model, inputs, run_id=run_id,
                                 selected_modules=SELECTED, forward_seed=FORWARD_SEED,
                                 output=output / "basin-raw")

    return execute_arm("basin-torchlens", case, run_id, result=result, capture_backend=capture)
