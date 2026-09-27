"""Optional dual-axis sink for the existing paired reproducibility collector."""

from contextlib import contextmanager

from rosetta_reality.diagnostics.dual_axis import Window, digest
from rosetta_reality.diagnostics.dual_axis_torch import module_capture, tensor_tree


class RolloutSink:
    def __init__(
        self, writer, *, checkpoint, update, episode, identity, modules, phase_resolver=None
    ):
        if not checkpoint or any(
            not identity.get(k) for k in ("processor", "action_contract", "runtime")
        ):
            raise ValueError(
                "Rollout requires sealed checkpoint/processor/contract/runtime identity"
            )
        self.writer, self.checkpoint, self.update, self.episode = (
            writer,
            checkpoint,
            update,
            episode,
        )
        self.identity, self.modules = identity, modules
        self.window = Window()
        self.step = 0
        self.previous_phase = None
        self.dense_until = -1
        self.phase_resolver = phase_resolver

    def emit(self, step, boundary, payload, *, identity=None, reasons=("every_control_step",)):
        self.writer.event(
            {
                "axis": "execution",
                "checkpoint": self.checkpoint,
                "update": self.update,
                "episode": self.episode,
                "rollout_step": step,
                "boundary": boundary,
            },
            {"payload": tensor_tree(payload, self.writer)},
            identity={**self.identity, **(identity or {})},
            reasons=reasons,
        )

    def finish(self, result):
        if not self.writer.closed:
            self.writer.seal(result["status"], error=result.get("error_type"))

    def record(self, name, value):
        if name == "reset":
            self.emit(0, "reset", value)
        elif name.startswith("predictions/"):
            step = int(name.split("/")[1])
            identity = {
                "input": digest(tensor_tree(value["observation"], self.writer)),
                "noise": digest(tensor_tree(value["noise"], self.writer)),
            }
            for boundary, key in (
                ("raw_input", "observation"),
                ("processed_input", "batch"),
                ("noise", "noise"),
                ("model_output", "normalized"),
                ("decoded_action", "decoded"),
                ("projected_action", "projected"),
            ):
                self.emit(step, boundary, value[key], identity=identity)
            self.step = step
        elif name.startswith("steps/"):
            _, text, side = name.split("/")
            step = int(text)
            self.emit(step, "executed_action" if side == "before" else "physics", value)
            if side == "after":
                task = value.get("task_geometry") or {}
                phase = (
                    self.phase_resolver(value)
                    if self.phase_resolver
                    else (value.get("reward"), value.get("done"), digest(task.get("contacts")))
                )
                if self.previous_phase is not None and phase != self.previous_phase:
                    self.dense_until = step + 8
                    window = self.window.snapshot()
                    records = window.pop("records")
                    window["records"] = [
                        self.writer.write(
                            f"windows/event-{self.writer.sequence:08d}-{i:04d}.json", row
                        )
                        for i, row in enumerate(records)
                    ]
                    self.writer.event(
                        {
                            "axis": "execution",
                            "checkpoint": self.checkpoint,
                            "update": self.update,
                            "episode": self.episode,
                            "rollout_step": step,
                        },
                        {
                            "phase_transition": [self.previous_phase, phase],
                            "preceding_window": window,
                        },
                        reasons=["phase_transition"],
                    )
                self.previous_phase = phase
                self.step = step + 1

    @contextmanager
    def capture(self, policy):
        with module_capture(policy, self.modules) as rows:
            yield
        self.window.add(self.step, {"rollout_step": self.step, "modules": rows})
        if self.step == 0 or self.step <= self.dense_until:
            self.emit(self.step, "model_internal", rows, reasons=["initial_or_dense_window"])
