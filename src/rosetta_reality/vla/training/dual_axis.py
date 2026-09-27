"""Opt-in observation of the pinned native update; no replacement training loop."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from rosetta_reality.diagnostics.dual_axis import (
    DeviationMonitor,
    EvidenceWriter,
    Sampling,
    Window,
    digest,
    relative,
)
from rosetta_reality.diagnostics.dual_axis_torch import (
    detached_tree,
    gradient_summary,
    isolated_probe,
    module_capture,
    native_methods,
    parameter_samples,
    sampled_update,
    snapshot,
    tensor_tree,
    tree_identity,
)


def runtime_identity():
    """Small actual runtime fingerprint for matching a separately measured admission."""
    import platform
    from importlib.metadata import version

    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "lerobot": version("lerobot"),
        "accelerate": version("accelerate"),
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


def validate_registration(plan):
    """Drafts and incomplete registrations cannot enter a live feature."""
    from rosetta_reality.diagnostics.dual_axis import integer

    if plan.get("status") != "registered" or plan.get("model_execution_authorized") is not True:
        raise ValueError("Diagnostic execution is not registered/authorized")
    integer(plan.get("total_updates"), "total_updates", 1)
    if not plan.get("modules") or not plan.get("probe_bank") or not plan.get("action_groups"):
        raise ValueError("Registration lacks module, sample or action identities")
    if (
        not isinstance(plan.get("deadline_epoch"), (float, int))
        or time.time() >= plan["deadline_epoch"]
    ):
        raise ValueError("Missing or expired diagnostic deadline")
    if plan.get("runtime_sha256") != digest(runtime_identity()):
        raise ValueError("Diagnostic runtime differs from registered admission")
    if not 0 < plan.get("resource_limits", {}).get("rss_bytes", 0) <= 16 * 1024**3:
        raise ValueError("Explicit memory limits required")
    for sample in plan["panels"]["extended"]:
        if sample["episode"] in plan["panels"]["hidden"]:
            raise ValueError("Hidden diagnostic sample")
        row = plan["probe_bank"].get(sample["sample_id"])
        if not row or any(
            not row.get("identity", {}).get(k)
            for k in ("input", "processor", "action_contract", "runtime")
        ):
            raise ValueError("Probe input/processor/contract/runtime identity missing")


def runtime_guard(plan):
    import resource

    import torch

    if time.time() >= plan["deadline_epoch"]:
        raise RuntimeError("Diagnostic deadline exceeded")
    limits = plan["resource_limits"]
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["rss_bytes"]:
        raise MemoryError("Diagnostic RSS limit exceeded")
    if torch.cuda.is_initialized():
        if (
            torch.cuda.memory_allocated() > limits["cuda_allocated_bytes"]
            or torch.cuda.memory_reserved() > limits["cuda_reserved_bytes"]
        ):
            raise MemoryError("Diagnostic CUDA limit exceeded")


class Collector:
    """Explicit callbacks supply sealed fixed inputs and processor-aware prediction."""

    def __init__(self, output, plan, *, probes, guard=lambda: None):
        self.plan = plan
        self.writer = EvidenceWriter(output, plan, maximum_bytes=plan["maximum_bytes"])
        self.sampling = Sampling(plan["schedule"], plan["total_updates"])
        self.window = Window(maximum_bytes=plan.get("ring_bytes", 256 * 1024**2))
        self.monitor = DeviationMonitor()
        self.probes, self.guard = probes, guard
        self.attempt = self.update = self.exposures = 0
        self.raw_batch = None
        self.failed = False
        self.epoch = None

    def coords(self, axis="training", **kwargs):
        return {
            "axis": axis,
            "attempt": self.attempt,
            "update": self.update,
            "exposures": self.exposures,
            "epoch": self.epoch,
            **kwargs,
        }

    def fixed(self, policy, optimizer):
        node = self.sampling.due(self.update)
        if node is None:
            return
        self.guard()
        if node["all_noises"]:
            saved = snapshot(policy, optimizer, self.writer)
            ref = self.writer.write(f"snapshots/update-{self.update:08d}.json", saved)
            self.writer.event(self.coords("snapshot"), {"snapshot": ref}, reasons=node["reasons"])
        # Callback owns registered samples/noises only. It must never advance the training loader.
        with isolated_probe(policy, state_objects=self.probes.state_objects):
            for sample, noise, payload, identity in self.probes(policy, node):
                self.guard()
                if sample["episode"] in self.plan["panels"]["hidden"]:
                    raise ValueError("Hidden episode in diagnostic inputs")
                coords = self.coords(
                    "probe",
                    sample_id=sample["sample_id"],
                    episode=sample["episode"],
                    frame=sample["frame"],
                    noise_id=noise["name"],
                    boundary="model_output",
                )
                metrics = payload.pop("metrics", [])
                event = self.writer.event(
                    coords,
                    {"payload": tensor_tree(payload, self.writer), "metrics": metrics},
                    identity=identity,
                    reasons=node["reasons"],
                )
                for metric in metrics:
                    if metric.get("category") != "behavior" or metric.get("aa_floor") is None:
                        continue
                    key = digest(
                        [
                            sample["sample_id"],
                            noise["name"],
                            metric["name"],
                            metric.get("group"),
                            metric.get("window"),
                            identity,
                        ]
                    )
                    alarm = self.monitor.observe(
                        key,
                        self.update,
                        metric["value"],
                        metric["aa_floor"],
                        regular="regular" in node["reasons"],
                        split=sample["split"],
                    )
                    if alarm and alarm["state"] == "candidate":
                        self.trigger({"alarm": alarm, "probe_sequence": event["sequence"]})

    def trigger(self, reason):
        self.sampling.trigger(self.update)
        window = self.window.snapshot()
        rows = window.pop("records")
        window["records"] = [
            self.writer.write(f"windows/event-{self.writer.sequence:08d}-{i:04d}.json", row)
            for i, row in enumerate(rows)
        ]
        self.writer.event(
            self.coords(), {"trigger": reason, "preceding_window": window}, reasons=["anomaly"]
        )

    def finish(self, error=None):
        error = error or ("NativeUpdateFailed" if self.failed else None)
        complete = error is None and self.update == self.plan["total_updates"]
        self.writer.event(
            self.coords(),
            {
                "completed_updates": self.update,
                "attempts": self.attempt,
                "expected_updates": self.plan["total_updates"],
            },
            status="observed" if complete else "failed",
            reasons=["completion"],
        )
        return self.writer.seal(
            "complete" if complete else "incomplete",
            error=error or (None if complete else "IncompleteUpdates"),
        )


@contextmanager
def observe_dual_axis(module, collector):
    """Wrap native cycle/update/clip/optimizer, preserving failed versus skipped calls."""
    if getattr(module, "_rosetta_dual_axis_active", False):
        raise RuntimeError("Concurrent dual-axis observation is unsupported")
    original_cycle, original_update = module.cycle, module.update_policy

    @wraps(original_cycle)
    def cycle(*args, **kwargs):
        for batch in original_cycle(*args, **kwargs):
            # Raw ingress is copied to bounded evidence only on actual consumption.
            collector.raw_batch = detached_tree(batch)
            yield batch

    @wraps(original_update)
    def update(*args, **kwargs):
        def arg(name, index):
            return kwargs.get(name, args[index] if len(args) > index else None)

        policy, batch, optimizer = arg("policy", 1), arg("batch", 2), arg("optimizer", 3)
        accelerator = arg("accelerator", 5)
        if policy is None or optimizer is None or accelerator is None:
            raise ValueError("Native update binding incomplete")
        collector.guard()
        if collector.update == 0 and collector.attempt == 0:
            collector.fixed(policy, optimizer)
        collector.attempt += 1
        began = time.perf_counter()
        before = parameter_samples(policy)
        pre = post = None
        attempted = completed = False
        raw = collector.raw_batch if collector.raw_batch is not None else batch
        if "epoch" in raw:
            epochs = raw["epoch"].detach().cpu().reshape(-1).tolist()
            if epochs and len(set(epochs)) == 1 and type(epochs[0]) is int:
                collector.epoch = epochs[0]
        identities = {
            key: tensor_tree(raw[key], collector.writer)
            for key in ("episode_index", "frame_index")
            if key in raw
        }
        detailed = (
            collector.update + 1 in collector.sampling.nodes
            or collector.update + 1 <= collector.sampling.dense_until
        )
        ingress = (
            tensor_tree({"raw": raw, "processed": batch}, collector.writer)
            if detailed
            else tree_identity({"raw": raw, "processed": batch})
        )
        rates = [float(g["lr"]) for g in optimizer.param_groups]
        missing = object()
        saved_step = vars(optimizer).get("step", missing)
        original_step = optimizer.step
        saved_clip = vars(accelerator).get("clip_grad_norm_", missing)
        original_clip = accelerator.clip_grad_norm_

        def clip(*a, **k):
            nonlocal pre, post
            pre = gradient_summary(policy)
            result = original_clip(*a, **k)
            post = gradient_summary(policy)
            return result

        def step(*a, **k):
            nonlocal attempted, completed, pre, post
            if attempted:
                raise RuntimeError("Multiple optimizer calls in one update")
            attempted = True
            if pre is None:
                pre = post = gradient_summary(policy)
            result = original_step(*a, **k)
            completed = not getattr(optimizer, "step_was_skipped", False)
            state = getattr(optimizer, "gradient_state", None)
            if state is not None and not state.sync_gradients:
                completed = False
            if completed:
                collector.update += 1
            return result

        optimizer.step, accelerator.clip_grad_norm_ = step, clip
        error = None
        modules = []
        native = []
        result = None
        try:
            with (
                module_capture(policy, collector.plan["modules"]) as modules,
                native_methods(policy, collector.writer, full=detailed) as native,
            ):
                result = original_update(*args, **kwargs)
            if not attempted:
                raise RuntimeError("Native update omitted optimizer call")
            if completed:
                ids = raw.get("episode_index")
                if ids is None or not hasattr(ids, "numel"):
                    raise ValueError("Actual sample identities required")
                collector.exposures += ids.numel()
            return result
        except BaseException as exc:
            error = type(exc).__name__
            collector.failed = True
            raise
        finally:
            for obj, name, old, installed in (
                (optimizer, "step", saved_step, step),
                (accelerator, "clip_grad_norm_", saved_clip, clip),
            ):
                if getattr(obj, name) is not installed:
                    raise RuntimeError("Diagnostic hook changed during native update")
                if old is missing:
                    delattr(obj, name)
                else:
                    setattr(obj, name, old)
            after = parameter_samples(policy)
            values = {
                "optimizer_attempted": attempted,
                "optimizer_completed": completed,
                "sample_identity": identities,
                "inputs": ingress,
                "raw_scope": "delivered_loader_batch_before_processor_not_original_disk_bytes",
                "gradient_preclip": pre,
                "gradient_postclip": post,
                "parameter_update": sampled_update(before, after),
                "lr_used": rates,
                "lr_after": [float(g["lr"]) for g in optimizer.param_groups],
                "native_boundaries": native,
                "loss": float(getattr(result[0].loss, "val", result[0].loss))
                if result is not None and hasattr(result[0], "loss")
                else None,
                "loss_parts": tensor_tree(result[1], collector.writer)
                if result is not None
                else None,
                "update_seconds": time.perf_counter() - began,
                "error": error,
            }
            collector.writer.event(
                collector.coords(),
                values,
                reasons=["every_update"],
                status="failed" if error else "observed",
            )
            collector.window.add(
                collector.update,
                {"attempt": collector.attempt, "update": collector.update, "modules": modules},
            )
            collector.raw_batch = None
            if error is None and completed:
                collector.fixed(policy, optimizer)

    module.cycle, module.update_policy = cycle, update
    module._rosetta_dual_axis_active = True
    error = None
    primary = None
    try:
        yield collector
    except BaseException as exc:
        error = type(exc).__name__
        primary = exc
        raise
    finally:
        for name, expected, original in (
            ("cycle", cycle, original_cycle),
            ("update_policy", update, original_update),
        ):
            if getattr(module, name) is not expected:
                raise RuntimeError("Native diagnostic wrapper changed before teardown")
            setattr(module, name, original)
        module._rosetta_dual_axis_active = False
        try:
            collector.finish(error)
        except BaseException as cleanup:
            if primary is None:
                raise
            if hasattr(primary, "add_note"):
                primary.add_note(f"Diagnostic finalization failed: {type(cleanup).__name__}")


def load_tensor_tree(value, root):
    """Read previously sealed numeric probe banks, with no pickle deserialization."""
    import torch

    if isinstance(value, dict) and "tensor" in value and "chunks" in value:
        spec, chunks = value["tensor"], value["chunks"]
        allowed = {
            str(dtype): dtype
            for dtype in (
                torch.float32,
                torch.float64,
                torch.float16,
                torch.bfloat16,
                torch.int64,
                torch.int32,
                torch.int16,
                torch.int8,
                torch.uint8,
                torch.bool,
            )
        }
        if spec["dtype"] not in allowed:
            raise ValueError("Unsupported probe dtype")
        raw = bytearray()
        for chunk in chunks:
            if chunk["offset"] != len(raw) or chunk["bytes"] > 32 * 1024**2:
                raise ValueError("Invalid probe chunk layout")
            data = relative(root, chunk["path"]).read_bytes()
            if len(data) != chunk["bytes"] or hashlib.sha256(data).hexdigest() != chunk["sha256"]:
                raise ValueError("Probe chunk identity changed")
            raw.extend(data)
            if len(raw) > 256 * 1024**2:
                raise ValueError("Probe tensor exceeds memory budget")
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("Probe tensor hash mismatch")
        return torch.frombuffer(raw, dtype=allowed[spec["dtype"]]).clone().reshape(spec["shape"])
    if isinstance(value, dict):
        return {k: load_tensor_tree(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [load_tensor_tree(v, root) for v in value]
    return value


class PreparedProbes:
    """Uses live saved processors and explicit-noise native predict_action_chunk."""

    def __init__(self, plan, root, preprocessor, postprocessor, *, state_objects=()):
        self.plan, self.root = plan, Path(root)
        self.pre, self.post = preprocessor, postprocessor
        decoder_state = [
            (step, name)
            for step in getattr(postprocessor, "steps", [])
            for name in ("last_model_action", "last_unclipped_action")
            if hasattr(step, name)
        ]
        self.state_objects = list(state_objects) + decoder_state

    def __call__(self, policy, node):
        import torch

        panels = self.plan["panels"]
        noises = panels["noise_conditions"] if node["all_noises"] else [panels["regular_noise"]]
        device = next(policy.parameters()).device
        for sample in panels[node["panel"]]:
            item = self.plan["probe_bank"][sample["sample_id"]]
            raw = load_tensor_tree(item["raw"], self.root)
            batch = self.pre(copy.deepcopy(raw))
            if digest(tree_identity(batch)) != item["processed_identity"]:
                raise ValueError("Live processor differs from sealed probe ingress")
            for noise in noises:
                actual_noise = load_tensor_tree(item["noise"][noise["name"]], self.root).to(device)
                with torch.no_grad(), module_capture(policy, self.plan["modules"]) as modules:
                    normalized = policy.predict_action_chunk(batch, noise=actual_noise.clone())
                    decoded = self.post(normalized.clone())
                # Metric semantics are sealed in the bank, never inferred from shape.
                target = load_tensor_tree(item["target"], self.root).to(decoded.device)
                valid = load_tensor_tree(item["valid_mask"], self.root).to(decoded.device)
                if (
                    decoded.shape != target.shape
                    or valid.shape != decoded.shape[:-1]
                    or valid.dtype != torch.bool
                ):
                    raise ValueError("Probe target/mask contract mismatch")
                if not bool(torch.isfinite(decoded).all()) or not bool(
                    torch.isfinite(target).all()
                ):
                    raise FloatingPointError("Nonfinite probe predictions or targets")
                metrics = []
                for group, indices in self.plan["action_groups"].items():
                    for window in ("first_action", "full_chunk"):
                        e = (decoded - target).abs()[..., indices]
                        mask = valid
                        if window == "first_action":
                            e, mask = e[:, :1], mask[:, :1]
                        if not bool(mask.any()):
                            raise ValueError("Empty probe scoring mask")
                        name = f"{group}/{window}/mae"
                        metrics.append(
                            {
                                "name": name,
                                "group": group,
                                "window": window,
                                "category": "behavior",
                                "value": float(e[mask].mean()),
                                "aa_floor": item.get("aa_floor", {})
                                .get(noise["name"], {})
                                .get(name),
                            }
                        )
                        threshold = self.plan.get("absolute_thresholds", {}).get(name)
                        if threshold is not None:
                            metrics[-1].update(
                                absolute_limit=threshold["limit"],
                                threshold_identity=threshold["identity"],
                            )
                identity = {**item["identity"], "noise": digest(item["noise"][noise["name"]])}
                yield (
                    sample,
                    noise,
                    {
                        "raw": raw,
                        "processed": batch,
                        "noise": actual_noise,
                        "normalized": normalized,
                        "decoded": decoded,
                        "target": target,
                        "mask": valid,
                        "modules": modules,
                        "metrics": metrics,
                    },
                    identity,
                )


class DualAxisDiagnosticsFeature:
    """Install last so teardown restores this outer observation before other features."""

    name = "dual_axis_diagnostics"

    def __init__(self, parameters):
        if set(parameters) != {"path", "sha256"}:
            raise ValueError("Dual-axis feature needs one sealed plan path and sha256")
        self.parameters = parameters
        self.scope = None

    def install(self, context):
        from .features import _diagnostics_destination, _lerobot_train_module

        root = Path(__file__).resolve().parents[4]
        path = relative(root, self.parameters["path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.parameters["sha256"]:
            raise ValueError("Diagnostic plan identity drift")
        plan = json.loads(raw)
        if plan.get("status") != "registered" or plan.get("source_run") != context.run_name:
            raise ValueError("Diagnostic draft or run mismatch")
        validate_registration(plan)
        if (
            plan.get("action_contract_sha256")
            != hashlib.sha256(context.contract_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Live Action Contract differs from diagnostic registration")
        admission = plan.get("admission", {})
        result = relative(root, admission["path"]).read_bytes()
        if hashlib.sha256(result).hexdigest() != admission["sha256"]:
            raise ValueError("Admission evidence changed")
        accepted = json.loads(result)
        if (
            accepted.get("status") != "passed"
            or accepted.get("evidence_kind") != "registered_model"
        ):
            raise ValueError("Real-model transparency and resource admission required")
        if accepted.get("implementation_files") != context.plan.get("implementation_files"):
            raise ValueError("Admission does not bind this implementation")
        if accepted.get("runtime_sha256") != plan["runtime_sha256"]:
            raise ValueError("Admission runtime identity differs")
        module = _lerobot_train_module()
        original = module.make_pre_post_processors
        owner = self

        def processors(*args, **kwargs):
            pair = original(*args, **kwargs)
            if owner.scope is not None:
                raise RuntimeError("Duplicate processor construction")
            bank_root = relative(root, plan["probe_bank_root"])
            probes = PreparedProbes(plan, bank_root, *pair)
            output = _diagnostics_destination(context, f"dual-axis-{context.run_name}")
            collector = Collector(output, plan, probes=probes, guard=lambda: runtime_guard(plan))
            owner.scope = observe_dual_axis(module, collector)
            owner.scope.__enter__()
            return pair

        self.module, self.original, self.wrapper = module, original, processors
        module.make_pre_post_processors = processors

    def restore(self, context):
        if self.module.make_pre_post_processors is not self.wrapper:
            raise RuntimeError("Processor hook changed")
        self.module.make_pre_post_processors = self.original
        if self.scope is None:
            raise RuntimeError("Diagnostic processor binding was never exercised")
        self.scope.__exit__(None, None, None)
