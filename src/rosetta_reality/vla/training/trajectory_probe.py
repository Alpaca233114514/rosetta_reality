"""File-only trajectory probe planning and saved-array diagnostics.

This module does not launch training, load a policy or access a dataset.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from rosetta_reality.vla.training.trajectory_scoring import (
    score_saved_predictions as score_saved_predictions,
)

TRAIN_EPISODES = (2, 49, 4, 23)
NOISE_CONDITIONS = ("zero", "seed_20260905", "seed_20260906", "seed_20260907")
PINNED_NATIVE_LOSS_SHA256 = "37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d"


def canonical_bytes(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def schedule(episodes, *, rounds=4, seed=20260809):
    """Shuffle a full 500-frame pass separately for each round."""
    episodes = tuple(episodes)
    if (
        not episodes
        or len(set(episodes)) != len(episodes)
        or not set(episodes) <= set(TRAIN_EPISODES)
    ):
        raise ValueError("Only distinct registered training episodes are allowed")
    if rounds != 4:
        raise ValueError("The registered probe has four rounds")
    result = []
    for epoch in range(rounds):
        samples = [[ep, frame] for ep in episodes for frame in range(500)]
        random.Random(seed + epoch).shuffle(samples)
        result.extend(samples)
    identities = Counter(map(tuple, result))
    if set(identities.values()) != {4} or len(result) != len(episodes) * 500 * 4:
        raise ValueError("Incomplete four-round frame coverage")
    return result


def effective_runtime_contract(policy, optimizer, features, scheduler, loss_mask, expected, ledger):
    """Inspect live objects and fail on a declaration/runtime mismatch.

    ``features`` maps installed feature names to installation evidence, and
    ``loss_mask`` carries measured valid elements and reduction denominator.
    An absent observation is a failure; this function never supplies one.
    """
    names = {id(parameter): name for name, parameter in policy.named_parameters()}
    trainable = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    groups = []
    assigned = []
    for group in optimizer.param_groups:
        members = [names.get(id(parameter)) for parameter in group["params"]]
        if None in members or not members:
            raise ValueError("Optimizer parameter group contains unknown or no parameters")
        assigned.extend(members)
        groups.append(
            {
                "parameters": members,
                "lr": float(group["lr"]),
                "betas": list(group["betas"]),
                "eps": float(group["eps"]),
                "weight_decay": float(group["weight_decay"]),
            }
        )
    if len(assigned) != len(set(assigned)) or set(assigned) != trainable:
        raise ValueError("Optimizer trainable parameter coverage differs")
    if set(trainable) != set(expected["trainable_parameters"]):
        raise ValueError("Trainable policy parameters drifted")
    if groups != expected["optimizer_groups"]:
        raise ValueError("Optimizer group or hyperparameter drifted")
    if not isinstance(features, dict) or set(features) != set(expected["features"]):
        raise ValueError("Installed feature set drifted")
    if any(
        not isinstance(value, dict)
        or value.get("installed") is not True
        or not value.get("evidence")
        for value in features.values()
    ):
        raise ValueError("Feature installation evidence is missing")
    actual_lr = [float(value) for value in scheduler.get_last_lr()]
    if actual_lr != [group["lr"] for group in groups] or actual_lr != expected["actual_lr"]:
        raise ValueError("Actual scheduler/optimizer LR drifted")
    if not loss_mask or loss_mask != expected["loss_mask"]:
        raise ValueError("Loss mask or reduction denominator drifted")
    observed = ledger.snapshot() if hasattr(ledger, "snapshot") else ledger
    if not isinstance(observed, dict) or not observed.get("status"):
        raise ValueError("Sample ledger evidence is missing")
    return {
        "status": "observed_contract_match",
        "trainable_parameters": sorted(trainable),
        "optimizer_groups": groups,
        "installed_features": features,
        "actual_lr": actual_lr,
        "loss_mask": loss_mask,
        "ledger": observed,
        "expected_save_grid": expected["save_grid"],
    }


def native_loss_descriptor(batch, *, source_path, source_sha256):
    """Describe default unweighted global-valid reduction from the actual batch."""
    if source_sha256 != PINNED_NATIVE_LOSS_SHA256:
        raise ValueError("Native loss reduction source is not the pinned implementation")
    if hashlib.sha256(Path(source_path).read_bytes()).hexdigest() != source_sha256:
        raise ValueError("Pinned upstream loss source changed")
    action = batch.get("action")
    pad = batch.get("action_is_pad")
    if action is None or pad is None or len(action.shape) != 3:
        raise ValueError("Actual action and padding tensors are required")
    flags = pad.detach().cpu().tolist() if hasattr(pad, "detach") else pad
    batch_size, horizon, dimensions = action.shape
    if len(flags) != batch_size or any(len(row) != horizon for row in flags):
        raise ValueError("Actual padding shape differs from action")
    valid = []
    for row in flags:
        if any(type(flag) is not bool for flag in row) or row != sorted(row):
            raise ValueError("Invalid actual tail padding")
        count = sum(not flag for flag in row)
        if count == 0:
            raise ValueError("Empty action target")
        valid.append(count)
    return {
        "reduction": "global_valid_action_entries",
        "upstream_source_sha256": source_sha256,
        "valid_steps_per_sample": valid,
        "denominator": sum(valid) * dimensions,
        "action_dimension": dimensions,
    }


def read_installed_features(stack, module):
    """Read the live feature stack and trainer markers, not plan declarations."""
    installed = getattr(stack, "installed", None)
    if not isinstance(installed, list) or not installed:
        raise ValueError("Active native FeatureStack is required")
    result = {}
    for name in installed:
        marker = f"_rosetta_v2_feature_{name}_installed"
        if hasattr(module, marker):
            if getattr(module, marker) is not True:
                raise ValueError("Installed feature marker disagrees with stack")
            evidence = "native_trainer_marker_and_stack"
        else:
            evidence = "native_feature_stack"
        result[name] = {"installed": True, "evidence": evidence}
    return result


def _base_optimizer(optimizer):
    seen = set()
    while optimizer is not None and id(optimizer) not in seen:
        seen.add(id(optimizer))
        inner = getattr(optimizer, "optimizer", None)
        if inner is None:
            return optimizer
        optimizer = inner
    raise ValueError("Optimizer wrapper cycle or missing optimizer")


@contextmanager
def observe_effective_runtime(module, *, expected, ledger, output, stack=None, feature_reader=None):
    """Install create-only first-batch checks around native trainer entrypoints.

    Place this outside the native launch and inside its feature installation
    lifetime. It does not launch the trainer or certify a dry-run as real.
    """
    maker = getattr(module, "make_optimizer_and_scheduler", None)
    updater = getattr(module, "update_policy", None)
    if not callable(maker) or not callable(updater):
        raise ValueError("Native optimizer and update entrypoints are required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    captured = {}

    @wraps(maker)
    def make(*args, **kwargs):
        config = kwargs.get("cfg", args[0] if args else None)
        if config is None:
            raise ValueError("Constructed native training config is required")
        steps, frequency = getattr(config, "steps", None), getattr(config, "save_freq", None)
        if type(steps) is not int or type(frequency) is not int or min(steps, frequency) < 1:
            raise ValueError("Actual native checkpoint grid is unavailable")
        grid = sorted(set(range(frequency, steps + 1, frequency)) | {steps})
        if grid != expected["save_grid"]:
            raise ValueError("Actual native checkpoint grid drifted")
        captured["save_grid"] = grid
        value = maker(*args, **kwargs)
        if not isinstance(value, tuple) or len(value) < 2:
            raise ValueError("Native optimizer/scheduler result differs")
        captured["optimizer"], captured["scheduler"] = value[:2]
        return value

    @wraps(updater)
    def update(*args, **kwargs):
        if kwargs.get("sample_weighter", args[8] if len(args) > 8 else None) is not None:
            raise ValueError("Weighted loss is outside the canonical probe contract")
        if "checked" not in captured:
            policy = kwargs.get("policy", args[1] if len(args) > 1 else None)
            batch = kwargs.get("batch", args[2] if len(args) > 2 else None)
            optimizer = kwargs.get("optimizer", args[3] if len(args) > 3 else None)
            if (
                policy is None
                or batch is None
                or _base_optimizer(optimizer) is not _base_optimizer(captured.get("optimizer"))
            ):
                raise ValueError("First native policy batch or optimizer identity missing")
            loss_mask = native_loss_descriptor(
                batch,
                source_path=expected["loss_source_path"],
                source_sha256=expected["loss_source_sha256"],
            )
            report = effective_runtime_contract(
                policy,
                optimizer,
                feature_reader(policy, module)
                if feature_reader is not None
                else read_installed_features(stack, module),
                captured["scheduler"],
                loss_mask,
                expected,
                ledger,
            )
            report["actual_save_grid"] = captured["save_grid"]
            report["feature_evidence_scope"] = (
                "synthetic_callback" if feature_reader is not None else "live_feature_stack"
            )
            with output.open("xb") as stream:
                stream.write(canonical_bytes(report))
            captured["checked"] = True
        return updater(*args, **kwargs)

    module.make_optimizer_and_scheduler = make
    module.update_policy = update
    try:
        yield
        if "checked" not in captured:
            raise RuntimeError("No native batch reached runtime-contract verification")
    finally:
        changed = (
            module.make_optimizer_and_scheduler is not make or module.update_policy is not update
        )
        if module.make_optimizer_and_scheduler is make:
            module.make_optimizer_and_scheduler = maker
        if module.update_policy is update:
            module.update_policy = updater
        if changed:
            import sys

            if sys.exc_info()[0] is None:
                raise RuntimeError("Native runtime hooks changed before restoration")
