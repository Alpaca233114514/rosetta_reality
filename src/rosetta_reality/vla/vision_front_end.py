"""Explicit visual front-end trainable scope (vfunfreeze candidate axis).

The pinned SmolVLA adaptation flags cannot express "make the visual front-end
trainable while keeping the language model frozen": ``train_expert_only=true``
freezes the whole VLM (vision tower, connector and text model together), and
switching it off opens the text stack to training.  This module owns the
explicit trainable-parameter scope instead: exactly ``vision_model`` and
``connector`` under the VLM become trainable; every other parameter keeps the
frozen-baseline ``requires_grad`` value it already had.  The upstream
``set_requires_grad`` calls in the pinned constructors are never patched —
the scope is applied once, after ``make_policy`` returns, inside the trainer
process through the ``vision_front_end_unfreeze`` feature.  Validation,
export and deployment never install it, and the saved policy config keeps the
frozen-baseline flags so an independently reloaded artifact freezes the
visual front-end again at inference time.

The scope application validates the complete VLM before changing any flag:
each declared module must contain parameters, scope parameters cannot be
shared outside their module, and every other VLM parameter must be frozen.
Rejected layouts leave the incoming model untouched.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Module paths relative to ``policy.model.vlm_with_expert``.
FRONT_END_SCOPE: tuple[str, ...] = (
    "vlm.model.vision_model",
    "vlm.model.connector",
)

SCOPE_PROFILE = "visual_front_end_unfreeze"
CHECKPOINT_PROFILE = "per_module_nonreentrant_activation_checkpointing"


def _front_end_modules(vlm_with_expert: Any) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for dotted_path in FRONT_END_SCOPE:
        try:
            module = _resolve_module(vlm_with_expert, dotted_path)
        except AttributeError as error:
            raise ValueError(
                f"The pinned model no longer exposes the declared scope module: {dotted_path}"
            ) from error
        modules[dotted_path] = module
    return modules


def install_front_end_checkpointing(policy_model: Any) -> dict[str, Any]:
    """Wrap each front-end module forward in non-reentrant checkpointing.

    A trainable SigLIP tower at 512x512 stores roughly 0.6--0.8 GiB of
    per-sample backward activations across its layers, which does not fit a
    24 GiB device at the registered batches.  Per-module ``torch.utils.
    checkpoint`` with ``use_reentrant=False`` recomputes each module forward
    during backward instead of storing its internals: the resulting updates
    are mathematically identical (the non-reentrant implementation also
    stashes and restores RNG state, keeping any dropout reproducible), at the
    cost of one extra forward per module.  The wrapper is a pure passthrough
    whenever gradients are disabled, so validation, export and deployment
    never enter the checkpoint path.
    """

    import torch
    import torch.utils.checkpoint as checkpoint

    vlm_with_expert = getattr(policy_model, "vlm_with_expert", None)
    if vlm_with_expert is None:
        raise ValueError("The policy model has no vlm_with_expert submodule.")
    modules = _front_end_modules(vlm_with_expert)
    # Validate all modules before wrapping any of them. A conflict in the
    # connector must not leave the vision tower partially installed.
    for dotted_path, module in modules.items():
        if not callable(getattr(module, "forward", None)) or not callable(
            getattr(module, "parameters", None)
        ):
            raise ValueError(f"The declared scope path is not a module: {dotted_path}")
        if getattr(module.forward, "_rosetta_front_end_checkpoint", False):
            raise RuntimeError(f"Front-end checkpointing is already installed: {dotted_path}")
    wrapped: list[str] = []
    for dotted_path, module in modules.items():
        original = module.forward

        def checkpointed_forward(*args, _original=original, _module=module, **kwargs):
            if torch.is_grad_enabled() and any(
                parameter.requires_grad for parameter in _module.parameters()
            ):
                return checkpoint.checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)

        checkpointed_forward._rosetta_front_end_checkpoint = True  # type: ignore[attr-defined]
        checkpointed_forward._rosetta_original_forward = original  # type: ignore[attr-defined]
        module.forward = checkpointed_forward  # type: ignore[method-assign, assignment]
        wrapped.append(dotted_path)
    return {
        "profile": CHECKPOINT_PROFILE,
        "wrapped_modules": wrapped,
        "use_reentrant": False,
        "passthrough_when_grad_disabled": True,
    }


def restore_front_end_checkpointing(policy_model: Any) -> None:
    """Remove the checkpoint wrappers installed above (tests and diagnostics)."""

    vlm_with_expert = getattr(policy_model, "vlm_with_expert", None)
    if vlm_with_expert is None:
        raise ValueError("The policy model has no vlm_with_expert submodule.")
    for module in _front_end_modules(vlm_with_expert).values():
        current = module.forward
        if getattr(current, "_rosetta_front_end_checkpoint", False):
            module.forward = getattr(  # type: ignore[method-assign, assignment]
                current, "_rosetta_original_forward"
            )


def _resolve_module(root: Any, dotted_path: str) -> Any:
    attribute = root
    for part in dotted_path.split("."):
        attribute = getattr(attribute, part)
    return attribute


def apply_front_end_scope(policy_model: Any) -> dict[str, Any]:
    """Apply the declared front-end scope to one built ``VLAFlowMatching``.

    ``policy_model`` is the ``VLAFlowMatching`` module (``policy.model``), so
    the reported parameter names are relative to it
    (``vlm_with_expert.vlm.model.vision_model.*``).  Returns the scope record
    used by the feature's create-only diagnostics report.  Raises
    ``ValueError`` on any structural surprise instead of training with a
    partially wrong trainable set.
    """

    import torch

    if not FRONT_END_SCOPE:
        raise ValueError("The visual front-end scope is empty.")

    vlm_with_expert = getattr(policy_model, "vlm_with_expert", None)
    if vlm_with_expert is None:
        raise ValueError("The policy model has no vlm_with_expert submodule.")
    modules = _front_end_modules(vlm_with_expert)
    for dotted_path, module in modules.items():
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"The declared scope path is not a module: {dotted_path}")

    scope_parameter_ids: set[int] = set()
    scope_names: list[str] = []
    for dotted_path, module in modules.items():
        parameters = list(module.named_parameters())
        if not parameters:
            raise ValueError(f"The declared scope module has no parameters: {dotted_path}")
        for name, parameter in parameters:
            if not isinstance(parameter, torch.nn.Parameter):
                raise ValueError(f"Scope entry is not a parameter: {dotted_path}.{name}")
            if id(parameter) in scope_parameter_ids:
                raise ValueError("A parameter is shared between declared scope modules.")
            scope_parameter_ids.add(id(parameter))
            scope_names.append(f"vlm_with_expert.{dotted_path}.{name}")

    scope_name_set = set(scope_names)
    # Enumerate aliases before mutating requires_grad. The default deduplicated
    # iterator hides a scope weight tied to a language embedding or projection.
    for name, parameter in policy_model.named_parameters(remove_duplicate=False):
        if name not in scope_name_set and id(parameter) in scope_parameter_ids:
            raise ValueError(f"Scope parameter is aliased outside the scope: {name}")

    parameters_by_name = dict(policy_model.named_parameters())
    if not scope_name_set <= parameters_by_name.keys():
        raise ValueError("Declared scope parameters are missing from the policy model.")
    # Check the entire VLM complement. Matching only text_model misses the
    # language_model namespace, token embeddings and vlm.lm_head; all([])
    # can otherwise incorrectly report a frozen language stack.
    frozen_vlm = {
        name: parameter
        for name, parameter in parameters_by_name.items()
        if name.startswith("vlm_with_expert.vlm.") and name not in scope_name_set
    }
    if not frozen_vlm:
        raise ValueError("No non-visual VLM parameters were found to verify as frozen.")
    for name, parameter in frozen_vlm.items():
        if parameter.requires_grad:
            raise ValueError(f"Non-visual VLM parameter must stay frozen: {name}")

    before = {
        name: bool(parameter.requires_grad) for name, parameter in policy_model.named_parameters()
    }
    newly_trainable = 0
    for name, parameter in policy_model.named_parameters():
        in_scope = id(parameter) in scope_parameter_ids
        if in_scope and not parameter.requires_grad:
            parameter.requires_grad = True
            newly_trainable += 1
        if in_scope and not parameter.requires_grad:
            raise ValueError(f"Scope parameter stayed frozen: {name}")

    after = {
        name: bool(parameter.requires_grad) for name, parameter in policy_model.named_parameters()
    }
    changed = sorted(name for name in before if before[name] != after[name])
    # Every parameter whose flag changed must be a declared scope parameter,
    # and every declared scope parameter must now be trainable.
    if set(changed) - scope_name_set:
        raise ValueError("The scope application changed parameters outside the declared front end.")
    for name in scope_names:
        if not after.get(name, False):
            raise ValueError(f"Declared scope parameter is not trainable: {name}")
    trainable_after = [name for name, flag in after.items() if flag]
    return {
        "profile": SCOPE_PROFILE,
        "scope_modules": list(FRONT_END_SCOPE),
        "scope_parameter_count": len(scope_names),
        "scope_parameter_names_sha256": hashlib.sha256(
            "\n".join(sorted(scope_names)).encode("utf-8")
        ).hexdigest(),
        "newly_trainable_parameter_count": newly_trainable,
        "trainable_parameter_count_after": len(trainable_after),
        "changed_parameter_count": len(changed),
        "language_model_stays_frozen": all(not after[name] for name in frozen_vlm),
        "frozen_non_visual_vlm_parameter_count": len(frozen_vlm),
        "frozen_non_visual_vlm_parameter_names_sha256": hashlib.sha256(
            "\n".join(sorted(frozen_vlm)).encode("utf-8")
        ).hexdigest(),
        "upstream_scope_flags_untouched": True,
    }


def scope_report_text(scope: dict[str, Any], **extra: Any) -> str:
    payload = {
        "schema_version": 1,
        "stage": "smolvla_vision_front_end_scope",
        "scope": scope,
        **extra,
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
