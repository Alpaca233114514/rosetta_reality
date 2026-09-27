"""Seal an explicitly supplied train/dev probe bank; no dataset discovery/download."""

import copy

from .dual_axis import EvidenceWriter, digest
from .dual_axis_torch import tensor_tree, tree_identity


def seal_bank(output, panels, load_sample, preprocess, *, noise_shape, identity, maximum_bytes):
    """Caller supplies registered canonical raw batches, projected targets and valid masks.

    Never called by plan preparation. Real cache access requires its own authorized stage.
    Processor/noise/Action Contract identities are measured separately, never guessed.
    """
    import torch

    if any(not identity.get(k) for k in ("processor", "action_contract", "runtime")):
        raise ValueError("Bank sealing requires processor/contract/runtime identities")
    rows = panels["extended"]
    # Exclude hidden and malformed IDs before the first reader invocation.
    if any(r["episode"] in panels["hidden"] or r["split"] not in ("train", "dev") for r in rows):
        raise ValueError("Forbidden probe split")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate probe identity")
    writer = EvidenceWriter(
        output, {"source_run": "sealed-probe-bank", **identity}, maximum_bytes=maximum_bytes
    )
    bank = {}
    try:
        noises = {}
        for condition in panels["noise_conditions"]:
            seed = condition["seed"]
            noise = (
                torch.zeros(noise_shape)
                if seed is None
                else torch.randn(
                    noise_shape, generator=torch.Generator(device="cpu").manual_seed(seed)
                )
            )
            noises[condition["name"]] = tensor_tree(noise, writer)
        for row in rows:
            raw, target, valid = load_sample(row)
            if (
                int(raw["episode_index"].reshape(-1)[0]) != row["episode"]
                or int(raw["frame_index"].reshape(-1)[0]) != row["frame"]
            ):
                raise ValueError("Actual probe sample identity mismatch")
            raw_tree = tensor_tree(raw, writer)
            processed = preprocess(copy.deepcopy(raw))
            target_tree, mask_tree = tensor_tree(target, writer), tensor_tree(valid, writer)
            input_identity = digest(
                {
                    "raw": raw_tree,
                    "target": target_tree,
                    "mask": mask_tree,
                    "processed": tree_identity(processed),
                }
            )
            bank[row["sample_id"]] = {
                "raw": raw_tree,
                "target": target_tree,
                "valid_mask": mask_tree,
                "noise": noises,
                "processed_identity": digest(tree_identity(processed)),
                "identity": {**identity, "input": input_identity},
                "aa_floor": {},
                "AA_status": "not_measured",
            }
        writer.write("bank.json", bank)
        writer.seal()
        return bank
    except BaseException as exc:
        if not writer.closed:
            writer.seal("incomplete", error=type(exc).__name__)
        raise
