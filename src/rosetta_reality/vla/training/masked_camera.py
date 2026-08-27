"""Masked-camera encoder skip for the pinned SmolVLA vision prefix.

Migrated verbatim in behavior from the frozen historical trainer
``scripts/train_smolvla_trackio.py`` so the version-2 harness owns it as a
first-class, restorable extension.  The pinned SmolVLA path appends fully
masked placeholder cameras after the real cameras.  Their embeddings cannot
contribute as attention keys, but the upstream implementation still invokes
the frozen vision encoder.  This replacement preserves the exact prefix
shapes and masks and creates same-shape zeros only for those registered
trailing placeholders.
"""

from __future__ import annotations

import math
from typing import Any

_INSTALL_MARKER = "_rosetta_v2_masked_camera_skip_installed"


def install_masked_camera_encoder_skip(modeling_module: Any) -> None:
    """Keep masked camera token geometry while skipping redundant vision calls."""

    import torch

    flow_class = modeling_module.VLAFlowMatching
    current = flow_class.embed_prefix
    if getattr(current, "_rosetta_masked_camera_skip", False):
        raise RuntimeError("The masked-camera skip is already installed.")

    def embed_prefix(
        self: Any,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.add_image_special_tokens or int(self.prefix_length) != 0:
            raise RuntimeError(
                "Masked-camera skipping requires the pinned no-special-token prefix."
            )
        empty_cameras = int(self.config.empty_cameras)
        real_cameras = len(images) - empty_cameras
        if (
            empty_cameras <= 0
            or real_cameras <= 0
            or len(images) != len(img_masks)
        ):
            raise RuntimeError("Masked-camera skipping received an invalid camera layout.")
        if (
            not getattr(self, "_rosetta_mask_contract_validated", False)
            and not torch.compiler.is_compiling()
        ):
            present = [bool(mask.any().item()) for mask in img_masks]
            if present != [True] * real_cameras + [False] * empty_cameras:
                raise RuntimeError(
                    "Only trailing, fully masked placeholder cameras may skip encoding."
                )
            self._rosetta_mask_contract_validated = True

        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []
        attention_pattern: list[int] = []
        image_template: torch.Tensor | None = None
        for image_index, (image, image_mask) in enumerate(
            zip(images, img_masks, strict=True)
        ):
            if image_index < real_cameras:
                image_embedding = self.vlm_with_expert.embed_image(image)
                image_template = image_embedding
            else:
                if image_template is None:
                    raise RuntimeError(
                        "A real camera embedding must precede placeholders."
                    )
                image_embedding = torch.zeros_like(image_template)
            image_dimension = image_embedding.shape[-1]
            image_embedding = image_embedding * torch.tensor(
                image_dimension**0.5,
                dtype=image_embedding.dtype,
                device=image_embedding.device,
            )
            batch_size, image_tokens = image_embedding.shape[:2]
            expanded_mask = image_mask[:, None].expand(batch_size, image_tokens)
            embs.append(image_embedding)
            pad_masks.append(expanded_mask)
            attention_pattern += [0] * image_tokens

        language_embedding = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        language_embedding = language_embedding * math.sqrt(language_embedding.shape[-1])
        embs.append(language_embedding)
        pad_masks.append(lang_masks)
        attention_pattern += [0] * language_embedding.shape[1]

        state_embedding = self.state_proj(state)
        if state_embedding.ndim == 2:
            state_embedding = state_embedding[:, None, :]
        embs.append(state_embedding)
        batch_size = state_embedding.shape[0]
        state_tokens = state_embedding.shape[1]
        pad_masks.append(
            torch.ones(
                batch_size,
                state_tokens,
                dtype=torch.bool,
                device=state_embedding.device,
            )
        )
        attention_pattern += [1] * state_tokens
        embeddings = torch.cat(embs, dim=1)
        padding = torch.cat(pad_masks, dim=1)
        attention = torch.tensor(
            attention_pattern, dtype=torch.bool, device=padding.device
        )[None, :].expand(batch_size, -1)
        return embeddings, padding, attention

    embed_prefix._rosetta_masked_camera_skip = True  # type: ignore[attr-defined]
    embed_prefix._rosetta_original = current  # type: ignore[attr-defined]
    flow_class.embed_prefix = embed_prefix
    setattr(modeling_module, _INSTALL_MARKER, True)


def restore_masked_camera_encoder_skip(modeling_module: Any) -> None:
    """Remove the installed skip. Intended for tests and diagnostics."""

    if getattr(modeling_module, _INSTALL_MARKER, False) is not True:
        raise RuntimeError("No masked-camera skip is installed.")
    flow_class = modeling_module.VLAFlowMatching
    current = flow_class.embed_prefix
    if getattr(current, "_rosetta_masked_camera_skip", False) is not True:
        raise RuntimeError("The masked-camera skip marker disagrees with the active surface.")
    flow_class.embed_prefix = getattr(current, "_rosetta_original")
    setattr(modeling_module, _INSTALL_MARKER, False)
