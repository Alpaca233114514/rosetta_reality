"""Run upstream LeRobot training with Rosetta's local-only Trackio logger."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


def _install_masked_camera_encoder_skip() -> None:
    """Keep masked camera token geometry while skipping redundant vision calls.

    The pinned SmolVLA path appends fully masked placeholder cameras after the
    real cameras. Their embeddings cannot contribute as attention keys, but the
    upstream implementation still invokes the frozen vision encoder. This
    opt-in replacement preserves the exact prefix shapes and masks and creates
    same-shape zeros only for those registered trailing placeholders.
    """

    import torch
    from lerobot.policies.smolvla import modeling_smolvla

    flow_class = modeling_smolvla.VLAFlowMatching
    current = flow_class.embed_prefix
    if getattr(current, "_rosetta_masked_camera_skip", False):
        return

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
                    raise RuntimeError("A real camera embedding must precede placeholders.")
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


def _convert_statistics(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    import torch

    converted: dict[str, dict[str, Any]] = {}
    for feature, raw_statistics in value.items():
        if not isinstance(raw_statistics, dict):
            raise ValueError("Train-only feature statistics must be mappings.")
        converted[feature] = {}
        for statistic, raw_value in raw_statistics.items():
            if not isinstance(raw_value, list) or not raw_value:
                raise ValueError("Train-only statistics must be non-empty lists.")
            dtype = torch.int64 if statistic == "count" else torch.float64
            converted[feature][statistic] = torch.tensor(raw_value, dtype=dtype)
    return converted


def _install_train_only_statistics(lerobot_train: Any) -> None:
    report_path = os.environ.get("ROSETTA_VLA_TRAIN_STATS_REPORT")
    if report_path is None:
        return
    path = Path(report_path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("ROSETTA_VLA_TRAIN_STATS_REPORT must identify an existing absolute path.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("status") != "complete"
        or report.get("stage") != "smolvla_train_only_normalization"
        or report.get("source_split") != "train"
        or report.get("validation_episodes_loaded") is not False
        or report.get("hidden_test_loaded") is not False
        or not isinstance(report.get("effective_stats"), dict)
    ):
        raise ValueError("SmolVLA train-only normalization report is invalid.")
    allowed_episodes = {int(value) for value in report["train_episodes"]}
    _convert_statistics(report["effective_stats"])
    original = lerobot_train.make_train_eval_datasets

    def make_train_eval_datasets(cfg: Any) -> tuple[Any, Any]:
        requested = {int(value) for value in (cfg.dataset.episodes or [])}
        if not requested or not requested.issubset(allowed_episodes):
            raise ValueError("Training episodes are outside the train-only normalization scope.")
        dataset, eval_dataset = original(cfg)
        dataset.meta.stats.update(_convert_statistics(report["effective_stats"]))
        if eval_dataset is not None:
            eval_dataset.meta.stats.update(_convert_statistics(report["effective_stats"]))
        return dataset, eval_dataset

    lerobot_train.make_train_eval_datasets = make_train_eval_datasets


def main() -> None:
    import lerobot.scripts.lerobot_train as lerobot_train

    from rosetta_reality.tracking.trackio_lerobot import TrackioLogger, finish_trackio

    if os.environ.get("ROSETTA_VLA_SKIP_FULLY_MASKED_CAMERA_ENCODING") == "1":
        _install_masked_camera_encoder_skip()
    _install_train_only_statistics(lerobot_train)
    lerobot_train.WandBLogger = TrackioLogger
    try:
        lerobot_train.main()
    finally:
        finish_trackio()


if __name__ == "__main__":
    main()
