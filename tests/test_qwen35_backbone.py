"""Prompt construction and pooling contracts for the lazy Qwen3.5 adapter."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from rosetta_reality.models.backbones.qwen35 import Qwen35Backbone


class _BaseProcessor:
    chat_template = None
    image_token = "<image>"
    vision_start_token = "<vision>"
    vision_end_token = "</vision>"

    def __init__(self) -> None:
        self.call: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.call = kwargs
        return kwargs


def _loaded_backbone(processor: Any, *, prompt_mode: str) -> Qwen35Backbone:
    backbone = Qwen35Backbone(
        "local-model",
        hidden_size=8,
        prompt_template="Act: {instruction}",
        prompt_mode=prompt_mode,
    )
    backbone._model = nn.Identity()
    backbone._processor = processor
    return backbone


def test_base_prompt_uses_explicit_multimodal_placeholders() -> None:
    processor = _BaseProcessor()
    backbone = _loaded_backbone(processor, prompt_mode="base_multimodal")

    encoded = backbone.prepare_inputs(
        {
            "top": torch.zeros(3, 8, 8),
            "wrist": torch.ones(3, 8, 8),
        },
        "insert peg",
    )

    assert encoded["text"] == (
        "<vision><image></vision>\n"
        "<vision><image></vision>\n"
        "Act: insert peg"
    )
    assert encoded["return_tensors"] == "pt"
    assert len(encoded["images"]) == 2
    assert processor.call is encoded


def test_chat_prompt_mode_rejects_base_processor_without_template() -> None:
    backbone = _loaded_backbone(_BaseProcessor(), prompt_mode="chat_template")

    with pytest.raises(RuntimeError, match="does not provide one"):
        backbone.prepare_inputs({"top": torch.zeros(3, 8, 8)}, "insert peg")


def test_unknown_prompt_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="prompt_mode"):
        Qwen35Backbone("local-model", prompt_mode="not-a-mode")


def _pooling_inputs() -> tuple[torch.Tensor, dict[str, torch.Tensor], nn.Module]:
    hidden = torch.tensor(
        [[[0.0], [1.0], [2.0], [3.0], [4.0], [9.0]]],
    )
    encoded = {
        "input_ids": torch.tensor([[1, 7, 7, 7, 7, 2]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    model = nn.Identity()
    model.config = SimpleNamespace(
        image_token_id=7,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    return hidden, encoded, model


def test_image_token_mean_excludes_text_tokens() -> None:
    backbone = Qwen35Backbone("local-model", hidden_size=1, pooling="image_token_mean")
    hidden, encoded, model = _pooling_inputs()

    pooled = backbone._pool_final_hidden(hidden, encoded, model)

    torch.testing.assert_close(pooled, torch.tensor([[2.5]]))
    assert backbone.hidden_size == 1


def test_image_spatial_pool_preserves_quadrant_order_and_declares_width() -> None:
    backbone = Qwen35Backbone("local-model", hidden_size=1, pooling="image_spatial_2x2")
    hidden, encoded, model = _pooling_inputs()

    pooled = backbone._pool_final_hidden(hidden, encoded, model)

    torch.testing.assert_close(pooled, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert backbone.hidden_size == 4


def test_combined_pooling_concatenates_global_then_spatial_features() -> None:
    backbone = Qwen35Backbone(
        "local-model",
        hidden_size=1,
        pooling="attention_masked_mean_plus_image_spatial_2x2",
    )
    hidden, encoded, model = _pooling_inputs()

    pooled = backbone._pool_final_hidden(hidden, encoded, model)

    torch.testing.assert_close(pooled, torch.tensor([[19 / 6, 1.0, 2.0, 3.0, 4.0]]))
    assert backbone.hidden_size == 5


def test_unknown_pooling_is_rejected() -> None:
    with pytest.raises(ValueError, match="pooling"):
        Qwen35Backbone("local-model", pooling="not-a-pooling")
