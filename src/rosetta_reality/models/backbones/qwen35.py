"""Lazy, local-first adapter skeleton for Qwen3.5 backbones."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

from rosetta_reality.models.backbones.base import BackboneBatch, VLABackbone


class Qwen35Backbone(VLABackbone):
    """Load and pool a Qwen3.5 model only when it is first used.

    The adapter defaults to ``local_files_only=True`` so a call cannot silently
    download weights. Device and dtype are caller-controlled. The input
    processing path is intentionally conservative because concrete Qwen3.5
    checkpoints may expose different multimodal processor details.
    """

    _POOLING_MULTIPLIERS = {
        "attention_masked_mean": 1,
        "image_token_mean": 1,
        "image_spatial_2x2": 4,
        "attention_masked_mean_plus_image_spatial_2x2": 5,
    }

    def __init__(
        self,
        model_id: str,
        *,
        hidden_size: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        local_files_only: bool = True,
        freeze: bool = True,
        pooling: str = "attention_masked_mean",
        prompt_template: str = "{instruction}",
        prompt_mode: str = "auto",
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not model_id:
            raise ValueError("model_id must be provided by configuration.")
        if hidden_size is not None and hidden_size <= 0:
            raise ValueError("hidden_size must be positive when provided.")
        if prompt_mode not in {"auto", "chat_template", "base_multimodal"}:
            raise ValueError(
                "prompt_mode must be 'auto', 'chat_template', or 'base_multimodal'."
            )
        if pooling not in self._POOLING_MULTIPLIERS:
            raise ValueError(
                "pooling must be 'attention_masked_mean', 'image_token_mean', "
                "'image_spatial_2x2', or "
                "'attention_masked_mean_plus_image_spatial_2x2'."
            )

        self.model_id = model_id
        self._hidden_size = hidden_size
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.freeze = freeze
        self.pooling = pooling
        self.prompt_template = prompt_template
        self.prompt_mode = prompt_mode
        self.model_kwargs = dict(model_kwargs or {})
        self.processor_kwargs = dict(processor_kwargs or {})
        self._model: nn.Module | None = None
        self._processor: Any | None = None

    @property
    def hidden_size(self) -> int:
        """Return the configured or loaded model width."""

        if self._hidden_size is None:
            raise RuntimeError(
                "Qwen35Backbone hidden_size is unknown. Provide it in configuration "
                "or call load() before composing the policy."
            )
        return self._hidden_size * self._POOLING_MULTIPLIERS[self.pooling]

    @property
    def is_loaded(self) -> bool:
        """Whether model weights and processor are currently in memory."""

        return self._model is not None and self._processor is not None

    def load(self) -> tuple[nn.Module, Any]:
        """Load a locally available model and processor on demand.

        Transformers is imported inside this method, so importing Rosetta
        Reality does not require it. With the default ``local_files_only``
        setting, a missing checkpoint produces an actionable error instead of a
        network request.
        """

        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        try:
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                "Qwen35Backbone requires the optional 'transformers' dependency. "
                "Install the project's qwen extra without downloading weights."
            ) from exc

        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            model_class = getattr(transformers, "AutoModelForCausalLM", None)
        processor_class = getattr(transformers, "AutoProcessor", None)
        if model_class is None or processor_class is None:
            raise RuntimeError(
                "The installed Transformers version does not expose the auto classes "
                "required by the Qwen3.5 adapter."
            )

        load_kwargs = dict(self.model_kwargs)
        load_kwargs["local_files_only"] = self.local_files_only
        if self.dtype is not None:
            load_kwargs["dtype"] = self.dtype

        processor_load_kwargs = dict(self.processor_kwargs)
        processor_load_kwargs["local_files_only"] = self.local_files_only

        try:
            processor = processor_class.from_pretrained(self.model_id, **processor_load_kwargs)
            model = model_class.from_pretrained(self.model_id, **load_kwargs)
        except OSError as exc:
            mode = "local cache" if self.local_files_only else "configured source"
            raise RuntimeError(
                f"Could not load Qwen checkpoint '{self.model_id}' from the {mode}. "
                "Rosetta never downloads model weights automatically."
            ) from exc

        if self.device is not None:
            model = model.to(self.device)
        if self.freeze:
            model.requires_grad_(False)
            model.eval()

        config = getattr(model, "config", None)
        width = getattr(config, "hidden_size", None)
        if width is None and config is not None:
            width = getattr(getattr(config, "text_config", None), "hidden_size", None)
        if self._hidden_size is None and width is not None:
            self._hidden_size = int(width)
        if self._hidden_size is None:
            raise RuntimeError("Loaded Qwen configuration does not declare a hidden size.")

        self._model = model
        self._processor = processor
        return model, processor

    @staticmethod
    def _pil_image(image: Tensor) -> Image.Image:
        if image.ndim != 3:
            raise ValueError(
                "Qwen image tensors must have shape [channels, height, width], "
                f"received {tuple(image.shape)}."
            )
        if image.shape[0] not in (1, 3, 4):
            raise ValueError(f"Unrecognized Qwen image channels: {tuple(image.shape)}.")
        value = image.detach().to(torch.float32).cpu()
        if value.max().item() <= 1.0:
            value = value.mul(255)
        value = value.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        if value.shape[-1] == 1:
            value = value[..., 0]
        return Image.fromarray(value)

    def prepare_inputs(self, images: Mapping[str, Tensor], instruction: str) -> Any:
        """Create one processor batch with explicit image placeholders and prompt."""

        _, processor = self.load()
        if not images:
            raise ValueError("At least one named image is required for multimodal Qwen input.")
        pil_images = [self._pil_image(image) for image in images.values()]
        prompt = self.prompt_template.format(instruction=instruction)
        chat_template = getattr(processor, "chat_template", None)
        use_chat_template = self.prompt_mode == "chat_template" or (
            self.prompt_mode == "auto" and bool(chat_template)
        )
        if self.prompt_mode == "chat_template" and not chat_template:
            raise RuntimeError(
                "The configured Qwen prompt_mode requires a chat template, but the "
                "local processor does not provide one."
            )
        if not use_chat_template:
            image_token = getattr(processor, "image_token", None)
            vision_start = getattr(processor, "vision_start_token", None)
            vision_end = getattr(processor, "vision_end_token", None)
            if not all(isinstance(value, str) and value for value in (
                image_token,
                vision_start,
                vision_end,
            )):
                raise RuntimeError(
                    "The Base multimodal prompt path requires image, vision-start, "
                    "and vision-end tokens from the local processor."
                )
            image_blocks = [
                f"{vision_start}{image_token}{vision_end}" for _ in pil_images
            ]
            return processor(
                images=pil_images,
                text="\n".join([*image_blocks, prompt]),
                return_tensors="pt",
            )

        content: list[dict[str, Any]] = [
            {"type": "image", "image": image} for image in pil_images
        ]
        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )
        conversation = [{"role": "user", "content": content}]
        return processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

    @staticmethod
    def _config_value(config: Any, name: str) -> Any:
        if isinstance(config, Mapping):
            return config.get(name)
        return getattr(config, name, None)

    def _image_token_mask(
        self,
        encoded: Mapping[str, Any],
        final_hidden: Tensor,
        model: nn.Module,
    ) -> Tensor:
        input_ids = encoded.get("input_ids")
        config = getattr(model, "config", None)
        image_token_id = self._config_value(config, "image_token_id")
        if not isinstance(input_ids, Tensor) or input_ids.shape != final_hidden.shape[:2]:
            raise RuntimeError("Image-token pooling requires aligned input_ids.")
        if not isinstance(image_token_id, int):
            raise RuntimeError("Loaded Qwen configuration does not declare image_token_id.")
        mask = input_ids == image_token_id
        attention_mask = encoded.get("attention_mask")
        if isinstance(attention_mask, Tensor) and attention_mask.shape == mask.shape:
            mask = mask & attention_mask.to(dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise RuntimeError("Image-token pooling found a sample without image tokens.")
        return mask

    def _spatial_image_pool(
        self,
        final_hidden: Tensor,
        image_mask: Tensor,
        encoded: Mapping[str, Any],
        model: nn.Module,
    ) -> Tensor:
        if final_hidden.shape[0] != 1:
            raise RuntimeError("image_spatial_2x2 currently requires feature_batch_size=1.")
        image_grid = encoded.get("image_grid_thw")
        if not isinstance(image_grid, Tensor) or image_grid.ndim != 2 or image_grid.shape[1] != 3:
            raise RuntimeError("image_spatial_2x2 requires image_grid_thw metadata.")
        config = getattr(model, "config", None)
        vision_config = self._config_value(config, "vision_config")
        merge_size = self._config_value(vision_config, "spatial_merge_size")
        if not isinstance(merge_size, int) or merge_size <= 0:
            raise RuntimeError("Loaded Qwen configuration lacks a valid spatial_merge_size.")

        positions = torch.nonzero(image_mask[0], as_tuple=False).flatten()
        split_points = torch.nonzero(positions[1:] != positions[:-1] + 1, as_tuple=False).flatten()
        starts = [0, *(int(index) + 1 for index in split_points)]
        stops = [*starts[1:], int(positions.numel())]
        segments = [positions[start:stop] for start, stop in zip(starts, stops)]
        if len(segments) != image_grid.shape[0]:
            raise RuntimeError("Image-token runs and image_grid_thw entries differ.")

        pooled_images: list[Tensor] = []
        for segment, grid in zip(segments, image_grid):
            temporal, height, width = (int(value) for value in grid.tolist())
            if height % merge_size or width % merge_size:
                raise RuntimeError("Qwen image grid is not divisible by spatial_merge_size.")
            merged_height = height // merge_size
            merged_width = width // merge_size
            expected = temporal * merged_height * merged_width
            if int(segment.numel()) != expected:
                raise RuntimeError(
                    "Image-token count does not match image_grid_thw after spatial merge."
                )
            tokens = final_hidden[0, segment].reshape(
                temporal,
                merged_height,
                merged_width,
                final_hidden.shape[-1],
            )
            spatial = tokens.mean(dim=0)
            rows = torch.tensor_split(spatial, 2, dim=0)
            quadrants = [quadrant for row in rows for quadrant in torch.tensor_split(row, 2, dim=1)]
            if any(quadrant.shape[0] == 0 or quadrant.shape[1] == 0 for quadrant in quadrants):
                raise RuntimeError("image_spatial_2x2 requires at least a 2x2 merged token grid.")
            pooled_images.append(
                torch.cat([quadrant.mean(dim=(0, 1)) for quadrant in quadrants], dim=-1)
            )
        return torch.stack(pooled_images).mean(dim=0, keepdim=True)

    def _pool_final_hidden(
        self,
        final_hidden: Tensor,
        encoded: Mapping[str, Any],
        model: nn.Module,
    ) -> Tensor:
        attention_mask = encoded.get("attention_mask")
        if (
            isinstance(attention_mask, Tensor)
            and attention_mask.shape[1] == final_hidden.shape[1]
        ):
            mask = attention_mask.to(dtype=final_hidden.dtype).unsqueeze(-1)
            denominator = mask.sum(dim=1).clamp_min(1.0)
            attention_pooled = (final_hidden * mask).sum(dim=1) / denominator
        else:
            attention_pooled = final_hidden.mean(dim=1)
        if self.pooling == "attention_masked_mean":
            return attention_pooled

        image_mask = self._image_token_mask(encoded, final_hidden, model)
        if self.pooling == "image_token_mean":
            mask = image_mask.to(dtype=final_hidden.dtype).unsqueeze(-1)
            return (final_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        spatial_pooled = self._spatial_image_pool(final_hidden, image_mask, encoded, model)
        if self.pooling == "attention_masked_mean_plus_image_spatial_2x2":
            return torch.cat((attention_pooled, spatial_pooled), dim=-1)
        return spatial_pooled

    def encode(self, observations: BackboneBatch) -> Tensor:
        """Process image/language inputs and mean-pool the final hidden state."""

        model, processor = self.load()
        encoded = observations.get("encoded_inputs")
        processor_inputs = dict(observations.get("processor_inputs", {}))

        if encoded is None and not processor_inputs:
            images = observations.get("images", observations.get("image"))
            instructions = observations.get("instructions", observations.get("instruction"))
            if isinstance(images, Mapping) and isinstance(instructions, str):
                encoded = self.prepare_inputs(images, instructions)
            else:
                if images is not None:
                    processor_inputs["images"] = images
                if instructions is not None:
                    processor_inputs["text"] = (
                        [instructions] if isinstance(instructions, str) else instructions
                    )
        if encoded is None and not processor_inputs:
            raise ValueError(
                "Qwen35Backbone expects image/language values or a 'processor_inputs' mapping."
            )

        if encoded is None:
            encoded = processor(**processor_inputs, return_tensors="pt")
        if self.device is not None:
            encoded = {
                key: value.to(self.device) if isinstance(value, Tensor) else value
                for key, value in encoded.items()
            }

        outputs = model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            final_hidden = hidden_states[-1]
        else:
            final_hidden = getattr(outputs, "last_hidden_state", None)
        if not isinstance(final_hidden, Tensor):
            raise RuntimeError("Qwen model output did not contain hidden states.")

        return self._pool_final_hidden(final_hidden, encoded, model)
