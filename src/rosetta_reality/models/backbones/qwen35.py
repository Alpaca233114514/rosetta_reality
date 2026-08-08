"""Lazy, local-first adapter skeleton for Qwen3.5 backbones."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from rosetta_reality.models.backbones.base import BackboneBatch, VLABackbone


class Qwen35Backbone(VLABackbone):
    """Load and pool a Qwen3.5 model only when it is first used.

    The adapter defaults to ``local_files_only=True`` so a call cannot silently
    download weights. Device and dtype are caller-controlled. The input
    processing path is intentionally conservative because concrete Qwen3.5
    checkpoints may expose different multimodal processor details.
    """

    def __init__(
        self,
        model_id: str,
        *,
        hidden_size: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        local_files_only: bool = True,
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not model_id:
            raise ValueError("model_id must be provided by configuration.")
        if hidden_size is not None and hidden_size <= 0:
            raise ValueError("hidden_size must be positive when provided.")

        self.model_id = model_id
        self._hidden_size = hidden_size
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.local_files_only = local_files_only
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
        return self._hidden_size

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
            load_kwargs["torch_dtype"] = self.dtype

        processor_load_kwargs = dict(self.processor_kwargs)
        processor_load_kwargs["local_files_only"] = self.local_files_only

        try:
            processor = processor_class.from_pretrained(self.model_id, **processor_load_kwargs)
            model = model_class.from_pretrained(self.model_id, **load_kwargs)
        except OSError as exc:
            mode = "local cache" if self.local_files_only else "configured source"
            raise RuntimeError(
                f"Could not load Qwen checkpoint '{self.model_id}' from the {mode}. "
                "M0 never downloads model weights automatically."
            ) from exc

        if self.device is not None:
            model = model.to(self.device)

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

    def encode(self, observations: BackboneBatch) -> Tensor:
        """Process image/language inputs and mean-pool the final hidden state."""

        model, processor = self.load()
        processor_inputs = dict(observations.get("processor_inputs", {}))

        if not processor_inputs:
            images = observations.get("images", observations.get("image"))
            instructions = observations.get("instructions", observations.get("instruction"))
            if images is not None:
                processor_inputs["images"] = images
            if instructions is not None:
                processor_inputs["text"] = (
                    [instructions] if isinstance(instructions, str) else instructions
                )
        if not processor_inputs:
            raise ValueError(
                "Qwen35Backbone expects image/language values or a 'processor_inputs' mapping."
            )

        encoded = processor(**processor_inputs, return_tensors="pt")
        if self.device is not None:
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

        outputs = model(**encoded, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            final_hidden = hidden_states[-1]
        else:
            final_hidden = getattr(outputs, "last_hidden_state", None)
        if not isinstance(final_hidden, Tensor):
            raise RuntimeError("Qwen model output did not contain hidden states.")

        attention_mask = encoded.get("attention_mask")
        if isinstance(attention_mask, Tensor) and attention_mask.shape[1] == final_hidden.shape[1]:
            mask = attention_mask.to(dtype=final_hidden.dtype).unsqueeze(-1)
            denominator = mask.sum(dim=1).clamp_min(1.0)
            return (final_hidden * mask).sum(dim=1) / denominator
        return final_hidden.mean(dim=1)

