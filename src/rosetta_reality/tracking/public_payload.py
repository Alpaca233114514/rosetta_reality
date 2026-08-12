"""Reject secrets and machine identity before data can enter a public dashboard."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

_KEY_SEGMENT = re.compile(r"[^a-z0-9]+")
_TOKEN_VALUE = re.compile(r"(?i)\bhf_[a-z0-9]{16,}\b")
_WINDOWS_PATH = re.compile(r"(?i)(?:^|[\s='\"])[a-z]:[\\/]")
_POSIX_PATH = re.compile(r"(?:^|[\s='\"(])/(?!/)[^\s'\"?&#]+")
_UNC_PATH = re.compile(r"(?:^|[\s='\"])\\\\")
_URL_QUERY = re.compile(r"(?i)https?://[^\s]+\?[^\s]+")
_SENSITIVE_KEY_SEGMENTS = frozenset(
    {
        "api",
        "authorization",
        "cookie",
        "credential",
        "env",
        "environment",
        "key",
        "password",
        "secret",
        "signed",
        "token",
        "webhook",
    }
)


def _key_segments(key: str) -> set[str]:
    return {segment for segment in _KEY_SEGMENT.split(key.lower()) if segment}


def _validate_key(key: Any, context: str) -> str:
    if not isinstance(key, str) or not key:
        raise ValueError(f"{context} keys must be non-empty strings.")
    matches = _key_segments(key) & _SENSITIVE_KEY_SEGMENTS
    if matches:
        raise ValueError(f"{context} contains a forbidden sensitive key segment.")
    return key


def _validate_string(value: str, context: str) -> None:
    if _TOKEN_VALUE.search(value):
        raise ValueError(f"{context} contains a credential-shaped value.")
    if (
        _WINDOWS_PATH.search(value)
        or _POSIX_PATH.search(value)
        or _UNC_PATH.search(value)
        or value.startswith("file://")
    ):
        raise ValueError(f"{context} contains a machine-specific path.")
    if _URL_QUERY.search(value):
        raise ValueError(f"{context} contains a URL query string.")


def validate_public_payload(value: Any, *, context: str = "payload") -> None:
    """Recursively prove that a Trackio payload contains only public-safe values."""

    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite number.")
        return
    if isinstance(value, str):
        _validate_string(value, context)
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _validate_key(raw_key, context)
            validate_public_payload(child, context=f"{context}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            validate_public_payload(child, context=f"{context}[{index}]")
        return
    raise ValueError(f"{context} contains unsupported type {type(value).__name__}.")


def sanitize_metrics(metrics: Mapping[str, Any], *, mode: str) -> dict[str, int | float]:
    """Keep finite numeric metrics only and prefix them with the train/eval role."""

    if mode not in {"train", "eval", "system"}:
        raise ValueError(f"Unsupported metric mode: {mode!r}.")
    sanitized: dict[str, int | float] = {}
    for raw_key, value in metrics.items():
        key = _validate_key(raw_key, "metrics")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            raise ValueError(f"Metric {key!r} is not finite.")
        public_key = f"{mode}/{key}"
        validate_public_payload({public_key: value}, context="metrics")
        sanitized[public_key] = value
    return sanitized
