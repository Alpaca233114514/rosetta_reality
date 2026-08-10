from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosetta_reality.data import ordered_feature_names


def _write_info(root: Path, feature: dict[str, object]) -> None:
    meta = root / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps({"features": {"action": feature}}),
        encoding="utf-8",
    )


def test_ordered_feature_names_flattens_lerobot_semantic_groups(tmp_path: Path) -> None:
    _write_info(
        tmp_path,
        {
            "shape": [4],
            "names": {"left": ["left_0", "left_1"], "right": ["right_0", "right_1"]},
        },
    )

    assert ordered_feature_names(tmp_path, "action") == (
        "left_0",
        "left_1",
        "right_0",
        "right_1",
    )


def test_ordered_feature_names_rejects_shape_only_compatibility(tmp_path: Path) -> None:
    _write_info(tmp_path, {"shape": [4], "names": {"motors": ["only_one"]}})

    with pytest.raises(ValueError, match="do not match its vector shape"):
        ordered_feature_names(tmp_path, "action")
