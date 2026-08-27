"""Collision-report reclassification tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.reclassify_smolvla_collision_reports as reclassification
from scripts.reclassify_smolvla_collision_reports import (
    _parse_pair,
    _reclassify_report,
)


class _Classifier:
    @staticmethod
    def is_unexpected_collision_pair(first: str, second: str) -> bool:
        return "table" in {first, second}


def test_parse_pair_rejects_ambiguous_serialization() -> None:
    assert _parse_pair("red_peg <-> finger") == ("red_peg", "finger")
    with pytest.raises(ValueError, match="Invalid serialized"):
        _parse_pair("red_peg")
    with pytest.raises(ValueError, match="Invalid serialized"):
        _parse_pair("a <-> b <-> c")


def test_reclassify_report_preserves_source_and_recounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reclassification, "REPOSITORY_ROOT", tmp_path)
    path = tmp_path / "episode.json"
    payload = {
        "seed": 1001,
        "metrics": {
            "unexpected_collisions": 5,
            "unexpected_collision_pairs": {
                "red_peg <-> finger": 3,
                "table <-> finger": 2,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _reclassify_report(path, _Classifier())  # type: ignore[arg-type]

    assert result["recorded_unexpected_contacts"] == 5
    assert result["reclassified_unexpected_contacts"] == 2
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_reclassify_report_rejects_incomplete_histogram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reclassification, "REPOSITORY_ROOT", tmp_path)
    path = tmp_path / "episode.json"
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "unexpected_collisions": 2,
                    "unexpected_collision_pairs": {"table <-> finger": 1},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="histogram total differs"):
        _reclassify_report(path, _Classifier())  # type: ignore[arg-type]
