import json
from pathlib import Path

import pytest

from rosetta_reality.integration import ActionPlan, ActionTarget

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _payload() -> dict:
    return {
        "schema_version": 1,
        "subtask": "grasp",
        "object": "red_peg",
        "target": {
            "kind": "point_3d",
            "values": [0.1, -0.2, 0.3],
            "reference_frame": "robot_base",
            "unit": "meter",
        },
        "motion_hint": "approach from above",
        "constraints": ["avoid_socket_rim"],
        "success_condition": "peg is held by both grippers",
        "replan_condition": "target is no longer visible",
    }


def test_action_plan_v1_round_trip_is_json_compatible() -> None:
    plan = ActionPlan.from_dict(_payload())

    assert plan.target == ActionTarget(
        kind="point_3d",
        values=(0.1, -0.2, 0.3),
        reference_frame="robot_base",
        unit="meter",
    )
    assert json.loads(json.dumps(plan.to_dict())) == _payload()


def test_action_plan_rejects_extra_fields_and_ambiguous_target_units() -> None:
    payload = _payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="extra"):
        ActionPlan.from_dict(payload)

    payload = _payload()
    payload["target"]["unit"] = ""
    with pytest.raises(ValueError, match="target.unit"):
        ActionPlan.from_dict(payload)


def test_action_plan_rejects_nonfinite_coordinates_and_duplicate_constraints() -> None:
    payload = _payload()
    payload["target"]["values"] = [float("nan")]
    with pytest.raises(ValueError, match="finite"):
        ActionPlan.from_dict(payload)

    payload = _payload()
    payload["constraints"] = ["avoid_socket_rim", "avoid_socket_rim"]
    with pytest.raises(ValueError, match="duplicates"):
        ActionPlan.from_dict(payload)


def test_action_plan_rejects_boolean_numbers_and_schema_versions() -> None:
    payload = _payload()
    payload["target"]["values"] = [True]
    with pytest.raises(ValueError, match="finite"):
        ActionPlan.from_dict(payload)

    payload = _payload()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        ActionPlan.from_dict(payload)


def test_json_schema_declares_the_same_required_wire_fields() -> None:
    schema = json.loads(
        (REPOSITORY_ROOT / "integration/schemas/action_plan.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(_payload())
    assert schema["properties"]["subtask"]["pattern"] == "\\S"
    assert schema["properties"]["object"]["pattern"] == "\\S"
    assert schema["properties"]["motion_hint"]["pattern"] == "\\S"
    assert schema["properties"]["constraints"]["items"]["pattern"] == "\\S"
    assert schema["properties"]["success_condition"]["pattern"] == "\\S"
    assert schema["properties"]["replan_condition"]["pattern"] == "\\S"
    target = schema["properties"]["target"]["oneOf"][1]["properties"]
    assert target["reference_frame"]["pattern"] == "\\S"
    assert target["unit"]["pattern"] == "\\S"
