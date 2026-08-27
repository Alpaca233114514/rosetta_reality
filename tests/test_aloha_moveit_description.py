"""Deterministic official ALOHA-to-MoveIt description composition tests."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.compose_aloha_moveit_description import (
    ARM_JOINTS,
    MJCF_ARM_LIMITS,
    compose_srdf,
    compose_urdf,
)


def _write_arm_urdf(path: Path, robot_name: str) -> None:
    robot = ET.Element("robot", {"name": robot_name})
    ET.SubElement(robot, "material", {"name": "interbotix_black"})
    parent = f"{robot_name}/base_link"
    ET.SubElement(robot, "link", {"name": parent})
    for index, name in enumerate((*ARM_JOINTS, "left_finger", "right_finger")):
        child = f"{robot_name}/link_{index}"
        ET.SubElement(robot, "link", {"name": child})
        joint = ET.SubElement(robot, "joint", {"name": name, "type": "revolute"})
        ET.SubElement(joint, "limit", {"lower": "-4", "upper": "4"})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": child})
        if name == "right_finger":
            ET.SubElement(joint, "mimic", {"joint": "left_finger", "multiplier": "-1"})
        parent = child
    control = ET.SubElement(robot, "ros2_control", {"name": "XSHardwareInterface"})
    ET.SubElement(control, "joint", {"name": "waist"})
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def _write_arm_srdf(path: Path, robot_name: str) -> None:
    robot = ET.Element("robot", {"name": robot_name})
    arm = ET.SubElement(robot, "group", {"name": "interbotix_arm"})
    for name in ARM_JOINTS:
        ET.SubElement(arm, "joint", {"name": name})
    gripper = ET.SubElement(robot, "group", {"name": "interbotix_gripper"})
    ET.SubElement(gripper, "link", {"name": f"{robot_name}/link_6"})
    state = ET.SubElement(
        robot,
        "group_state",
        {"name": "Home", "group": "interbotix_arm"},
    )
    ET.SubElement(state, "joint", {"name": "waist", "value": "0"})
    ET.SubElement(
        robot,
        "end_effector",
        {
            "name": "interbotix_gripper",
            "parent_link": f"{robot_name}/link_7",
            "group": "interbotix_gripper",
        },
    )
    ET.SubElement(
        robot,
        "disable_collisions",
        {
            "link1": f"{robot_name}/base_link",
            "link2": f"{robot_name}/link_0",
            "reason": "Adjacent",
        },
    )
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def test_compose_urdf_prefixes_joint_identity_and_preserves_official_links(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.urdf"
    right = tmp_path / "right.urdf"
    _write_arm_urdf(left, "vx300s_left")
    _write_arm_urdf(right, "vx300s_right")

    root, mappings = compose_urdf(left, right)

    joint_names = {joint.attrib["name"] for joint in root.findall("joint")}
    assert {f"left_{name}" for name in ARM_JOINTS}.issubset(joint_names)
    assert {f"right_{name}" for name in ARM_JOINTS}.issubset(joint_names)
    assert "left_base_fixed" in joint_names
    assert "right_base_fixed" in joint_names
    assert not root.findall("ros2_control")
    assert mappings["left"]["waist"] == "left_waist"
    for side in ("left", "right"):
        for short_name, expected in MJCF_ARM_LIMITS.items():
            joint = next(
                item
                for item in root.findall("joint")
                if item.attrib["name"] == f"{side}_{short_name}"
            )
            limit = joint.find("limit")
            assert (float(limit.attrib["lower"]), float(limit.attrib["upper"])) == expected
    right_finger = next(
        joint for joint in root.findall("joint") if joint.attrib["name"] == "right_right_finger"
    )
    assert right_finger.find("mimic").attrib["joint"] == "right_left_finger"


def test_compose_srdf_retains_official_groups_states_and_collision_exemptions(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.srdf"
    right = tmp_path / "right.srdf"
    _write_arm_srdf(left, "vx300s_left")
    _write_arm_srdf(right, "vx300s_right")

    root = compose_srdf(left, right)

    groups = {group.attrib["name"]: group for group in root.findall("group")}
    assert set(groups) == {
        "bimanual",
        "left_arm",
        "left_arm_position_priority",
        "left_gripper",
        "right_arm",
        "right_arm_position_priority",
        "right_gripper",
    }
    assert [joint.attrib["name"] for joint in groups["left_arm"].findall("joint")] == [
        f"left_{name}" for name in ARM_JOINTS
    ]
    for side in ("left", "right"):
        chain = groups[f"{side}_arm_position_priority"].find("chain")
        assert chain is not None
        assert chain.attrib == {
            "base_link": f"vx300s_{side}/base_link",
            "tip_link": f"vx300s_{side}/ee_gripper_link",
        }
    states = root.findall("group_state")
    assert {state.attrib["group"] for state in states} == {"left_arm", "right_arm"}
    assert {state.find("joint").attrib["name"] for state in states} == {
        "left_waist",
        "right_waist",
    }
    assert len(root.findall("disable_collisions")) == 2


def test_planner_selects_registered_arm_groups_without_srdf_enumeration() -> None:
    source = (
        Path("integration/aloha_moveit2/src/aloha_moveit_planner.cpp")
        .read_text(encoding="utf-8")
    )

    assert "bimanual_group_->getSubgroups" not in source
    assert "model_->getJointModelGroup(group_name)" in source
    assert "kPositionPriorityArmGroups[subgroup] : kArmGroups[subgroup]" in source
    assert "bimanual_registered_subgroup_missing" in source
