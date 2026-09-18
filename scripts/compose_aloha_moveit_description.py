#!/usr/bin/env python3
"""Compose two official ALOHA VX300S descriptions for MoveIt 2.

Interbotix publishes the ALOHA VX300S links with a ``robot_name`` namespace,
but its joint names are intentionally unprefixed because the supported dual-arm
launch runs two independent ROS namespaces.  A single MoveIt planning scene
needs globally unique joint names.  This build-time adapter expands the official
single-arm xacro twice, prefixes only joint identifiers, and attaches both bases
to the Gym-ALOHA world transforms.  It contains no planner implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import BinaryIO

ARM_JOINTS = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)
MJCF_ARM_LIMITS = {
    "waist": (-3.14158, 3.14158),
    "shoulder": (-1.85005, 1.25664),
    "elbow": (-1.76278, 1.60570),
    "forearm_roll": (-3.14158, 3.14158),
    "wrist_angle": (-1.86750, 2.23402),
    "wrist_rotate": (-3.14158, 3.14158),
}
SIDE_SPECS = {
    "left": {
        "robot_name": "vx300s_left",
        "base_xyz": (-0.469, 0.5, 0.0),
        "base_rpy": (0.0, 0.0, 0.0),
    },
    "right": {
        "robot_name": "vx300s_right",
        "base_xyz": (0.469, 0.5, 0.0),
        "base_rpy": (0.0, 0.0, 3.1416),
    },
}


def _local_name(tag: object) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", maxsplit=1)[-1]


def _read_robot(path: Path, *, label: str) -> ET.Element:
    root = ET.parse(path).getroot()
    if _local_name(root.tag) != "robot":
        raise ValueError(f"{label} must contain one expanded <robot> root.")
    if any(_local_name(element.tag).startswith("xacro:") for element in root.iter()):
        raise ValueError(f"{label} must be expanded before composition.")
    return root


def _prefix_joint_references(root: ET.Element, prefix: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for element in root.iter():
        if _local_name(element.tag) == "joint" and "name" in element.attrib:
            old_name = element.attrib["name"]
            new_name = f"{prefix}{old_name}"
            element.attrib["name"] = new_name
            mapping.setdefault(old_name, new_name)
        if _local_name(element.tag) == "mimic" and "joint" in element.attrib:
            element.attrib["joint"] = f"{prefix}{element.attrib['joint']}"
    return mapping


def _format_vector(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.15g}" for value in values)


def _reconcile_arm_limits(root: ET.Element, side: str) -> None:
    joints = {
        element.attrib["name"]: element
        for element in root.findall("joint")
        if "name" in element.attrib
    }
    for short_name, (mjcf_lower, mjcf_upper) in MJCF_ARM_LIMITS.items():
        joint_name = f"{side}_{short_name}"
        joint = joints.get(joint_name)
        if joint is None:
            raise ValueError(f"Official {side} URDF lacks arm joint {joint_name}.")
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise ValueError(f"Official arm joint {joint_name} lacks finite limits.")
        official_lower = float(limit.attrib["lower"])
        official_upper = float(limit.attrib["upper"])
        effective_lower = max(official_lower, mjcf_lower)
        effective_upper = min(official_upper, mjcf_upper)
        if not effective_lower < effective_upper:
            raise ValueError(f"Official and Gym-ALOHA limits do not overlap for {joint_name}.")
        limit.attrib["lower"] = f"{effective_lower:.15g}"
        limit.attrib["upper"] = f"{effective_upper:.15g}"


def _make_fixed_base_joint(side: str) -> ET.Element:
    spec = SIDE_SPECS[side]
    joint = ET.Element("joint", {"name": f"{side}_base_fixed", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": "world"})
    ET.SubElement(
        joint,
        "child",
        {"link": f"{spec['robot_name']}/base_link"},
    )
    ET.SubElement(
        joint,
        "origin",
        {
            "xyz": _format_vector(spec["base_xyz"]),
            "rpy": _format_vector(spec["base_rpy"]),
        },
    )
    return joint


def _validate_urdf(root: ET.Element) -> None:
    links = {
        element.attrib["name"]
        for element in root
        if _local_name(element.tag) == "link"
    }
    joints = {
        element.attrib["name"]: element
        for element in root
        if _local_name(element.tag) == "joint"
    }
    if len(joints) != sum(
        1 for element in root if _local_name(element.tag) == "joint"
    ):
        raise ValueError("Composed URDF contains duplicate joint names.")
    for side, spec in SIDE_SPECS.items():
        expected = {f"{side}_{name}" for name in ARM_JOINTS}
        missing = expected.difference(joints)
        if missing:
            raise ValueError(f"Composed URDF is missing {side} arm joints: {sorted(missing)}")
        base_link = f"{spec['robot_name']}/base_link"
        if base_link not in links:
            raise ValueError(f"Composed URDF is missing official base link {base_link}.")
    for joint_name, joint in joints.items():
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {joint_name} lacks a parent or child link.")
        for link_name in (parent.attrib.get("link"), child.attrib.get("link")):
            if link_name not in links:
                raise ValueError(f"Joint {joint_name} references unknown link {link_name}.")


def compose_urdf(left_path: Path, right_path: Path) -> tuple[ET.Element, dict[str, dict[str, str]]]:
    output = ET.Element("robot", {"name": "aloha_bimanual"})
    output.append(
        ET.Comment(
            " Generated from the official Interbotix aloha_vx300s.urdf.xacro; "
            "Rosetta changes namespacing and fixed base transforms only. "
        )
    )
    output.append(ET.Element("link", {"name": "world"}))
    material_names: set[str] = set()
    mappings: dict[str, dict[str, str]] = {}
    for side, source_path in (("left", left_path), ("right", right_path)):
        source = deepcopy(_read_robot(source_path, label=f"{side} URDF"))
        for control in list(source):
            if _local_name(control.tag) == "ros2_control":
                source.remove(control)
        mappings[side] = _prefix_joint_references(source, f"{side}_")
        _reconcile_arm_limits(source, side)
        output.append(_make_fixed_base_joint(side))
        for child in source:
            if _local_name(child.tag) == "material":
                name = child.attrib.get("name", "")
                if name in material_names:
                    continue
                material_names.add(name)
            output.append(deepcopy(child))
    _validate_urdf(output)
    return output, mappings


def _rename_srdf_element(element: ET.Element, side: str) -> None:
    prefix = f"{side}_"
    tag = _local_name(element.tag)
    if tag == "group":
        name = element.attrib.get("name")
        if name == "interbotix_arm":
            element.attrib["name"] = f"{side}_arm"
        elif name == "interbotix_gripper":
            element.attrib["name"] = f"{side}_gripper"
    elif tag == "group_state":
        group = element.attrib.get("group")
        if group == "interbotix_arm":
            element.attrib["group"] = f"{side}_arm"
        elif group == "interbotix_gripper":
            element.attrib["group"] = f"{side}_gripper"
    elif tag == "end_effector":
        element.attrib["name"] = f"{side}_gripper"
        element.attrib["group"] = f"{side}_gripper"
    for descendant in element.iter():
        if _local_name(descendant.tag) == "joint" and "name" in descendant.attrib:
            descendant.attrib["name"] = f"{prefix}{descendant.attrib['name']}"


def compose_srdf(left_path: Path, right_path: Path) -> ET.Element:
    output = ET.Element("robot", {"name": "aloha_bimanual"})
    output.append(
        ET.Comment(
            " Per-arm groups and collision exemptions are transformed from the "
            "official Interbotix vx300s.srdf.xacro. "
        )
    )
    bimanual = ET.SubElement(output, "group", {"name": "bimanual"})
    ET.SubElement(bimanual, "group", {"name": "left_arm"})
    ET.SubElement(bimanual, "group", {"name": "right_arm"})
    for side, source_path in (("left", left_path), ("right", right_path)):
        source = _read_robot(source_path, label=f"{side} SRDF")
        for child in source:
            transformed = deepcopy(child)
            _rename_srdf_element(transformed, side)
            output.append(transformed)
        position_group = ET.SubElement(
            output,
            "group",
            {"name": f"{side}_arm_position_priority"},
        )
        ET.SubElement(
            position_group,
            "chain",
            {
                "base_link": f"vx300s_{side}/base_link",
                "tip_link": f"vx300s_{side}/ee_gripper_link",
            },
        )
    group_names = {
        element.attrib["name"]
        for element in output.findall("group")
        if "name" in element.attrib
    }
    expected_groups = {
        "left_arm",
        "right_arm",
        "left_arm_position_priority",
        "right_arm_position_priority",
        "left_gripper",
        "right_gripper",
        "bimanual",
    }
    if not expected_groups.issubset(group_names):
        raise ValueError(
            "Composed SRDF is missing official per-arm groups: "
            f"{sorted(expected_groups.difference(group_names))}"
        )
    return output


def _xml_bytes(root: ET.Element) -> bytes:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(
    stream: BinaryIO,
    *,
    urdf: bytes,
    srdf: bytes,
    mappings: dict[str, dict[str, str]],
    source_commit: str,
    source_archive_sha256: str,
) -> None:
    payload = {
        "schema_version": 1,
        "generator": "scripts/compose_aloha_moveit_description.py",
        "upstream": {
            "repository": "Interbotix/interbotix_ros_manipulators",
            "commit": source_commit,
            "archive_sha256": source_archive_sha256,
            "urdf": "interbotix_xsarm_descriptions/urdf/aloha_vx300s.urdf.xacro",
            "srdf": "interbotix_xsarm_moveit/config/srdf/vx300s.srdf.xacro",
        },
        "base_transforms": SIDE_SPECS,
        "joint_limit_policy": {
            "rule": "intersection_of_official_urdf_and_gym_aloha_mjcf",
            "gym_aloha_mjcf_bounds": MJCF_ARM_LIMITS,
        },
        "arm_joint_order": {
            side: [mappings[side][name] for name in ARM_JOINTS]
            for side in SIDE_SPECS
        },
        "generated": {
            "urdf_sha256": _sha256(urdf),
            "srdf_sha256": _sha256(srdf),
        },
        "planner_implementation": "upstream_moveit2_ompl",
    }
    stream.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-urdf", type=Path, required=True)
    parser.add_argument("--right-urdf", type=Path, required=True)
    parser.add_argument("--left-srdf", type=Path, required=True)
    parser.add_argument("--right-srdf", type=Path, required=True)
    parser.add_argument("--output-urdf", type=Path, required=True)
    parser.add_argument("--output-srdf", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    urdf_root, mappings = compose_urdf(args.left_urdf, args.right_urdf)
    srdf_root = compose_srdf(args.left_srdf, args.right_srdf)
    urdf = _xml_bytes(urdf_root)
    srdf = _xml_bytes(srdf_root)
    _write_new(args.output_urdf, urdf)
    _write_new(args.output_srdf, srdf)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("xb") as stream:
        _write_manifest(
            stream,
            urdf=urdf,
            srdf=srdf,
            mappings=mappings,
            source_commit=args.source_commit,
            source_archive_sha256=args.source_archive_sha256,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
