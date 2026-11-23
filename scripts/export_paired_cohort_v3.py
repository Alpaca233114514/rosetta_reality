"""Shared-prefix cohort export requires independent v3 physical/input audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from export_paired_cohort_v2 import export as original_export


def require_fork_audit(audit, summary):
    if (audit.get("collection_protocol") != "shared_real_prefix_causal_fork_v3"
            or summary.get("collection_protocol") != audit["collection_protocol"]
            or audit.get("shared_cached_branch_input_verified") is not True
            or audit.get("postfork_render_bitwise_equality_claimed") is not False
            or audit.get("physical_steps_verified") != summary.get("environment_steps")
            or audit.get("logical_trajectory_frames") != summary.get("logical_trajectory_frames")
            or audit.get("shared_prefix_physical_steps") != summary.get("shared_prefix_physical_steps")):
        raise ValueError("independent shared-prefix audit missing or counters disagree")


def export(root,audit_path,camera_map):
    audit=json.loads(audit_path.read_bytes())
    summary=json.loads((root/"summary.json").read_bytes())
    require_fork_audit(audit,summary)
    result=original_export(root,audit_path,camera_map)
    result.update(collection_protocol=summary["collection_protocol"],
                  shared_cached_branch_input_verified=True,
                  shared_prefix_physical_steps=summary["shared_prefix_physical_steps"],
                  logical_trajectory_frames=summary["logical_trajectory_frames"],
                  physical_steps_verified=audit["physical_steps_verified"],
                  postfork_render_bitwise_equality_claimed=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--audit",required=True,type=Path)
    parser.add_argument("--camera-map",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    result=export(args.root,args.audit,json.loads(args.camera_map.read_bytes()))
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
