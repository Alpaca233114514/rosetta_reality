"""Independent shared-prefix provenance/physical accounting plus v2 label audit."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("_original_paired_audit",Path(__file__).with_name("audit_paired_collection_v2.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def accounting(logical_frames, prefix_physical, restore_steps):
    if any(type(v) is not int or v < 0 for v in (logical_frames,prefix_physical,restore_steps)):
        raise ValueError("invalid physical counter")
    if prefix_physical > logical_frames:
        raise ValueError("prefix counter exceeds logical frames")
    return logical_frames-prefix_physical+restore_steps


def audit(root):
    import numpy as np
    root = root.resolve()
    result = base.audit(root)
    inventory = base.manifest_inventory(json.loads((root/"manifest.json").read_bytes()))
    read = lambda path:json.loads(base.verified_bytes(root,path,inventory))
    plan,summary = read("registration.json"),read("summary.json")
    if plan.get("collection_protocol") != "shared_real_prefix_causal_fork_v3" or summary.get("collection_protocol") != plan["collection_protocol"]:
        raise ValueError("shared prefix protocol identity")
    logical = sum(a["rows"] for a in summary["attempts"])
    prefix_physical = 0
    reconstructed_calls = 0
    for fork in summary["forks"]:
        prefix = fork["pair_id"]+"/"
        if read(prefix+"fork-protocol.json") != fork:
            raise ValueError("fork receipt disagreement")
        shared = read(prefix+"shared-prefix/rows.json")
        rows = shared["rows"]
        if shared["actual_physical_steps"] != len(rows) or fork["shared_prefix_physical_steps"] != len(rows):
            raise ValueError("shared physical denominator")
        prefix_physical += len(rows)
        for role in ("S","R"):
            labels = read(prefix+role+"/labels.json")["rows"]
            if len(labels) < len(rows):
                raise ValueError("shared prefix missing from logical trajectory")
            for i,row in enumerate(rows):
                if any(labels[i][key] != row[key] for key in row if key != "positive_target"):
                    raise ValueError("shared prefix replaced or mutated")
                if not row["images"].startswith(prefix+"shared-prefix/"):
                    raise ValueError("prefix image lacks shared original lineage")
        d = fork["fork_step"]
        if d is None:
            continue
        if d != len(rows):
            raise ValueError("fork index is not common prefix boundary")
        history = read(prefix+"teacher-history.json")
        if len(history) != d+1 or fork["teacher_history_calls"] != len(history):
            raise ValueError("teacher reconstruction denominator")
        for i,row in enumerate(rows):
            if history[i] != {"geometry":row["before"],"action":row["action"],"phase":row["phase"]}:
                raise ValueError("teacher history not actual common geometry/action")
        replays = fork["teacher_reconstruction"]
        if len(replays) != 2 or {r["role"] for r in replays} != {"S","R"}:
            raise ValueError("two restored branch histories not evidenced")
        if any(r["calls"] != len(history) or r["all_decisions_exact"] is not True for r in replays):
            raise ValueError("teacher reconstruction not exact")
        reconstructed_calls += 2*len(history)
        base.verified_bytes(root,prefix+"fork-integration.npz",inventory)
        receipt = base.receipt_json(root,prefix+"fork-python.json.gz",inventory)
        if not receipt.get("python_state") or not receipt.get("source_sha256"):
            raise ValueError("complete fork Python/RNG receipt missing")
        native_path = prefix+"fork-native-observation.npz"
        base.verified_bytes(root,native_path,inventory)
        with np.load(base.safe(root,native_path),allow_pickle=False) as native:
            identities = {k:{"sha256":hashlib.sha256(native[k].tobytes()).hexdigest(),"dtype":native[k].dtype.str,"shape":list(native[k].shape)} for k in native.files}
        if identities != fork["cached_native_images"]:
            raise ValueError("fork cached native identity")
        for role in ("S","R"):
            labels = read(prefix+role+"/labels.json")["rows"]
            if len(labels) <= d:
                raise ValueError("registered restored branch not executed")
            first = labels[d]
            if first["state"] != fork["cached_state"] or first["before"] != history[-1]["geometry"] or first["phase"] != history[-1]["phase"]:
                raise ValueError("actual branch input not saved cached state/geometry")
            if role == "S" and first["action"] != history[-1]["action"]:
                raise ValueError("ordinary branch fork action changed")
            base.verified_bytes(root,first["images"],inventory)
            with np.load(base.safe(root,first["images"]),allow_pickle=False) as image:
                actual = {k:{"sha256":hashlib.sha256(image[k].tobytes()).hexdigest(),"dtype":image[k].dtype.str,"shape":list(image[k].shape)} for k in image.files}
            if actual != identities:
                raise ValueError("actual native branch first observation differs from captured fork input")
    restore_steps = 0
    for relative in inventory:
        if relative.endswith(("-restore.json","-restore.json.gz")):
            restore_steps += base.receipt_json(root,relative,inventory)["engineering_steps"]
    physical = accounting(logical,prefix_physical,restore_steps)
    if (logical != summary["logical_trajectory_frames"] or prefix_physical != summary["shared_prefix_physical_steps"]
            or physical != summary["environment_steps"] or reconstructed_calls != summary["teacher_reconstruction_decision_calls"]):
        raise ValueError("physical/logical/reconstruction accounting discrepancy")
    result.update(collection_protocol=plan["collection_protocol"],shared_prefix_physical_steps=prefix_physical,
                  logical_trajectory_frames=logical,physical_steps_verified=physical,
                  engineering_restore_steps=restore_steps,teacher_reconstruction_decision_calls=reconstructed_calls,
                  shared_cached_branch_input_verified=True,
                  postfork_render_bitwise_equality_claimed=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    try:
        result=audit(args.root)
    except Exception as error:
        result={"schema":"rosetta.paired_collection_audit.v2","passed":False,"error_type":type(error).__name__,"reason":str(error)}
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
