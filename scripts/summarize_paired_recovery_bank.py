"""Separate registered, real-branch and teacher-feasible recovery-bank denominators."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def partition(registration, source, policy):
    registered=registration["pairs"]
    ids=[p["pair_id"] for p in registered]
    if registration["phase"] != "bank" or len(ids) != 10 or len(set(ids)) != len(ids):
        raise ValueError("fixed ten-parent recovery bank required")
    states={r["pair_id"]:r for r in source["bank"]}
    measurements={r["pair_id"]:r for r in policy.get("rows",[])}
    if (len(states) != len(source["bank"]) or len(measurements) != len(policy.get("rows",[]))
            or not set(states)<=set(ids) or not set(measurements)<=set(ids)):
        raise ValueError("bank identity/denominator drift")
    attempts={r["attempt_id"]:r for r in source["attempts"]}
    subsets={"registered":ids,"actual_branch_created":[i for i in ids if states.get(i,{}).get("branch_state")],
             "teacher_feasible":[i for i in ids if states.get(i,{}).get("feasible") is True]}
    if not set(subsets["teacher_feasible"])<=set(subsets["actual_branch_created"]):
        raise ValueError("teacher-feasible state has no restorable actual branch")
    for identity,row in measurements.items():
        if row.get("teacher_feasible") is not (states.get(identity,{}).get("feasible") is True):
            raise ValueError("evaluation teacher-feasible label differs from real source")
        if row.get("measured_steps",0)>0 and identity not in subsets["actual_branch_created"]:
            raise ValueError("measurement has no real branch")
    results={}
    for name,identities in subsets.items():
        source_task=source_geometry=source_chain=source_unmeasured=0
        measured=task=geometry=chain=outcome_unknown=0
        statuses={}
        for identity in identities:
            attempt=attempts.get(identity+"-R",{})
            source_unmeasured += attempt.get("task_success") is None
            source_task += attempt.get("task_success") is True
            source_geometry += attempt.get("recovery") is not None
            source_chain += attempt.get("verified_recovery_chain") is True
            row=measurements.get(identity,{"status":"not_recorded","measured_steps":0})
            status=row.get("status","not_recorded")
            statuses[status]=statuses.get(status,0)+1
            was_measured=row.get("measured_steps",0)>0
            measured+=was_measured
            outcome_unknown+=row.get("branch_task_success") is None
            task+=was_measured and row.get("branch_task_success") is True
            event=row.get("recovery_event") or {}
            geometry+=was_measured and event.get("geometric_recovery") is True
            chain+=was_measured and event.get("verified_recovery_chain") is True
        results[name]={"denominator":len(identities),"parent_ids":identities,
                       "source_teacher_task_success":source_task,"source_teacher_geometry_recovery":source_geometry,
                       "source_teacher_verified_chain":source_chain,"source_teacher_unmeasured":source_unmeasured,
                       "policy_measured":measured,"policy_unmeasured":len(identities)-measured,
                       "policy_task_outcome_unknown":outcome_unknown,
                       "policy_task_success":task,"policy_geometry_recovery":geometry,"policy_verified_chain":chain,
                       "policy_statuses":statuses}
    return {"subsets":results,"unproven_source_branch_count":len(subsets["actual_branch_created"])-len(subsets["teacher_feasible"]),
            "unproven_source_branches_are_exploratory":True,
            "legacy_object_scene_nonoverlap":"not_established","teacher_qualified":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",required=True,type=Path)
    parser.add_argument("--source-audit",required=True,type=Path)
    parser.add_argument("--policy-summary",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    root=args.bank.resolve()
    manifest_raw=(root/"manifest.json").read_bytes()
    manifest=json.loads(manifest_raw)
    audit_raw=args.source_audit.read_bytes()
    audit=json.loads(audit_raw)
    summary_raw=(root/"summary.json").read_bytes()
    if (audit.get("passed") is not True or audit.get("manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest()
            or audit.get("summary_sha256") != hashlib.sha256(summary_raw).hexdigest()):
        raise ValueError("bank source audit not bound to original payload identities")
    inventory={m["path"]:m for m in manifest["files"]}
    if len(inventory) != len(manifest["files"]):
        raise ValueError("duplicate source member")
    for relative,member in inventory.items():
        part=PurePosixPath(relative)
        target=root.joinpath(*part.parts).resolve()
        if part.is_absolute() or ".." in part.parts or "\\" in relative or ":" in relative or not target.is_relative_to(root):
            raise ValueError("unsafe source member")
        raw=target.read_bytes()
        if len(raw) != member["bytes"] or hashlib.sha256(raw).hexdigest() != member["sha256"]:
            raise ValueError("source changed after original audit")
    registration=json.loads((root/"registration.json").read_bytes())
    policy_raw=args.policy_summary.read_bytes()
    result=partition(registration,json.loads(summary_raw),json.loads(policy_raw))
    result.update(schema="rosetta.paired_recovery_bank_partition.v1",
                  bank_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                  bank_summary_sha256=hashlib.sha256(summary_raw).hexdigest(),
                  bank_audit_sha256=hashlib.sha256(audit_raw).hexdigest(),
                  policy_summary_sha256=hashlib.sha256(policy_raw).hexdigest(),
                  policy_trace_independently_recomputed_by_this_script=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))
    return 0


if __name__=="__main__":main()
