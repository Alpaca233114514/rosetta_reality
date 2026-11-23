"""Read-only sealed schedule exposure accounting, no ML/data/image imports."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def summarize(schedule, original, auxiliary):
    if schedule.get("status") != "sealed":
        return {"status":schedule.get("status"),"actual_input_exposures":0,
                "planned_updates":schedule.get("planned_updates"),"training_executed_claimed":False}
    entries=schedule["entries"]
    if digest(entries) != schedule["entries_sha256"]:
        raise ValueError("sealed schedule entries changed")
    if len(entries) != schedule["updates"]*4 or schedule["batch_size"] != 4:
        raise ValueError("registered input budget disagreement")
    inventories={"D":{},"S":{},"R":{}}
    if original.get("hidden_test_loaded") is not False or original.get("source_split") != "train":
        raise ValueError("original inventory is not isolated D train")
    for row in original["inputs"]:
        identity=(row["episode"],row["frame"])
        if identity in inventories["D"]:
            raise ValueError("D inventory duplicates")
        valid,pad=row["valid_target_slots"],row["padding_slots"]
        if any(type(v) is not int or v<0 for v in (valid,pad)) or valid+pad != 50:
            raise ValueError("D native chunk accounting")
        inventories["D"][identity]=(valid,pad)
    for source in ("S","R"):
        for row in auxiliary.get(source+"_inputs",[]):
            if row.get("positive_supervision") is not True or row.get("chunk_crosses_unqualified_action") is not False:
                raise ValueError("auxiliary supervision not admitted")
            identity=(row["episode"],row["frame"])
            if identity in inventories[source]:
                raise ValueError("auxiliary inventory duplicates")
            inventories[source][identity]=(50,0)
    counts=Counter()
    groups={}
    common=[]
    for index,entry in enumerate(entries):
        if entry["update"] != index//4+1 or entry["slot"] != index%4:
            raise ValueError("schedule ordering drift")
        source=entry["source"]
        identity=(entry["episode"],entry["frame"])
        if source not in inventories or identity not in inventories[source]:
            raise ValueError("scheduled sample absent from real admitted inventory")
        if index%4<3:
            if source != "D":
                raise ValueError("common slots not D")
            common.append({"episode":identity[0],"frame":identity[1]})
        valid,pad=inventories[source][identity]
        key=source+":"+("common_D" if index%4<3 else "auxiliary")
        group=groups.setdefault(key,{"input_exposures":0,"valid_target_slot_exposures":0,"padding_slot_exposures":0})
        group["input_exposures"]+=1
        group["valid_target_slot_exposures"]+=valid
        group["padding_slot_exposures"]+=pad
        counts[(source,*identity)]+=1
    unique_valid=sum(inventories[s][(ep,frame)][0] for s,ep,frame in counts)
    unique_pad=sum(inventories[s][(ep,frame)][1] for s,ep,frame in counts)
    return {"status":"sealed_metadata_accounted","updates":schedule["updates"],
            "planned_input_exposures":len(entries),"unique_input_frames":len(counts),
            "repeated_input_exposures":len(entries)-len(counts),
            "unique_frame_valid_target_slots":unique_valid,"unique_frame_padding_slots":unique_pad,
            "valid_target_slot_exposures":sum(v["valid_target_slot_exposures"] for v in groups.values()),
            "padding_slot_exposures":sum(v["padding_slot_exposures"] for v in groups.values()),
            "groups":groups,"common_D_entries_sha256":digest(common),
            "frame_exposure_min":min(counts.values()),"frame_exposure_max":max(counts.values()),
            "unit":"native temporal action slots, not padded feature dimensions",
            "training_executed_claimed":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original",required=True,type=Path)
    parser.add_argument("--cohort",required=True,type=Path)
    for arm in "abc":
        parser.add_argument("--schedule-"+arm,required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    original=json.loads(args.original.read_bytes())
    cohort=json.loads(args.cohort.read_bytes())
    arms={}
    hashes={"original":hashlib.sha256(args.original.read_bytes()).hexdigest(),
            "cohort":hashlib.sha256(args.cohort.read_bytes()).hexdigest()}
    for arm in "ABC":
        path=getattr(args,"schedule_"+arm.lower())
        raw=path.read_bytes()
        hashes[arm]=hashlib.sha256(raw).hexdigest()
        arms[arm]=summarize(json.loads(raw),original,cohort)
    common={v["common_D_entries_sha256"] for v in arms.values() if "common_D_entries_sha256" in v}
    if len(common)>1:
        raise ValueError("common D exposures differ between arms")
    result={"schema":"rosetta.paired_schedule_exposures.v1","source_file_sha256":hashes,
            "arms":arms,"common_D_schedule_exact":len(common)==1,
            "fixed_input_budget_is_not_equal_valid_target_budget":True,
            "statistics_only_no_training_or_hidden_data_loaded":True,
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
