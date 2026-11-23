"""Bounded actual EGL/NPZ/state cost measurement; no expert labels or model."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    import numpy as np
    import torch
    from check_q0_state_restore import PythonState
    from diagnose_teacher_cross_pose import finite_json
    from rosetta_reality.eval.reproducibility_capture import PhysicsReader
    from rosetta_reality.sim import GymAlohaEnvironment,load_action_contract
    from rosetta_reality.sim.scripted_teacher_observer import build_observation
    contract=load_action_contract(ROOT/"configs/sim/aloha_insertion_smolvla.yaml")
    env=GymAlohaEnvironment(contract,maximum_episode_steps=500)
    samples=[]
    error=None
    reset_seconds=None
    snapshots=[]
    try:
        tick=time.monotonic()
        current=env.reset(seed=2400)
        reset_seconds=time.monotonic()-tick
        reader=PhysicsReader(env)
        for step in range(2):
            if time.monotonic()-started>120:raise TimeoutError("120-second benchmark cap")
            if sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())>64*1024**2:raise OSError("64MiB benchmark cap")
            before=finite_json(build_observation(env,contract))
            image_path=args.output/f"frame-{step:04d}.npz"
            pixels=current["raw"].get("pixels",current["raw"].get("images",{}))
            arrays={str(k):np.asarray(v) for k,v in pixels.items()}
            if not arrays or any(a.dtype!=np.uint8 or a.ndim!=3 for a in arrays.values()):raise ValueError("Native RGB pixels unavailable")
            tick=time.monotonic()
            np.savez_compressed(image_path,**arrays)
            encode=time.monotonic()-tick
            action=current["robot_state"].clone()
            contract.validate_tensor(action,allow_chunk=False)
            if contract.clip(action)[1].any():raise ValueError("Hold command would clip")
            tick=time.monotonic()
            current,reward,done,info=env.step(action)
            elapsed=time.monotonic()-tick
            after=finite_json({"integration":reader()["integration"],"geometry":build_observation(env,contract),"reward":reward,"done":done,"info":info,"contacts":env.contact_pairs(),"joint_limits":env.state_limit_violation_count(),"unexpected_collisions":env.unexpected_collision_count()})
            safe=not(after["joint_limits"] or after["unexpected_collisions"] or bool(env.last_clip_mask.any()))
            row={"step":step,"pts_seconds":step/50,"images":image_path.name,"images_sha256":hashlib.sha256(image_path.read_bytes()).hexdigest(),"state":before["robot_state"],"action":finite_json(action),"task":"Insert the peg into the socket.","action_source":"registered_hold_benchmark","positive_target":False,"before":before,"after":after,"phase":"benchmark","safe":safe}
            samples.append({"step":step,"compressed_image_bytes":image_path.stat().st_size,"step_seconds":elapsed,"encode_seconds":encode,"compact_row_bytes":len(json.dumps(row,sort_keys=True).encode())+1,"pretty_row_bytes":len(json.dumps(row,sort_keys=True,indent=2).encode())+1,"shapes":{k:list(v.shape) for k,v in arrays.items()}})
            if step in (0,1):
                python=PythonState([env],reader.physics)
                receipt={"integration_spec":reader.spec,"physics_model_sha256":reader.model_sha,"python_state":python.receipt(),"exact_replays":["0"*64,"0"*64],"replays":[{"step":after,"images":{},"python":python.receipt()}]*2,"passed":False,"engineering_only_cost_record":True}
                snapshot_bytes=json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
                with (args.output/f"state-cost-{step:04d}.json").open("xb") as stream:stream.write(snapshot_bytes)
                snapshots.append(len(snapshot_bytes))
            if not safe or done:break
    except Exception as exc:error=type(exc).__name__
    finally:env.close()
    maximum=lambda key:max((s[key] for s in samples),default=0)
    frame_cost=maximum("compressed_image_bytes")+maximum("compact_row_bytes")+maximum("pretty_row_bytes")+2048
    snapshot_cost=max(snapshots,default=0)
    result={"schema":"rosetta.paired_collection_cost.v1","status":"measured" if error is None else "incomplete","error_type":error,"seed":2400,"frames":samples,"reset_seconds":reset_seconds,"maximum_frame_cost_bytes":frame_cost,"maximum_restore_receipt_bytes":snapshot_cost,"Q1_worst_case_bytes":frame_cost*4000+snapshot_cost*8+16*1024**2,"Q2_worst_case_bytes":frame_cost*20000+snapshot_cost*40+16*1024**2,"step_seconds_median":statistics.median(s["step_seconds"] for s in samples) if samples else None,"wall_seconds":time.monotonic()-started,"model_forwards":0,"optimizer_updates":0,"training_inputs_created":False,"shutdown_requested":False}
    with (args.output/"result.json").open("x") as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write("\n")
    print(json.dumps({k:result[k] for k in ("status","error_type","maximum_frame_cost_bytes","maximum_restore_receipt_bytes","Q1_worst_case_bytes","Q2_worst_case_bytes","wall_seconds")}))
    return 1 if error else 0


if __name__=="__main__":raise SystemExit(main())
