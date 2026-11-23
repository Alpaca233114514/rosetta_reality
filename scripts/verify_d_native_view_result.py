"""Independent arithmetic/identity audit of received D native small evidence."""
import argparse
import hashlib
import json
from pathlib import Path

TRAIN=(49,4,23,43,21,37,18,34,0,47,38,29,3,26,14,17,44,30,15,42,10,35,25,32,19,36,41,28,8,27,16,11,2,20,9,39,46,48,12,40)
NORM="263880ec3adfddb8517a50fa5483e7c8f32f0c208243c229cbc72f8e9cf8d988"


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def verify(root):
    view=root/"view";manifest=read(view/"view-manifest.json");proof=read(view/"native-read-verification.json")
    assert proof["status"]=="passed" and proof["native_read_frames"]==20000 and proof["native_episode_count"]==40
    assert proof["view_manifest_sha256"]==digest(view/"view-manifest.json")
    assert proof["label_parity"]=="all_selected_states_and_chunk_actions_exact" and proof["pixel_parity"]=="all_selected_frames_exact"
    assert proof["native_action_query"]=="per_dataset_verified_train_only_tensor_cache"
    assert set(proof["original_episodes"])==set(TRAIN) and manifest["all_40_episodes_present"] is True
    assert manifest["normalization_sha256"]==NORM and digest(view/"normalization.json")==NORM
    assert manifest["protected_rows_exported"]==0 and manifest["normal_training_source_mixed_parquet_or_video_required"] is False
    assert manifest["shared_video_preroll_may_be_decoded_inside_isolated_builder"] is True
    info=read(view/"meta/info.json")
    assert info["total_episodes"]==40 and info["total_frames"]==20000 and info["fps"]==50
    assert info["features"]["observation.images.top"]["dtype"]=="image" and info["video_path"] is None
    mapping={int(k):v for k,v in read(view/"episode-map.json")["native_to_original"].items()}
    assert mapping==dict(enumerate(sorted(TRAIN)))
    inventory=read(view/"original-inventory.json");rows=inventory["inputs"]
    assert [(row["episode"],row["frame"]) for row in rows]==[(ep,frame) for ep in TRAIN for frame in range(500)]
    full=[];valid=pad=0
    for row in rows:
        expected=sum(row["frame"]+offset<500 for offset in range(50))
        assert row["valid_target_slots"]==expected and row["padding_slots"]==50-expected
        assert row["positive_supervision"] is True and row["chunk_crosses_unqualified_action"] is False
        valid+=expected;pad+=50-expected
        if expected==50:full.append(row)
    assert inventory["full_chunk_inputs"]==full and len(full)==18040 and valid==951000 and pad==49000
    assert inventory["input_frames"]==20000 and inventory["full_chunk_input_frames"]==18040
    assert inventory["valid_target_slots"]==valid and inventory["padding_slots"]==pad and inventory["full_chunk_valid_target_slots"]==902000
    pixels=read(view/"pixel-identity.json")["frames"]
    assert len(pixels)==20000 and len({(row["original_episode"],row["frame"]) for row in pixels})==20000
    assert {(row["original_episode"],row["frame"]) for row in pixels}=={(ep,frame) for ep in TRAIN for frame in range(500)}
    assert all(mapping[row["native_episode"]]==row["original_episode"] and len(row["rgb_sha256"])==64 for row in pixels)
    for name in ("worker-exited.json","guard-exited.json"):
        exit_record=read(root/name)
        assert exit_record["error_type"] is None and exit_record["shutdown_requested"] is False
        assert exit_record["cleanup"]["status"]=="owned_group_empty"
    assert read(root/"worker-exited.json")["outcomes"]==[{"name":"build","returncode":0},{"name":"read","returncode":0}]
    local_members=0
    for relative,expected in manifest["files"].items():
        path=view/relative
        if path.is_file():assert digest(path)==expected;local_members+=1
    return {"status":"passed","frames_independently_recounted":20000,"full50_inputs":len(full),"valid_slots":valid,
            "padding_slots":pad,"pixel_identity_entries":len(pixels),"received_manifest_members_sha_verified":local_members,
            "view_manifest_sha256":digest(view/"view-manifest.json"),"native_proof_sha256":digest(view/"native-read-verification.json"),
            "model_loaded":False,"scientific_training_success_claimed":False,"shutdown_requested":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("root",type=Path);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();result=verify(args.root)
    with args.output.open("x") as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write("\n")
    print(json.dumps(result,sort_keys=True))
