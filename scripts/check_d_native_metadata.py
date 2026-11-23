"""CPU metadata lookup and filtered native identity parity without model loads."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import resource
import sys
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))


def main():
    import torch
    from rosetta_reality.diagnostics.d_train_native_view import TRAIN,CAMERA,load_native_training_view
    from PIL import Image
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--manifest-sha256",required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();started=time.monotonic();experiment={"dataset":{"train_episodes":list(TRAIN)}}
    dataset=load_native_training_view(args.root,args.manifest_sha256,experiment)
    def columns(dataset):
        begin=time.monotonic()
        with patch.object(Image.Image,"load",side_effect=AssertionError("Metadata lookup decoded image")):
            episodes=list(dataset.hf_dataset["episode_index"]);frames=list(dataset.hf_dataset["frame_index"])
        return list(zip(episodes,frames)),time.monotonic()-begin
    pairs,elapsed=columns(dataset)
    assert pairs==[(ep,frame) for ep in sorted(TRAIN) for frame in range(500)]
    assert len(dataset)==20000 and dataset.meta.total_episodes==40 and dataset.meta.fps==50
    del dataset;gc.collect()
    subset=load_native_training_view(args.root,args.manifest_sha256,experiment,episodes=[49,4])
    subset_pairs,subset_elapsed=columns(subset)
    assert subset_pairs==[(ep,frame) for ep in (4,49) for frame in range(500)]
    labels=json.loads((args.root/"label-identity.json").read_text())["episodes"]
    pixels={(row["original_episode"],row["frame"]):row["rgb_sha256"] for row in json.loads((args.root/"pixel-identity.json").read_text())["frames"]}
    for index in (0,450,499,500,950,999):
        row=subset[index];ep,frame=subset_pairs[index]
        assert int(row["episode_index"])==ep and int(row["frame_index"])==frame
        target=torch.tensor([labels[str(ep)][min(frame+offset,499)]["action"] for offset in range(50)],dtype=torch.float32)
        assert torch.equal(row["action"],target)
        assert torch.equal(row["observation.state"],torch.tensor(labels[str(ep)][frame]["state"],dtype=torch.float32))
        assert int((~row["action_is_pad"]).sum())==min(50,500-frame)
        rgb=(row[CAMERA]*255).round().to(torch.uint8).permute(1,2,0).contiguous().numpy().tobytes()
        assert hashlib.sha256(rgb).hexdigest()==pixels[(ep,frame)]
    result={"status":"passed","full_metadata_rows":20000,"full_metadata_lookup_seconds":elapsed,
            "filtered_original_episodes":[49,4],"filtered_metadata_rows":1000,"filtered_metadata_lookup_seconds":subset_elapsed,
            "filtered_boundary_native_reads":6,"metadata_image_load_forbidden_and_not_called":True,
            "elapsed_seconds":time.monotonic()-started,"maximum_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "view_manifest_sha256":args.manifest_sha256,"model_loaded":False,"optimizer_updates":0,"shutdown_requested":False}
    with args.output.open("x") as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write("\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":main()
