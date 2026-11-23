"""Train-only native view preparation and explicit source/native episode mapping.

Only a previously independently verified isolation export supplies state/action
rows. Shared source videos/episode metadata are opened in the isolated builder,
never by the resulting normal training reader. No model or optimizer is loaded.
"""
from __future__ import annotations
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import time

TRAIN = (49,4,23,43,21,37,18,34,0,47,38,29,3,26,14,17,44,30,15,42,10,35,25,32,19,36,41,28,8,27,16,11,2,20,9,39,46,48,12,40)
PROTECTED = frozenset((22,13,7,33,45,31,6,1,24,5))
EXPORT_SHA = "c55ae3ec2c93541dfee7ac611e9c20a3577b666df1cd15501a6e0db4d40c6187"
NORMALIZATION_SHA = "263880ec3adfddb8517a50fa5483e7c8f32f0c208243c229cbc72f8e9cf8d988"
REVISION = "cc571a3c661df81b566dbfde3d5c1e85fcdf7884"
CAMERA = "observation.images.top"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def read(path):
    result=json.loads(Path(path).read_text())
    json.dumps(result,allow_nan=False)
    return result


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")


def safe(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or ".." in relative.parts:raise ValueError("Unsafe source member")
    path=(Path(root)/relative).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve()):raise ValueError("Source member escaped root")
    return path


def inventory(lengths):
    if set(lengths)-set(TRAIN):raise ValueError("Inventory contains protected or unknown episode")
    rows=[]
    for ep in TRAIN:
        if ep not in lengths:continue
        if type(lengths[ep]) is not int or lengths[ep]!=500:raise ValueError("Original episode length drift")
        for frame in range(lengths[ep]):
            slots=min(50,lengths[ep]-frame)
            rows.append({"episode":ep,"frame":frame,"valid_target_slots":slots,
                         "padding_slots":50-slots,"positive_supervision":True,
                         "chunk_crosses_unqualified_action":False})
    full=[row for row in rows if row["padding_slots"]==0]
    return {"source_split":"train","hidden_test_loaded":False,"inputs":rows,
            "full_chunk_inputs":full,"input_frames":len(rows),"full_chunk_input_frames":len(full),
            "valid_target_slots":sum(row["valid_target_slots"] for row in rows),
            "padding_slots":sum(row["padding_slots"] for row in rows),
            "full_chunk_valid_target_slots":len(full)*50,"camera_map":{CAMERA:"top"},
            "normalization_source":"D_train_only","inventory_not_execution":True}


def decode_episode(path,start,fps,count,shape):
    import av
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        stream.codec_context.thread_count=2
        container.seek(max(0,int((start-1e-4)/float(stream.time_base))),stream=stream,
                       backward=True,any_frame=False)
        expected=0
        for frame in container.decode(stream):
            if frame.pts is None:raise ValueError("Missing video PTS")
            timestamp=float(frame.pts*frame.time_base)
            index=round((timestamp-start)*fps)
            if index<0:continue
            if index>=count:break
            if index!=expected or abs(timestamp-(start+expected/fps))>1e-4:
                raise ValueError("Video/derived-frame alignment drift")
            pixels=frame.to_ndarray(format="rgb24")
            if list(pixels.shape)!=shape or str(pixels.dtype)!="uint8":raise ValueError("Pixel shape/dtype drift")
            yield expected,pixels
            expected+=1
            if expected==count:break
        if expected!=count:raise ValueError("Incomplete selected video frames")


def build(export_root,source_root,normalization_path,output,episodes,*,max_seconds=600,max_bytes=256*1024**2):
    """Isolated builder; images are lossless PNG bytes inside new Parquet files."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    from PIL import Image
    started=time.monotonic();output=Path(output);source_root=Path(source_root);export_root=Path(export_root)
    if not episodes or set(episodes)-set(TRAIN) or len(set(episodes))!=len(episodes):raise ValueError("Train whitelist only")
    if not 0<max_seconds<=1800 or not 0<max_bytes<=4*1024**3:raise ValueError("Explicit bounded preparation required")
    manifest=read(export_root/"manifest.json");verification=read(export_root/"verification.json")
    if verification["status"]!="passed" or verification["manifest_sha256"]!=sha(export_root/"manifest.json") or manifest["protected_rows_exported"]!=0:
        raise ValueError("Actual independently audited isolation export required")
    exported=next(item for item in manifest["exports"] if item["dataset"]=="lerobot/aloha_sim_insertion_human")
    derived=safe(export_root,exported["path"])
    if exported["sha256"]!=EXPORT_SHA or sha(derived)!=EXPORT_SHA or exported["revision"]!=REVISION:
        raise ValueError("Original D train-only payload identity drift")
    if sha(normalization_path)!=NORMALIZATION_SHA:raise ValueError("Common original D-only statistics identity drift")
    normalization=read(normalization_path)
    if normalization["source_split"]!="train" or normalization["hidden_test_loaded"] is not False or set(normalization["train_episodes"])!=set(TRAIN):
        raise ValueError("Normalization is not original D train40")
    receipt=read(source_root/"rosetta_download_manifest.json")
    if sha(source_root/"rosetta_download_manifest.json")!=exported["source_manifest_sha256"]:
        raise ValueError("Original video/metadata receipt drift")
    members={item["path"]:item for item in receipt["files"]}
    def source_member(relative):
        item=members[relative];path=safe(source_root,relative)
        if path.stat().st_size!=item["bytes"] or sha(path)!=item["sha256"]:raise ValueError("Source metadata/video drift")
        return path
    info=read(source_member("meta/info.json"))
    if info["fps"]!=50 or info["features"][CAMERA]["shape"]!=[480,640,3]:raise ValueError("Original camera/fps drift")
    metadata_paths=sorted(name for name in members if name.startswith("meta/episodes/") and name.endswith(".parquet"))
    # Source metadata/video access remains inside this isolated preparation process.
    episode_meta={}
    for name in metadata_paths:
        for row in pq.read_table(source_member(name)).to_pylist():
            if row["episode_index"] in episodes:episode_meta[row["episode_index"]]=row
    table=pq.read_table(derived)
    grouped={}
    for row in table.to_pylist():
        ep=row["episode_index"]
        if ep not in TRAIN:raise ValueError("Protected row in audited export")
        grouped.setdefault(ep,[]).append(row)
    if set(grouped)!=set(TRAIN) or any(len(rows)!=500 for rows in grouped.values()):raise ValueError("Derived D coverage drift")
    output.mkdir(parents=True,exist_ok=False)
    with (output/"normalization.json").open("xb") as stream:
        stream.write(Path(normalization_path).read_bytes())
    def budget():
        if time.monotonic()-started>=max_seconds or sum(p.stat().st_size for p in output.rglob("*") if p.is_file())>max_bytes:
            raise RuntimeError("Native preparation time/evidence budget exceeded")
    mapping={native:original for native,original in enumerate(sorted(episodes))}
    schema=pa.schema([
        pa.field("episode_index",pa.int64()),pa.field("frame_index",pa.int64()),
        pa.field("timestamp",pa.float32()),pa.field("observation.state",pa.list_(pa.float32(),14)),
        pa.field("action",pa.list_(pa.float32(),14)),pa.field("next.done",pa.bool_()),
        pa.field("index",pa.int64()),pa.field("task_index",pa.int64()),
        pa.field(CAMERA,pa.struct([pa.field("bytes",pa.binary()),pa.field("path",pa.string())]))])
    episode_records=[];pixel_records=[];global_index=0
    for native,original in mapping.items():
        budget();rows=grouped[original];meta=episode_meta[original]
        if [row["frame_index"] for row in rows]!=list(range(500)):raise ValueError("Derived frame order drift")
        video_relative=info["video_path"].format(video_key=CAMERA,
            chunk_index=meta[f"videos/{CAMERA}/chunk_index"],file_index=meta[f"videos/{CAMERA}/file_index"])
        video=source_member(video_relative)
        path=output/f"data/chunk-000/file-{native:03d}.parquet";path.parent.mkdir(parents=True,exist_ok=True)
        buffered=[]
        with pq.ParquetWriter(path,schema,compression="zstd") as writer:
            for frame,pixels in decode_episode(video,meta[f"videos/{CAMERA}/from_timestamp"],50,500,[480,640,3]):
                budget();row=rows[frame]
                if abs(row["timestamp"]-frame/50)>1e-4:raise ValueError("Derived timestamp drift")
                for key in ("action","observation.state"):
                    if len(row[key])!=14 or not all(math.isfinite(x) for x in row[key]):raise ValueError("Nonfinite derived vector")
                buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format="PNG",compress_level=3)
                payload=buffer.getvalue()
                decoded=np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
                if not np.array_equal(decoded,pixels):raise ValueError("Lossless image roundtrip failed")
                copied={key:row[key] for key in ("frame_index","timestamp","observation.state","action","next.done")}
                copied.update(episode_index=native,index=global_index+frame,task_index=0,
                              **{CAMERA:{"bytes":payload,"path":None}})
                buffered.append(copied)
                pixel_records.append({"native_episode":native,"original_episode":original,"frame":frame,
                                      "rgb_sha256":hashlib.sha256(pixels.tobytes()).hexdigest()})
                if len(buffered)==16:writer.write_table(pa.Table.from_pylist(buffered,schema=schema));buffered=[]
            if buffered:writer.write_table(pa.Table.from_pylist(buffered,schema=schema))
        episode_records.append({"episode_index":native,"tasks":meta["tasks"],"length":500,
            "dataset_from_index":global_index,"dataset_to_index":global_index+500,
            "data/chunk_index":0,"data/file_index":native,"meta/episodes/chunk_index":0,"meta/episodes/file_index":0})
        global_index+=500;budget()
    new_info=copy.deepcopy(info)
    new_info.update(total_episodes=len(mapping),total_frames=global_index,splits={"train":f"0:{len(mapping)}"},video_path=None)
    new_info["features"][CAMERA]["dtype"]="image"
    new_info["features"][CAMERA].pop("video_info",None)
    save(output/"meta/info.json",new_info)
    stats=copy.deepcopy(normalization["effective_stats"])
    stats[CAMERA]=normalization["visual_statistics"]
    save(output/"meta/stats.json",stats)
    tasks=pq.read_table(source_member("meta/tasks.parquet"))
    pq.write_table(tasks,output/"meta/tasks.parquet")
    meta_path=output/"meta/episodes/chunk-000/file-000.parquet";meta_path.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episode_records),meta_path)
    save(output/"episode-map.json",{"native_to_original":{str(k):v for k,v in mapping.items()},
                                   "original_order":list(TRAIN),"normalization_sha256":NORMALIZATION_SHA})
    selected_inventory=inventory({ep:500 for ep in episodes})
    selected_inventory.update(derived_payload_sha256=EXPORT_SHA,normalization_sha256=NORMALIZATION_SHA)
    save(output/"original-inventory.json",selected_inventory)
    save(output/"full-D-inventory.json",inventory({ep:500 for ep in TRAIN}))
    save(output/"pixel-identity.json",{"frames":pixel_records,"lossless_png_roundtrip":True})
    save(output/"label-identity.json",{"episodes":{str(ep):[
        {"frame":row["frame_index"],"action":row["action"],"state":row["observation.state"]}
        for row in grouped[ep]] for ep in sorted(episodes)},"source_derived_sha256":EXPORT_SHA})
    budget()
    files={p.relative_to(output).as_posix():sha(p) for p in output.rglob("*") if p.is_file()}
    save(output/"view-manifest.json",{"schema":"rosetta.D_train_native_view.v1","status":"prepared_unverified_native_read",
        "files":files,"train_original_episodes":sorted(episodes),"all_40_episodes_present":set(episodes)==set(TRAIN),
        "isolation_manifest_sha256":sha(export_root/"manifest.json"),"derived_payload_sha256":EXPORT_SHA,
        "normalization_sha256":NORMALIZATION_SHA,"protected_rows_exported":0,
        "shared_video_preroll_may_be_decoded_inside_isolated_builder":True,
        "normal_training_source_mixed_parquet_or_video_required":False,"elapsed_seconds":time.monotonic()-started,
        "payload_bytes":sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),"model_loaded":False,
        "optimizer_updates":0,"shutdown_requested":False})
    budget()
    return output


class EpisodeColumnProxy:
    def __init__(self,native,mapping):
        self.native,self.mapping=native,mapping
        if hasattr(native,"select_columns"):
            fields=[key for key in ("episode_index","frame_index","index","timestamp","next.done","task_index")
                    if key in native.column_names]
            # HF Column iteration retains all columns for a custom formatting
            # transform. Remove image/action columns AND that transform for
            # metadata lookups, so building the scientific ID index does not
            # decode every PNG. Native __getitem__ remains untouched.
            self.metadata=native.select_columns(fields).with_format(None)
        else:self.metadata=native
    def __getattr__(self,name):return getattr(self.native,name)
    def __getitem__(self,key):
        value=(self.metadata if isinstance(key,str) and key in ("episode_index","frame_index","index","timestamp","next.done","task_index") else self.native)[key]
        if key=="episode_index":return [self.mapping[int(ep)] for ep in value]
        return value


def logical_native_dataset(root,original_episodes,*,delta_timestamps=None,camera_uint8=False,action_tensor_cache=True):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root=Path(root);manifest=read(root/"view-manifest.json")
    for relative,expected in manifest["files"].items():
        if sha(safe(root,relative))!=expected:raise ValueError("Native view payload changed")
    mapping={int(k):v for k,v in read(root/"episode-map.json")["native_to_original"].items()}
    inverse={original:native for native,original in mapping.items()}
    if set(original_episodes)-set(inverse) or set(original_episodes)&PROTECTED:raise ValueError("Native view episode mapping unavailable/protected")
    native=LeRobotDataset(repo_id="lerobot/aloha_sim_insertion_human",root=root,
                         episodes=[inverse[ep] for ep in original_episodes],revision=REVISION,
                         delta_timestamps=delta_timestamps,download_videos=False,video_backend="pyav")
    if action_tensor_cache and delta_timestamps is not None:
        import pyarrow.parquet as pq
        # Bound to this dataset instance and its verified self-contained files.
        # The native reader still computes absolute chunk indices and padding,
        # reads every current image/state and applies normal image transforms.
        # Only repeated HF action-column conversion is replaced by exact tensor
        # indexing; no original mixed source is opened or global API modified.
        records=[]
        for relative in sorted(manifest["files"]):
            if relative.startswith("data/") and relative.endswith(".parquet"):
                records.extend(pq.read_table(root/relative,columns=["index","action"]).to_pylist())
        records.sort(key=lambda row:row["index"])
        if [row["index"] for row in records]!=list(range(len(records))):raise ValueError("Native action absolute index drift")
        action_column=torch.tensor([row["action"] for row in records],dtype=torch.float32)
        if action_column.shape!=(500*len(mapping),14) or not torch.isfinite(action_column).all():
            raise ValueError("Train-only native action cache coverage drift")
        reader=native._ensure_reader()
        original_query=reader._query_hf_dataset
        def cached_query(query_indices):
            others={key:indices for key,indices in query_indices.items() if key!="action"}
            result=original_query(others) if others else {}
            if "action" in query_indices:result["action"]=action_column[query_indices["action"]]
            return result
        reader._query_hf_dataset=cached_query
    class LogicalDataset:
        def __len__(self):return len(native)
        def __getattr__(self,name):return getattr(native,name)
        @property
        def hf_dataset(self):return EpisodeColumnProxy(native.hf_dataset,mapping)
        def __getitem__(self,index):
            row=dict(native[index])
            row["episode_index"]=torch.tensor(mapping[int(row["episode_index"])],dtype=torch.int64)
            if camera_uint8:
                image=row[CAMERA]
                if image.dtype!=torch.float32 or not torch.isfinite(image).all():raise ValueError("Native PNG conversion dtype drift")
                recovered=(image*255).round().to(torch.uint8)
                if not torch.equal(recovered.to(torch.float32)/255,image):raise ValueError("Native RGB bytes not exactly recoverable")
                row[CAMERA]=recovered
            return row
    return LogicalDataset()


def accepted_training_view(root,expected_manifest_sha256,*,native_read_sha256=None):
    """One admission predicate shared by both actual native entry points."""
    root=Path(root)
    manifest=read(root/"view-manifest.json");proof=read(root/"native-read-verification.json")
    if (sha(root/"view-manifest.json")!=expected_manifest_sha256
            or (native_read_sha256 is not None and sha(root/"native-read-verification.json")!=native_read_sha256)
            or manifest["all_40_episodes_present"] is not True
            or set(manifest["train_original_episodes"])!=set(TRAIN)
            or manifest["normalization_sha256"]!=NORMALIZATION_SHA
            or proof["status"]!="passed" or proof["native_read_frames"]!=20000
            or set(proof["original_episodes"])!=set(TRAIN)
            or proof["view_manifest_sha256"]!=expected_manifest_sha256
            or proof["normalization_sha256"]!=NORMALIZATION_SHA
            or proof.get("label_parity")!="all_selected_states_and_chunk_actions_exact"
            or proof.get("pixel_parity")!="all_selected_frames_exact"
            or sha(root/"normalization.json")!=NORMALIZATION_SHA):
        raise ValueError("Full original D40 native acceptance/identity required")
    return manifest,proof


def load_native_training_view(root,expected_manifest_sha256,experiment,*,episodes=None):
    """Native dataset with source IDs, float32 CHW RGB and original D statistics.

    This is the dedicated launcher's direct dataset entry. It accepts only a
    fully materialized, independently read and hash-bound original train40 view.
    Metadata describes the actual forty native episodes; source IDs appear in
    hf_dataset episode columns and returned records, never as extra batch keys.
    """
    root=Path(root)
    accepted_training_view(root,expected_manifest_sha256)
    declared=list(experiment["dataset"]["train_episodes"])
    selected=declared if episodes is None else list(episodes)
    if declared!=list(TRAIN) or not selected or set(selected)-set(declared):
        raise ValueError("Original D train episode declaration drift")
    return logical_native_dataset(root,selected,delta_timestamps={"action":[i/50 for i in range(50)]})


def install_native_D_factory(module,view_root,*,view_manifest_sha256,native_read_sha256,
                             delta_timestamps,camera_uint8=False):
    """Explicit opt-in factory hook; original row reader is never called.

    The execution owner composes this with its feature stack. The supplied
    module must be the actual upstream factory binding called by its trainer.
    Returning the restorer makes exact wrapper ownership reviewable.
    """
    root=Path(view_root).resolve()
    accepted_training_view(root,view_manifest_sha256,native_read_sha256=native_read_sha256)
    original=module.make_train_eval_datasets
    if getattr(original,"_rosetta_D_native_view",False):raise RuntimeError("D native factory already installed")
    def factory(cfg):
        declared=Path(cfg.dataset.root).resolve()
        episodes=list(cfg.dataset.episodes or [])
        if declared!=root or not episodes or set(episodes)-set(TRAIN) or cfg.dataset.repo_id!="lerobot/aloha_sim_insertion_human":
            raise ValueError("Trainer D root/source/episode boundary drift")
        if getattr(cfg.dataset,"eval_split",0.)!=0.:raise ValueError("No protected eval split may be carved from D")
        return logical_native_dataset(root,episodes,delta_timestamps=delta_timestamps,
                                      camera_uint8=camera_uint8),None
    factory._rosetta_D_native_view=True
    module.make_train_eval_datasets=factory
    def restore():
        if module.make_train_eval_datasets is not factory:raise RuntimeError("Native D factory ownership changed")
        module.make_train_eval_datasets=original
    return restore


def verify_native_read(root,*,proof_output=None,action_tensor_cache=True):
    import torch
    torch.set_num_threads(2)
    root=Path(root);manifest=read(root/"view-manifest.json");episodes=manifest["train_original_episodes"]
    dataset=logical_native_dataset(root,episodes,delta_timestamps={"action":[i/50 for i in range(50)]},camera_uint8=True,
                                   action_tensor_cache=action_tensor_cache)
    mapping=read(root/"episode-map.json")["native_to_original"]
    pixels={(row["original_episode"],row["frame"]):row["rgb_sha256"] for row in read(root/"pixel-identity.json")["frames"]}
    labels=read(root/"label-identity.json")
    if labels["source_derived_sha256"]!=EXPORT_SHA:raise ValueError("Label audit source drift")
    targets={int(ep):torch.tensor([row["action"] for row in rows],dtype=torch.float32)
             for ep,rows in labels["episodes"].items()}
    states={int(ep):torch.tensor([row["state"] for row in rows],dtype=torch.float32)
            for ep,rows in labels["episodes"].items()}
    checked=0
    for index in range(len(dataset)):
        row=dataset[index];ep,frame=int(row["episode_index"]),int(row["frame_index"])
        if ep not in episodes or tuple(row["action"].shape)!=(50,14):raise ValueError("Native source/shape drift")
        if not torch.isfinite(row["action"]).all() or int((~row["action_is_pad"]).sum())!=min(50,500-frame):raise ValueError("Native padding/action drift")
        expected=targets[ep][torch.arange(frame,frame+50).clamp(max=499)]
        if not torch.equal(row["action"],expected) or not torch.equal(row["observation.state"],states[ep][frame]):
            raise ValueError("Native original state/action chunk values drift")
        rgb=row[CAMERA].permute(1,2,0).contiguous().numpy().tobytes()
        if hashlib.sha256(rgb).hexdigest()!=pixels[(ep,frame)]:raise ValueError("Native RGB/source pixel identity mismatch")
        checked+=1
    if checked!=500*len(episodes):raise ValueError("Native row coverage drift")
    result={"status":"passed","native_read_frames":checked,"original_episodes":episodes,
        "view_manifest_sha256":sha(root/"view-manifest.json"),"native_episode_count":len(mapping),
        "normalization_sha256":NORMALIZATION_SHA,"source_mixed_payload_opened":False,
        "label_parity":"all_selected_states_and_chunk_actions_exact",
        "pixel_parity":"all_selected_frames_exact","model_loaded":False,"optimizer_updates":0,"shutdown_requested":False}
    result["native_action_query"]="per_dataset_verified_train_only_tensor_cache" if action_tensor_cache else "unmodified_upstream_hf_column"
    save(Path(proof_output) if proof_output is not None else root/"native-read-verification.json",result)
    return result
