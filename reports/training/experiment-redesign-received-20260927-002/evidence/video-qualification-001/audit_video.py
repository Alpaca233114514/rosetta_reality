import hashlib,json,time,math
from pathlib import Path
import pyarrow.dataset as ds
import av
root=Path('/root/autodl-tmp/rosetta')
job=root/'runs/aloha-train-isolation-20260927-004'
out=job/'video-qualification-001'
plan=json.loads((root/'workspaces/20260927T084016Z-66a5bbcdbdcd-a188785d7f32/configs/diagnostics/aloha_train_isolation_20260927_004.json').read_text())
allowed=plan['train_episodes'];started=time.monotonic();reports=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
for src in plan['sources']:
 p=root/src['root'];receipt=json.loads((p/'rosetta_download_manifest.json').read_text())
 cols=['episode_index','length','videos/observation.images.top/chunk_index','videos/observation.images.top/file_index','videos/observation.images.top/from_timestamp','videos/observation.images.top/to_timestamp']
 mp=p/'meta/episodes/chunk-000/file-000.parquet';mi=next(x for x in receipt['files'] if x['path']=='meta/episodes/chunk-000/file-000.parquet');assert sha(mp)==mi['sha256']
 rows=ds.dataset(mp,format='parquet').to_table(columns=cols,filter=ds.field('episode_index').isin(allowed)).to_pylist()
 assert {r['episode_index'] for r in rows}==set(allowed)
 assert all(r['length']==src['expected_episode_frames'] for r in rows)
 assert all(r['videos/observation.images.top/chunk_index']==0 and r['videos/observation.images.top/file_index']==0 for r in rows)
 vp=p/'videos/observation.images.top/chunk-000/file-000.mp4';vi=next(x for x in receipt['files'] if x['path']=='videos/observation.images.top/chunk-000/file-000.mp4');assert sha(vp)==vi['sha256']
 expected={round((r['videos/observation.images.top/from_timestamp']+f/50)*50) for r in rows for f in range(r['length'])}
 observed=set();duplicates=0
 with av.open(str(vp)) as c:
  st=c.streams.video[0]
  assert (st.width,st.height)==(640,480)
  for packet in c.demux(st):
   if packet.pts is None:continue
   tick=round(float(packet.pts*st.time_base)*50)
   if tick in expected:
    duplicates+=tick in observed;observed.add(tick)
 assert observed==expected and not duplicates
 samples=[]
 with av.open(str(vp)) as c:
  st=c.streams.video[0];st.thread_type='AUTO';st.codec_context.thread_count=2
  for ep in allowed[:2]:
   r=next(r for r in rows if r['episode_index']==ep)
   for f in (0,r['length']//2,r['length']-1):
    target=r['videos/observation.images.top/from_timestamp']+f/50
    c.seek(int(target/st.time_base),stream=st,backward=True,any_frame=False)
    matched=False
    for image in c.decode(st):
     if time.monotonic()-started>50:raise TimeoutError('bounded decode')
     ts=float(image.pts*image.time_base)
     if ts<target-1e-4:continue
     assert abs(ts-target)<1e-4 and (image.width,image.height)==(640,480)
     matched=True;break
    assert matched
    samples.append({'episode':ep,'frame':f,'timestamp_matched':True,'dimensions_matched':True})
 reports.append({'dataset':src['dataset'],'revision':src['revision'],'metadata_sha256':sha(mp),'video_sha256':sha(vp),'train_packet_timestamps_verified':len(observed),'pixel_decode_samples':samples,'raw_payload_exported':False,'explicit_success_or_recovery_columns':False})
with (out/'result.json').open('x') as f:json.dump({'status':'passed','sources':reports,'elapsed_seconds':time.monotonic()-started,'scope':'all train packet timestamps; two train episodes per source sampled at start/middle/end','full_video_decode_verified':False,'success_recovery_labels_verified':False,'training_ready':False,'protected_frames_may_be_decoded_during_keyframe_seek':True,'protected_frames_exported':0},f,indent=2)
