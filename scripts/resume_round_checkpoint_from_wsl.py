import hashlib,json,pathlib,subprocess,sys,tarfile,time
source=sys.argv[1]; prior=pathlib.Path(sys.argv[2]); dest=pathlib.Path(sys.argv[3])
assert source.startswith('/root/autodl-tmp/rosetta/runs/') and '..' not in source
assert str(dest).startswith('/mnt/d/rosetta_reality/') and not dest.exists()
dest.mkdir(parents=True); out=dest/'verified';out.mkdir()
inv=json.loads((prior/'remote-before.json').read_text()); (dest/'remote-before.json').write_bytes((prior/'remote-before.json').read_bytes())
ssh=['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-p','20497','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4','root@connect.cqa1.seetacloud.com']
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
# Recover regular member prefixes from the preserved failed tar; no extraction of links.
try:
 with tarfile.open(prior/'checkpoint.tar.partial',mode='r|') as ar:
  for m in ar:
   if m.isdir():continue
   rel=pathlib.PurePosixPath(m.name)
   assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts
   p=out/rel;p.parent.mkdir(parents=True,exist_ok=True)
   f=ar.extractfile(m)
   with p.open('xb') as target:
    while True:
     block=f.read(1024*1024)
     if not block:break
     target.write(block)
except tarfile.ReadError:
 pass
proof=[]
for row in inv['files']:
 p=out/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(3):
  n=p.stat().st_size if p.exists() else 0
  assert n<=row['bytes']
  remote_path=source+'/'+row['path']
  if n:
   code="import hashlib;h=hashlib.sha256();f=open("+repr(remote_path)+",'rb');remaining="+str(n)+"\nwhile remaining:\n b=f.read(min(1048576,remaining));assert b;h.update(b);remaining-=len(b)\nprint(h.hexdigest())"
   remotehash=subprocess.check_output(ssh+["/root/autodl-tmp/rosetta/envs/smolvla-cuda-001/bin/python -"],input=code.encode(),timeout=45).decode().strip()
   assert digest(p)==remotehash, 'Partial source prefix mismatch'
  if n==row['bytes'] and digest(p)==row['sha256']:break
  with p.open('ab') as target:
   try:
    proc=subprocess.run(ssh+[f"dd if='{remote_path}' iflag=skip_bytes skip={n} bs=1M status=none"],stdout=target,timeout=1200)
   except subprocess.TimeoutExpired:
    proc=subprocess.CompletedProcess([],124)
  print(json.dumps({'path':row['path'],'attempt':attempt,'received':p.stat().st_size,'returncode':proc.returncode}),flush=True)
  if proc.returncode==0 and p.stat().st_size==row['bytes'] and digest(p)==row['sha256']:break
 else:raise RuntimeError('Bounded per-file recovery exhausted: '+row['path'])
 assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
 proof.append(row)
code="import pathlib,hashlib,json\nr=pathlib.Path("+repr(source)+")\nrows=[]\nfor p in sorted(r.rglob('*')):\n if p.is_dir():continue\n assert p.is_file() and not p.is_symlink() and p.stat().st_nlink==1\n h=hashlib.sha256()\n with p.open('rb') as f:\n  for b in iter(lambda:f.read(1048576),b''):h.update(b)\n rows.append({'path':p.relative_to(r).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})\nprint(json.dumps({'files':rows,'bytes':sum(x['bytes'] for x in rows)}))"
after=json.loads(subprocess.check_output(ssh+["/root/autodl-tmp/rosetta/envs/smolvla-cuda-001/bin/python -"],input=code.encode(),timeout=120))
assert after['files']==inv['files'] and after['bytes']==inv['bytes']
assert sorted(p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file())==sorted(x['path'] for x in inv['files'])
(dest/'remote-after.json').write_text(json.dumps(after,sort_keys=True,indent=2))
result={'schema':'rosetta.full_checkpoint_backup.v1','status':'passed','files':len(proof),'bytes':inv['bytes'],'all_file_sha256_verified':True,'remote_unchanged_during_transfer':True,'model_loaded':False,'remote_deleted':False,'shutdown_requested':False,'recovery':'preserved_failed_tar_prefix_remote_sha_verified_then_byte_resume'}
(dest/'backup-verification.json').write_text(json.dumps(result,sort_keys=True,indent=2));print(json.dumps(result),flush=True)
