#!/usr/bin/env bash
# Create-only full checkpoint backup. No model imports, remote writes or deletion.
set -Eeuo pipefail
root=${1:?Exact remote checkpoint directory required}
destination=${2:?Exact new local backup directory required}
[[ "$root" =~ ^/root/autodl-tmp/rosetta/runs/[A-Za-z0-9_./-]+$ ]]
[[ "$root" != */../* && "$root" != */.. && "$root" != */./* ]]
[[ "$destination" =~ ^/mnt/d/rosetta_reality/[A-Za-z0-9_./-]+$ ]]
[[ "$destination" != */../* && "$destination" != */.. && "$destination" != */./* ]]
test ! -e "$destination"
resolved_destination=$(realpath -m -- "$destination")
[[ "$resolved_destination" == /mnt/d/rosetta_reality/* ]]
sshbin=/mnt/c/Windows/System32/OpenSSH/ssh.exe
remote() {
  "$sshbin" -p 20497 -o BatchMode=yes -o ConnectTimeout=12 \
    root@connect.cqa1.seetacloud.com "$@"
}
mkdir -p "$(dirname -- "$destination")"
mkdir "$destination"
inventory() {
  remote "/root/autodl-tmp/rosetta/envs/smolvla-cuda-001/bin/python - '$root'" <<'PY'
import hashlib,json,pathlib,re,sys
root=pathlib.Path(sys.argv[1])
if not root.is_dir() or root.is_symlink() or root.resolve()!=root:raise ValueError('Real checkpoint directory required')
files=[]
for path in sorted(root.rglob('*')):
    relative=path.relative_to(root).as_posix()
    if not re.fullmatch(r'[A-Za-z0-9_./-]+',relative):raise ValueError('Unsafe checkpoint member')
    if path.is_symlink():raise ValueError('Symlink checkpoint is not self-contained')
    if path.is_dir():continue
    if not path.is_file() or path.stat().st_nlink!=1:raise ValueError('Special/hardlinked member requires explicit materialization')
    before=path.stat();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('Checkpoint mutated during inventory')
    files.append({'path':relative,'bytes':before.st_size,'sha256':digest.hexdigest()})
if not files:raise ValueError('Empty checkpoint')
if len({row['path'].casefold() for row in files})!=len(files):raise ValueError('Case-colliding checkpoint paths on Windows backup volume')
names={row['path'] for row in files}
if not {'pretrained_model/model.safetensors','pretrained_model/config.json'}<=names or not any(name.startswith('training_state/') for name in names):
    raise ValueError('Full native checkpoint root with policy/config and training state required; export-only is insufficient')
print(json.dumps({'schema':'rosetta.full_checkpoint_inventory.v1','files':files,'bytes':sum(x['bytes'] for x in files)},sort_keys=True))
PY
}
inventory > "$destination/remote-before.json"
# This is filesystem evidence processing only; it does not load ML/data packages.
python3 - "$destination" <<'PY'
import json,pathlib,shutil,sys
destination=pathlib.Path(sys.argv[1]);inventory=json.loads((destination/'remote-before.json').read_text())
if shutil.disk_usage(destination).free<2*inventory['bytes']+1024**3:raise OSError('Reserve for archive and full extracted checkpoint required')
with (destination/'payload.sha256').open('x') as stream:
    for row in inventory['files']:stream.write(row['sha256']+'  '+row['path']+'\n')
PY
timeout --signal=TERM --kill-after=10 1800 \
  "$sshbin" -p 20497 -o BatchMode=yes -o ConnectTimeout=12 \
  root@connect.cqa1.seetacloud.com "tar --format=posix -C '$root' -cf - ." > "$destination/checkpoint.tar.partial"
mv -- "$destination/checkpoint.tar.partial" "$destination/checkpoint.tar"
mkdir "$destination/verified"
# Check every archive member before extraction; preserve failures for review.
tar -tf "$destination/checkpoint.tar" > "$destination/archive-members.txt"
python3 - "$destination" <<'PY'
import pathlib,sys,tarfile
destination=pathlib.Path(sys.argv[1])
with tarfile.open(destination/'checkpoint.tar') as archive:
    for member in archive:
        path=pathlib.PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):raise ValueError('Unsafe archive member/type')
PY
tar --no-same-owner --no-same-permissions -xf "$destination/checkpoint.tar" -C "$destination/verified"
(cd "$destination/verified" && sha256sum -c ../payload.sha256) > "$destination/verified.log"
inventory > "$destination/remote-after.json"
cmp "$destination/remote-before.json" "$destination/remote-after.json"
python3 - "$destination" <<'PY'
import hashlib,json,pathlib,sys
destination=pathlib.Path(sys.argv[1]);inventory=json.loads((destination/'remote-before.json').read_text())
observed=sorted(path.relative_to(destination/'verified').as_posix() for path in (destination/'verified').rglob('*') if path.is_file())
if observed!=sorted(row['path'] for row in inventory['files']):raise ValueError('Full checkpoint coverage drift')
with (destination/'backup-verification.json').open('x') as stream:
    json.dump({'schema':'rosetta.full_checkpoint_backup.v1','status':'passed','files':len(observed),'bytes':inventory['bytes'],
      'before_after_remote_inventory_sha256':hashlib.sha256((destination/'remote-before.json').read_bytes()).hexdigest(),
      'all_file_sha256_verified':True,'remote_unchanged_during_transfer':True,'model_loaded':False,
      'remote_deleted':False,'shutdown_requested':False},stream,sort_keys=True,indent=2)
PY
cat "$destination/backup-verification.json"
