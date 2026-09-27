#!/usr/bin/env bash
set -euo pipefail
set -C
sshbin=/mnt/c/Windows/System32/OpenSSH/ssh.exe
remote_root=/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001
receipt_root=reports/training/targeted-dataset-staging-20260927-001
archive=.work/targeted-dataset-metadata-20260927-001.tar
test ! -e "$receipt_root"
test ! -e "$archive"
cat .work/dataset_closeout_20260927.py | "$sshbin" -o BatchMode=yes -p 20497 root@connect.cqa1.seetacloud.com "(set -C; cat > $remote_root/closeout.py)"
"$sshbin" -o BatchMode=yes -p 20497 root@connect.cqa1.seetacloud.com "python $remote_root/closeout.py"
"$sshbin" -o BatchMode=yes -p 20497 root@connect.cqa1.seetacloud.com "cd $remote_root && tar -cf - plan.json stream-execution.json rj45-subset-metadata-audit.json existing-verification.jsonl independent-verification-stream.json result-stream.json exit-code-stream.txt closeout-observation.json provenance.sha256 lerobot/aloha_sim_insertion_human/cc571a3c661df81b566dbfde3d5c1e85fcdf7884/rosetta_download_manifest.json amandlek/mimicgen_datasets/33016f8a62c02334f929f2913af8fdd2a8a129e1/rosetta_download_manifest.json open_x_embodiment/austin_sirius_dataset_converted_externally_to_rlds/rosetta_download_manifest.json REBOOT26/USB-A_recovery_install/cfa3498a3982eb24554e88e77140247871eee3eb/rosetta_download_manifest.json REBOOT26/rj45_recovery_install/f6acd5b69394ffd4d2d8bd30079306c9e2061bbe/rosetta_download_manifest.json" > "$archive"
while IFS= read -r entry; do
    case "$entry" in /*|../*|*/../*|*'/..') printf 'unsafe receipt archive path\n' >&2; exit 1;; esac
done < <(tar -tf "$archive")
tar -tvf "$archive" | awk 'substr($0,1,1)!="-" {bad=1} END {exit bad}'
mkdir "$receipt_root"
tar --no-same-owner --no-same-permissions -xf "$archive" -C "$receipt_root"
printf 'metadata_receipts_retrieved=%s\n' "$receipt_root"
