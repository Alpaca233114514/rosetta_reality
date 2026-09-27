#!/usr/bin/env bash
set -u
if [ ! -r /etc/network_turbo ]; then
  printf 'network_turbo_missing\n'
  exit 0
fi
source /etc/network_turbo
for url in \
  'https://huggingface.co/datasets/lerobot/aloha_sim_insertion_scripted/resolve/8ab660912970111cbb26738b11458e6fc4a4aed1/README.md' \
  'https://huggingface.co/datasets/REBOOT26/USB-A_recovery_install/resolve/cfa3498a3982eb24554e88e77140247871eee3eb/meta/info.json'; do
  printf '%s\n' "$url"
  curl -4 -L -I -sS -o /dev/null -w '%{http_code} %{remote_ip}\n' --max-time 25 "$url" || true
done
