#!/usr/bin/env bash
set -u
for url in \
  'https://hf-mirror.com/datasets/lerobot/aloha_sim_insertion_scripted/resolve/8ab660912970111cbb26738b11458e6fc4a4aed1/README.md' \
  'https://hf-mirror.com/datasets/lerobot/aloha_sim_insertion_scripted/resolve/8ab660912970111cbb26738b11458e6fc4a4aed1/videos/observation.images.top/chunk-000/file-000.mp4' \
  'https://hf-mirror.com/datasets/robomimic/robomimic_datasets/resolve/74fa018461f479cd9fd15b924a16103012096203/v1.5/can/paired/low_dim_v15.hdf5'; do
  printf '%s\n' "$url"
  curl -4 -L -I -sS -o /dev/null -w '%{http_code} %{remote_ip} %{size_download}\n' --max-time 30 "$url" || true
done
