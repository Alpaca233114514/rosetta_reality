#!/usr/bin/env bash
set -u
for url in https://huggingface.co https://hf-mirror.com https://storage.googleapis.com https://www.google.com; do
  printf '%s ' "$url"
  curl -4 -L -I -sS -o /dev/null -w '%{http_code} %{remote_ip}\n' --max-time 12 "$url" || true
done
printf 'proxy_variables_set='
env | cut -d= -f1 | grep -Ei 'proxy|endpoint' | tr '\n' ','
printf '\n'
df -h /root/autodl-tmp
