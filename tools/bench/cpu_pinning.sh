#!/usr/bin/env bash
set -euo pipefail

if command -v lscpu >/dev/null 2>&1; then
  mask=$(lscpu -p=CPU,CORE,SOCKET,ONLINE \
    | awk -F, 'BEGIN { OFS="," } !/^#/ && $4=="Y" && $3==0 { if (!seen[$2]++) print $1 }' \
    | sort -n | paste -sd, -)
else
  mask=$(for f in /sys/devices/system/cpu/cpu*/topology/thread_siblings_list; do
    [[ -r $f ]] || continue
    entry=$(<"$f")
    entry=${entry%%,*}
    entry=${entry%%-*}
    printf '%s\n' "$entry"
  done | sort -n | uniq | paste -sd, -)
fi

if [[ -z ${mask:-} ]]; then
  echo "[cpu_pinning] failed to determine physical core mask" >&2
  exit 1
fi

printf '%s\n' "$mask"
