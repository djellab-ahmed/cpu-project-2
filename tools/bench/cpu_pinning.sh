#!/usr/bin/env bash
set -euo pipefail
# Print a comma-separated list of the first (primary) logical CPU for each
# physical core on socket 0. Robust to "0-1" or "0,1" formats. Online CPUs only.

if command -v lscpu >/dev/null 2>&1; then
  # lscpu -p=CPU,CORE,SOCKET,ONLINE ; ignore '#' comments
  mapfile -t primaries < <(
    lscpu -p=CPU,CORE,SOCKET,ONLINE \
    | awk -F, '!/^#/ && $4 == "Y" && $3 == 0 { if (!seen[$2]++) print $1 }' \
    | sort -n
  )
else
  # Fallback: read sysfs topology and take the first id from "A,B" or "A-B"
  mapfile -t primaries < <(
    for f in /sys/devices/system/cpu/cpu*/topology/thread_siblings_list; do
      s=$(<"$f")
      # First entry before comma or dash
      echo "$s" | sed 's/,.*//' | sed 's/-.*//'
    done | sort -n | uniq
  )
fi

if ((${#primaries[@]}==0)); then
  echo "No online CPUs detected on socket 0" >&2
  exit 1
fi

printf '%s\n' "$(IFS=,; echo "${primaries[*]}")"
