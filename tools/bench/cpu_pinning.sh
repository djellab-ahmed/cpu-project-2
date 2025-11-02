#!/usr/bin/env bash
set -euo pipefail
# Print one logical CPU per physical core (first sibling) as "0,2,4,..."
mapfile -t primaries < <(for f in /sys/devices/system/cpu/cpu*/topology/thread_siblings_list; do
  cut -d',' -f1 "$f"
done | sort -n | uniq)
IFS=, ; echo "${primaries[*]}"
