#!/bin/bash
# Safe R2 offload: upload -> checksum check -> delete local.
# Exits non-zero if any step fails; local data is never deleted unless checksum matches.
# Usage: r2_offload_step.sh <local_path> <r2_prefix>
set -euo pipefail
local_path="$1"
r2_prefix="$2"
remote="r2:crescendai-bucket/$r2_prefix"
echo "[$(date -u +%H:%M:%S)] copy $local_path -> $remote"
rclone copy "$local_path" "$remote" --transfers=4
echo "[$(date -u +%H:%M:%S)] check $local_path vs $remote"
rclone check "$local_path" "$remote"
echo "[$(date -u +%H:%M:%S)] PARITY OK, deleting local $local_path"
rm -rf "$local_path"
echo "[$(date -u +%H:%M:%S)] DONE $local_path"
