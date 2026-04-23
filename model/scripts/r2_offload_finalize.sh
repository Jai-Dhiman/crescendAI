#!/bin/bash
# Waits for AMT and model_improvement R2 uploads to reach parity, then deletes local.
# The initial background copies (Step 4, Step 7) only did copy+check; this finalizes them
# by re-verifying and deleting local once sizes match expected.
set -u
LOG="data/manifests/r2_offload_finalize.log"
echo "[$(date -u +%H:%M:%S)] finalize waiter started" | tee -a "$LOG"

for pair in "data/evals/youtube_amt:evals/youtube_amt" "data/checkpoints/model_improvement:checkpoints/archive/model_improvement"; do
  local_path="${pair%%:*}"
  r2_prefix="${pair##*:}"
  remote="r2:crescendai-bucket/$r2_prefix"
  if [[ ! -d "$local_path" ]]; then
    echo "[$(date -u +%H:%M:%S)] skip $local_path (already deleted)" | tee -a "$LOG"
    continue
  fi
  echo "[$(date -u +%H:%M:%S)] polling $local_path" | tee -a "$LOG"
  until rclone check "$local_path" "$remote" >/dev/null 2>&1; do
    sleep 300
    echo "[$(date -u +%H:%M:%S)] $local_path not yet at parity, keep waiting" | tee -a "$LOG"
  done
  echo "[$(date -u +%H:%M:%S)] $local_path PARITY OK, deleting local" | tee -a "$LOG"
  rm -rf "$local_path"
  echo "[$(date -u +%H:%M:%S)] $local_path deleted" | tee -a "$LOG"
done

echo "[$(date -u +%H:%M:%S)] finalize waiter done" | tee -a "$LOG"
