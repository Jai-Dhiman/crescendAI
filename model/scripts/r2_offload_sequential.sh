#!/bin/bash
# Sequential, resumable R2 offload for the three pending targets.
# Order: smallest first, so AMT clears in <2h and model_improvement before raw/competition starts.
# rclone copy is idempotent — already-uploaded files are skipped.
set -u
LOG="data/manifests/r2_offload_sequential.log"
echo "[$(date -u +%F\ %H:%M:%S)] sequential offload start, pid=$$" | tee -a "$LOG"

run_one() {
  local local_path="$1"
  local r2_prefix="$2"
  local remote="r2:crescendai-bucket/$r2_prefix"
  if [[ ! -d "$local_path" ]]; then
    echo "[$(date -u +%H:%M:%S)] skip $local_path (already deleted)" | tee -a "$LOG"
    return 0
  fi
  echo "[$(date -u +%H:%M:%S)] copy $local_path -> $remote" | tee -a "$LOG"
  rclone copy "$local_path" "$remote" --transfers=4 --retries=10 --low-level-retries=20 \
    --stats=5m --stats-one-line 2>&1 | tee -a "$LOG"
  echo "[$(date -u +%H:%M:%S)] check $local_path" | tee -a "$LOG"
  if rclone check "$local_path" "$remote" 2>&1 | tee -a "$LOG" | grep -q "0 differences"; then
    echo "[$(date -u +%H:%M:%S)] PARITY OK -- deleting $local_path" | tee -a "$LOG"
    rm -rf "$local_path"
  else
    echo "[$(date -u +%H:%M:%S)] PARITY FAIL on $local_path -- LOCAL KEPT" | tee -a "$LOG"
    return 1
  fi
}

run_one data/evals/youtube_amt evals/youtube_amt
run_one data/checkpoints/model_improvement checkpoints/archive/model_improvement
run_one data/raw/competition raw/competition

echo "[$(date -u +%F\ %H:%M:%S)] sequential offload finished" | tee -a "$LOG"
