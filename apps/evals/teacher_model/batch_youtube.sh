#!/usr/bin/env bash
# Sequential overnight YouTube transcription batch.
# Runs one channel at a time with inter-channel pauses to avoid rate limiting.
# Uses Chrome cookies for authentication.
#
# Usage:
#   cd apps/evals
#   bash teacher_model/batch_youtube.sh 2>&1 | tee /tmp/batch_youtube.log
#
# The script is safe to re-run: already-transcribed videos are skipped
# because their corpus file already exists (process_video checks this via
# the provenance manifest).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EVALS_DIR"

BROWSER="chrome"
LIMIT=50          # videos per channel for this run
INTER_CHANNEL_PAUSE=90  # seconds between channels

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_channel() {
    local channel="$1"
    local manifest="$2"
    local limit="${3:-$LIMIT}"
    log "Starting channel: $channel (limit=$limit)"
    uv run python -m teacher_model.transcribe \
        --channel "$channel" \
        --limit "$limit" \
        --provider assemblyai \
        --cookies-from-browser "$BROWSER" \
        --manifest "teacher_model/data/$manifest" \
        --tier tier1_youtube \
        || log "WARN: channel $channel exited non-zero (partial results may be saved)"
    log "Done: $channel. Pausing ${INTER_CHANNEL_PAUSE}s..."
    sleep "$INTER_CHANNEL_PAUSE"
}

# ---------------------------------------------------------------------------
# Phase 1: Re-run the 6 channels that failed due to rate limiting
# These already have partial results — the pipeline will skip saved videos.
# ---------------------------------------------------------------------------

log "=== PHASE 1: Re-running rate-limited channels ==="

run_channel "https://www.youtube.com/@tonebasePiano/videos"  "provenance_tonebase.jsonl"   50
run_channel "https://www.youtube.com/@CurtisInstitute/videos" "provenance_curtis.jsonl"    50
run_channel "https://www.youtube.com/rcmlondon/videos"        "provenance_rcm.jsonl"        50
run_channel "https://www.youtube.com/@CarnegiehHall/videos"   "provenance_carnegie.jsonl"   50
run_channel "https://www.youtube.com/@chopininstitute/videos" "provenance_chopin.jsonl"     50
run_channel "https://www.youtube.com/@RoyalAcademyofMusic/videos" "provenance_ram.jsonl"   50

log "=== PHASE 1 complete ==="

# ---------------------------------------------------------------------------
# Phase 2: Extend the two highest-yield channels (Josh Wright + Nahre Sol)
# Already have 21 and 13 videos respectively — extend to 200 each.
# ---------------------------------------------------------------------------

log "=== PHASE 2: Extending top-yield channels ==="

run_channel "https://www.youtube.com/joshwrightpiano/videos"  "provenance_joshwright.jsonl" 200
run_channel "https://www.youtube.com/@NahreSol/videos"        "provenance_nahresol.jsonl"   200

log "=== PHASE 2 complete ==="

# ---------------------------------------------------------------------------
# Phase 3: Status check
# ---------------------------------------------------------------------------

log "=== Final corpus status ==="
uv run python -m teacher_model.corpus_builder status

log "=== Batch complete ==="
