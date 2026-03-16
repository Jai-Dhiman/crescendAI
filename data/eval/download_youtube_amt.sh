#!/bin/bash
# Download the 50 YouTube AMT validation recordings for eval.
# Extracts audio as 24kHz mono WAV (matching production pipeline).
#
# Usage: bash data/eval/download_youtube_amt.sh

set -e

RECORDINGS_JSONL="model/data/intermediate_cache/recordings.jsonl"
OUTPUT_DIR="data/eval/youtube_amt"

if [ ! -f "$RECORDINGS_JSONL" ]; then
    echo "ERROR: $RECORDINGS_JSONL not found. Run from project root."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Count total
TOTAL=$(wc -l < "$RECORDINGS_JSONL" | tr -d ' ')
echo "Downloading $TOTAL recordings to $OUTPUT_DIR/"
echo ""

i=0
while IFS= read -r line; do
    i=$((i + 1))
    VIDEO_ID=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['video_id'])")
    OUTPUT_FILE="$OUTPUT_DIR/${VIDEO_ID}.wav"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "[$i/$TOTAL] $VIDEO_ID -- already exists, skipping"
        continue
    fi

    echo "[$i/$TOTAL] $VIDEO_ID -- downloading..."
    yt-dlp \
        --extract-audio \
        --audio-format wav \
        --postprocessor-args "ffmpeg:-ar 24000 -ac 1" \
        --output "$OUTPUT_DIR/%(id)s.%(ext)s" \
        --no-playlist \
        --quiet \
        "https://www.youtube.com/watch?v=${VIDEO_ID}" 2>&1 || {
            echo "  FAILED: $VIDEO_ID (may be unavailable or region-locked)"
        }
done < "$RECORDINGS_JSONL"

echo ""
echo "Done. Downloaded files:"
ls -1 "$OUTPUT_DIR"/*.wav 2>/dev/null | wc -l | tr -d ' '
echo " WAV files in $OUTPUT_DIR/"
