#!/usr/bin/env bash
# download_test_data.sh -- Reproduces the audio gate test dataset
# Requires: yt-dlp, ffmpeg
# Piano clips are symlinked from model/data/evals/youtube_amt/
# Non-piano clips are downloaded from YouTube
# Mixed clips are synthesized from existing clips
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

mkdir -p "$DATA_DIR"/{piano,speech,noise,quiet_piano,mixed}

echo "=== Linking piano clips from youtube_amt ==="
PIANO_IDS="RuGl4Jnbv7A JnXqNRVLUvg 7pwRjv4cdLA JSg4D0Nwio4 rZk1HkgTxL4 2GuHebuElok 2IOJBILVEhY 0NC0ca1rLpE 7ZuHQaD62Fg B0OyrsmPTlU 7oeGflOpmvM e4gyr2Pjc7Y 0s60_AoMHM0 bfoqeJZn5AI ckNuyscRSQo mZdBzID0kVI he98w0WCGIg G8cOZdu9C5I sBQXzKKfhAQ LqoV4ZW7xTA"
for id in $PIANO_IDS; do
  src="$REPO_ROOT/model/data/evals/youtube_amt/${id}.wav"
  if [ -f "$src" ]; then
    ln -sf "$src" "$DATA_DIR/piano/${id}.wav"
  else
    echo "WARNING: Missing piano source: $src"
  fi
done
echo "Piano: $(ls "$DATA_DIR/piano/"*.wav 2>/dev/null | wc -l | tr -d ' ') clips"

download_clip() {
  local url="$1" output="$2" start="$3" end="$4"
  if [ -f "$output" ]; then
    echo "  Already exists: $(basename "$output")"
    return 0
  fi
  yt-dlp -x --audio-format wav --postprocessor-args "-ar 24000 -ac 1" \
    --download-sections "*${start}-${end}" \
    -o "$output" "$url" 2>/dev/null || echo "  FAILED: $url"
}

echo ""
echo "=== Downloading speech clips ==="
download_clip "https://www.youtube.com/watch?v=arj7oStGLkU" "$DATA_DIR/speech/arj7oStGLkU.wav" "0:30" "0:45"
download_clip "https://www.youtube.com/watch?v=UF8uR6Z6KLc" "$DATA_DIR/speech/UF8uR6Z6KLc.wav" "0:30" "0:45"
download_clip "https://www.youtube.com/watch?v=8S0FDjFBj8o" "$DATA_DIR/speech/8S0FDjFBj8o.wav" "0:30" "0:45"
download_clip "https://www.youtube.com/watch?v=W3I3kAg2J7w" "$DATA_DIR/speech/W3I3kAg2J7w.wav" "0:30" "0:45"
download_clip "https://www.youtube.com/watch?v=3PuHGKnboNY" "$DATA_DIR/speech/3PuHGKnboNY.wav" "1:00" "1:15"
echo "Speech: $(ls "$DATA_DIR/speech/"*.wav 2>/dev/null | wc -l | tr -d ' ') clips"

echo ""
echo "=== Downloading noise clips ==="
download_clip "https://www.youtube.com/watch?v=8s5H76F3SIs" "$DATA_DIR/noise/8s5H76F3SIs.wav" "0:30" "0:45"
download_clip "https://www.youtube.com/watch?v=mPZkdNFkNps" "$DATA_DIR/noise/mPZkdNFkNps.wav" "0:30" "0:45"
download_clip "https://www.youtube.com/watch?v=sGkh1W5cbH4" "$DATA_DIR/noise/sGkh1W5cbH4.wav" "0:30" "0:45"
download_clip "https://www.youtube.com/watch?v=gaGrHUekGrc" "$DATA_DIR/noise/gaGrHUekGrc.wav" "1:00" "1:15"
download_clip "https://www.youtube.com/watch?v=1KaOrSuWZeM" "$DATA_DIR/noise/1KaOrSuWZeM.wav" "0:30" "0:45"
echo "Noise: $(ls "$DATA_DIR/noise/"*.wav 2>/dev/null | wc -l | tr -d ' ') clips"

echo ""
echo "=== Downloading quiet piano clips ==="
download_clip "https://www.youtube.com/watch?v=CvFH_6DNRCY" "$DATA_DIR/quiet_piano/CvFH_6DNRCY.wav" "0:10" "0:25"
download_clip "https://www.youtube.com/watch?v=9E6b3swbnWg" "$DATA_DIR/quiet_piano/9E6b3swbnWg.wav" "0:10" "0:25"
download_clip "https://www.youtube.com/watch?v=4Tr0otuiQuU" "$DATA_DIR/quiet_piano/4Tr0otuiQuU.wav" "0:10" "0:25"
download_clip "https://www.youtube.com/watch?v=wygy721nzRc" "$DATA_DIR/quiet_piano/wygy721nzRc.wav" "0:10" "0:25"
download_clip "https://www.youtube.com/watch?v=KpOtuoHL45Y" "$DATA_DIR/quiet_piano/KpOtuoHL45Y.wav" "0:10" "0:25"
echo "Quiet piano: $(ls "$DATA_DIR/quiet_piano/"*.wav 2>/dev/null | wc -l | tr -d ' ') clips"

echo ""
echo "=== Downloading mixed clips ==="
download_clip "https://www.youtube.com/watch?v=827jmswqnEA" "$DATA_DIR/mixed/piano_lesson_1.wav" "2:00" "2:15"
download_clip "https://www.youtube.com/watch?v=zGBXA1tBiLw" "$DATA_DIR/mixed/masterclass_1.wav" "3:00" "3:15"

echo ""
echo "=== Synthesizing mixed clips ==="
PIANO_15S="/tmp/audio_gate_piano_15s.wav"
ffmpeg -y -i "$DATA_DIR/piano/RuGl4Jnbv7A.wav" -ss 30 -t 15 -ar 24000 -ac 1 "$PIANO_15S" 2>/dev/null

if [ ! -f "$DATA_DIR/mixed/synth_piano_speech.wav" ]; then
  ffmpeg -y -i "$PIANO_15S" -i "$DATA_DIR/speech/arj7oStGLkU.wav" \
    -filter_complex "[0:a][1:a]amix=inputs=2:duration=shortest" \
    -ar 24000 -ac 1 "$DATA_DIR/mixed/synth_piano_speech.wav" 2>/dev/null
fi

if [ ! -f "$DATA_DIR/mixed/synth_piano_cafe.wav" ]; then
  ffmpeg -y -i "$PIANO_15S" -i "$DATA_DIR/noise/gaGrHUekGrc.wav" \
    -filter_complex "[0:a]volume=0.7[a];[1:a]volume=0.3[b];[a][b]amix=inputs=2:duration=shortest" \
    -ar 24000 -ac 1 "$DATA_DIR/mixed/synth_piano_cafe.wav" 2>/dev/null
fi

if [ ! -f "$DATA_DIR/mixed/synth_quiet_piano_speech.wav" ]; then
  ffmpeg -y -i "$DATA_DIR/quiet_piano/CvFH_6DNRCY.wav" -i "$DATA_DIR/speech/UF8uR6Z6KLc.wav" \
    -filter_complex "[0:a][1:a]amix=inputs=2:duration=shortest" \
    -ar 24000 -ac 1 "$DATA_DIR/mixed/synth_quiet_piano_speech.wav" 2>/dev/null
fi

rm -f "$PIANO_15S"
echo "Mixed: $(ls "$DATA_DIR/mixed/"*.wav 2>/dev/null | wc -l | tr -d ' ') clips"

echo ""
echo "=== DATASET SUMMARY ==="
total=0
for dir in piano speech noise quiet_piano mixed; do
  count=$(ls "$DATA_DIR/$dir/"*.wav 2>/dev/null | wc -l | tr -d ' ')
  total=$((total + count))
  echo "  $dir: $count"
done
echo "  TOTAL: $total clips"
