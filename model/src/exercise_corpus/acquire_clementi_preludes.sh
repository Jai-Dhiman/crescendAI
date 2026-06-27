#!/usr/bin/env bash
# Acquire Clementi's Op.42 key-preludes (#49 exercise-book expansion).
#
# Source: Mutopia Project, Clementi Op.42 "Introduction to the Art of Playing on
# the Piano Forte" -- a public-domain method book (Clementi d.1832). The book's
# prebuilt MIDI zip holds 78 lessons; we ingest ONLY Clementi's own 18 short
# key-preludes (the *-prel-* files, 3-6 bars each). The other 60 lessons are
# arrangements of OTHER composers (Scarlatti/Mozart/Handel/etc.) -- repertoire
# by attribution, not Clementi drills -- and are skipped. No LilyPond needed:
# Mutopia ships the compiled MIDI.
#
# The .mid files are gitignored (regenerable); this script is the durable record.
# Run from anywhere:
#   bash model/src/exercise_corpus/acquire_clementi_preludes.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$SCRIPT_DIR/../../data/scores/exercise_primitives/raw/clementi_preludes"
URL="https://www.mutopiaproject.org/ftp/ClementiM/O42/clementi-op42/clementi-op42-mids.zip"

mkdir -p "$RAW"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "downloading Clementi Op.42 MIDI zip ..."
curl -sS -L --fail --max-time 60 -o "$tmp/op42.zip" "$URL"
# Extract only the prelude lessons (Clementi's own key-preludes).
unzip -o -j "$tmp/op42.zip" '*prel*' -d "$tmp/prel" >/dev/null

n=0
for f in $(ls "$tmp"/prel/*prel*.mid | sort); do
  les="$(basename "$f" | sed -E 's/.*les([0-9]+).*/\1/')"
  cp "$f" "$RAW/clementi_prelude_les${les}.mid"
  n=$((n + 1))
done
if [ "$n" -ne 18 ]; then
  echo "ERROR: expected 18 Clementi preludes, extracted $n (zip layout may have changed)" >&2
  exit 1
fi
echo "extracted $n Clementi Op.42 preludes -> $RAW"
