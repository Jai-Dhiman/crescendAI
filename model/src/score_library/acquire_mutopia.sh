#!/usr/bin/env bash
# Acquire the Mutopia "second wave" keyboard corpus for the piece-ID catalog.
#
# Two inputs, both regenerated here into the staging dir (gitignored, outside the
# repo): (1) a shallow clone of the Mutopia LilyPond sources -- needed ONLY for
# the authoritative `mutopiainstrument` header that the .mid export discards (see
# mutopia_filter.py); (2) the keyboard .mid files themselves, downloaded politely
# (2.5s spacing, ban-probe backoff) from the committed URL manifest.
#
# The URL manifest (model/data/manifests/mutopia_keyboard_manifest.tsv) is the
# durable record of WHICH Mutopia pieces this wave covers; the downloaded MIDIs +
# clone are regenerable artifacts.
#
# Run from anywhere:  bash model/src/score_library/acquire_mutopia.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="$MODEL_DIR/data/manifests/mutopia_keyboard_manifest.tsv"
STAGE="${MUTOPIA_STAGING_DIR:-$HOME/crescendai_corpus_staging}"
MIDI_DIR="$STAGE/mutopia_midi"
CLONE="$STAGE/mutopia"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"

[ -f "$MANIFEST" ] || { echo "ABORT: manifest not found at $MANIFEST" >&2; exit 1; }
mkdir -p "$MIDI_DIR" "$STAGE"

echo "== (1) clone Mutopia LilyPond sources (for instrument headers) =="
if [ -d "$CLONE/.git" ]; then
  git -C "$CLONE" pull --ff-only --quiet && echo "  pulled mutopia"
else
  git clone --depth 1 --quiet https://github.com/MutopiaProject/MutopiaProject.git "$CLONE" \
    && echo "  cloned mutopia"
fi

echo "== (2) probe for ban clearance =="
PROBE="https://www.mutopiaproject.org/ftp/BurgmullerJFF/O100/25EF-01/25EF-01.mid"
cleared=0
for i in $(seq 1 45); do
  code=$(curl -sS -A "$UA" -o /dev/null -w "%{http_code}" --max-time 20 "$PROBE" || true)
  if [ "$code" = "200" ]; then echo "  ban cleared (probe $i)"; cleared=1; break; fi
  echo "  probe $i: HTTP $code -> wait 120s"; sleep 120
done
[ "$cleared" = "1" ] || { echo "ABORT: still banned after ~90min" >&2; exit 2; }

echo "== (3) polite download (2.5s spacing) =="
ok=0; fail=0
while IFS=$'\t' read -r composer url fname; do
  [ -n "${fname:-}" ] || continue
  if [ -f "$MIDI_DIR/$fname" ]; then ok=$((ok+1)); continue; fi
  if curl -sS -A "$UA" -L --fail -o "$MIDI_DIR/$fname" "$url" 2>/dev/null; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); rm -f "$MIDI_DIR/$fname"
  fi
  sleep 2.5
done < "$MANIFEST"
echo "DONE ok=$ok fail=$fail on-disk=$(find "$MIDI_DIR" -name '*.mid' | wc -l | tr -d ' ')"
echo "Next: just catalog-add-mutopia  (filter -> dedup -> ingest)"
