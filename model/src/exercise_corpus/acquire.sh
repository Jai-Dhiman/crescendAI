#!/usr/bin/env bash
# Acquire the exercise-corpus per-piece MIDI from the Mutopia Project.
#
# The .mid files are gitignored (regenerable artifacts); this script is the
# durable, reproducible record of exactly what makes up the corpus. All sources
# are public domain (composers pre-1928). Run from anywhere:
#
#   bash model/src/exercise_corpus/acquire.sh
#
# Writes into model/data/scores/exercise_primitives/raw/<source>/. Hanon (ex.
# 1-20) is the one source NOT fetched here: it is a committed/ already-present
# MusicXML book, not Mutopia per-piece MIDI.
set -euo pipefail

# Anchor to model/ regardless of CWD (script lives at model/src/exercise_corpus/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$SCRIPT_DIR/../../data/scores/exercise_primitives/raw"
BASE="https://www.mutopiaproject.org/ftp"

dl() { curl -sS -L --fail -o "$2" "$1" && echo "ok  $(basename "$2")" || { echo "FAIL $1" >&2; return 1; }; }

mkdir -p "$RAW"/{bach_inventions,czerny_op840,satie,chopin_op28,burgmuller_op100}

echo "== Bach Two-Part Inventions BWV 772-786 (15) =="
for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15; do
  bwv=$((771 + 10#$i))
  dl "$BASE/BachJS/BWV${bwv}/bach-invention-${i}/bach-invention-${i}.mid" \
     "$RAW/bach_inventions/bach_invention_${i}.mid"
done

echo "== Czerny Op.840 Nos.1-10 =="
for n in 1 2 3 4 5 6 7 8 9 10; do
  nn=$(printf "%02d" "$n")
  dl "$BASE/CzernyC/O840/op840-${n}/op840-${n}.mid" \
     "$RAW/czerny_op840/czerny_op840_no${nn}.mid"
done

echo "== Satie Gymnopedies + Gnossiennes (6; asymmetric paths) =="
dl "$BASE/SatieE/gymnopedie_1/gymnopedie_1.mid" "$RAW/satie/satie_gymnopedie_1.mid"
dl "$BASE/SatieE/gymnopedie_2/gymnopedie_2.mid" "$RAW/satie/satie_gymnopedie_2.mid"
dl "$BASE/SatieE/gymnopedie_3/gymnopedie_3.mid" "$RAW/satie/satie_gymnopedie_3.mid"
dl "$BASE/SatieE/Gnossienne/no_1/no_1.mid"      "$RAW/satie/satie_gnossienne_1.mid"
dl "$BASE/SatieE/Gnossienne/no_2/no_2.mid"      "$RAW/satie/satie_gnossienne_2.mid"
dl "$BASE/SatieE/Gnossienne/no_3/no_3.mid"      "$RAW/satie/satie_gnossienne_3.mid"

echo "== Chopin Preludes Op.28 Nos.1-24 (No.4 Suffocation ed.) =="
for n in $(seq 1 24); do
  nn=$(printf "%02d" "$n")
  dl "$BASE/ChopinFF/O28/Chop-28-${n}/Chop-28-${n}.mid" \
     "$RAW/chopin_op28/chopin_op28_no${nn}.mid"
done

echo "== Burgmuller Op.100 Nos.1-18 =="
for n in $(seq 1 18); do
  nn=$(printf "%02d" "$n")
  dl "$BASE/BurgmullerJFF/O100/25EF-${nn}/25EF-${nn}.mid" \
     "$RAW/burgmuller_op100/burgmuller_op100_no${nn}.mid"
done

echo "Done. Corpus MIDI in $RAW"
