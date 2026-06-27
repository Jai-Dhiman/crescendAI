#!/usr/bin/env bash
# Acquire the Chopin Etudes Op.10 & Op.25 exercise-book source (#49).
#
# Source: pl-wnifc/humdrum-chopin-first-editions (public domain; Chopin d.1849).
# The repo carries 42 .krn files tagged `!!!AGN: Etude` -- multiple FIRST
# EDITIONS of the same etudes. We keep the two COMPLETE single-edition sets
# (Op.10 = 010-1b-Sm Nos.1-12, Op.25 = 025-1b-LE Nos.1-12) = 24 distinct etudes,
# one edition family per opus, then convert **kern -> MIDI with verovio.
#
# The .mid files are gitignored (regenerable); this script is the durable record.
# Run from anywhere:
#   bash model/src/exercise_corpus/acquire_chopin_etudes.sh
#
# Requires: git, and the model venv with the `verovio` python module.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$SCRIPT_DIR/../../data/scores/exercise_primitives/raw/chopin_etudes"
STAGING="${CRESCENDAI_CORPUS_STAGING:-$HOME/crescendai_corpus_staging}"
REPO="$STAGING/humdrum-chopin-first-editions"
PY="${PYTHON:-$SCRIPT_DIR/../../.venv/bin/python}"
[ -x "$PY" ] || PY="$(cd "$SCRIPT_DIR/../../.." && pwd)/model/.venv/bin/python"

mkdir -p "$STAGING" "$RAW"
if [ ! -d "$REPO/kern" ]; then
  echo "cloning humdrum-chopin-first-editions into $STAGING ..."
  git clone --depth 1 https://github.com/pl-wnifc/humdrum-chopin-first-editions.git "$REPO"
fi

"$PY" - "$REPO/kern" "$RAW" <<'PYEOF'
import sys, glob, os, re
import verovio
src, out = sys.argv[1], sys.argv[2]
# Complete single-edition sets only: Op.10 (010-1b-Sm) + Op.25 (025-1b-LE).
files = sorted(f for f in glob.glob(os.path.join(src, "*.krn"))
               if re.search(r"0(10|25)-1b-", os.path.basename(f)))
if len(files) != 24:
    raise SystemExit(f"expected 24 complete-edition etudes, found {len(files)}: "
                     "repo layout may have changed")
tk = verovio.toolkit()
tk.setOptions({"inputFrom": "humdrum"})
ok = 0
for i, f in enumerate(files, 1):
    dst = os.path.join(out, f"chopin_etude_no{i:02d}.mid")
    if not tk.loadFile(f):
        raise SystemExit(f"verovio failed to load {os.path.basename(f)}")
    if not tk.renderToMIDIFile(dst):
        raise SystemExit(f"verovio failed to render MIDI for {os.path.basename(f)}")
    ok += 1
print(f"converted {ok}/24 Chopin etudes -> {out}")
PYEOF
