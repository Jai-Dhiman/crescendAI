#!/usr/bin/env bash
# Acquire the #49 KernScores corpus expansion: per-piece MIDI converted from
# craigsapp humdrum (**kern) GitHub repos via the verovio python module.
#
# All sources are unambiguously public domain (composers pre-1928) AND
# commercial-clean (no NC/SA encumbrance on the engravings -- craigsapp humdrum
# is released into the public domain). This is the durable, reproducible record
# of exactly what the KernScores half of the exercise corpus is made of; the
# .mid outputs are gitignored regenerable artifacts.
#
# Run from anywhere:
#   bash model/src/exercise_corpus/acquire_kernscores.sh
#
# Writes into model/data/scores/exercise_primitives/raw/<source>/.
#
# DEDUP NOTE: craigsapp `chopin-preludes` (Op.28) is intentionally NOT acquired
# -- it duplicates the #17 `chopin` (Mutopia Op.28) source. No other craigsapp
# repo overlaps the #17 six.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RAW="$MODEL_DIR/data/scores/exercise_primitives/raw"
CLONE="${KERNSCORES_CLONE_DIR:-$HOME/crescendai_corpus_staging/kernscores}"
mkdir -p "$RAW" "$CLONE"

# repo-name -> raw/<source-dir> mapping. chopin-preludes deliberately omitted.
declare -a REPOS=(
  "beethoven-piano-sonatas:beethoven_sonatas"
  "mozart-piano-sonatas:mozart_sonatas"
  "scarlatti-keyboard-sonatas:scarlatti_sonatas"
  "haydn-keyboard-sonatas:haydn_sonatas"
  "joplin:joplin_rags"
  "chopin-mazurkas:chopin_mazurkas"
)

echo "== Cloning/refreshing craigsapp humdrum repos =="
for pair in "${REPOS[@]}"; do
  repo="${pair%%:*}"
  if [ -d "$CLONE/$repo/.git" ]; then
    git -C "$CLONE/$repo" pull --ff-only --quiet && echo "  pulled $repo"
  else
    git clone --depth 1 --quiet "https://github.com/craigsapp/$repo.git" "$CLONE/$repo" \
      && echo "  cloned $repo"
  fi
done

echo "== Converting **kern -> MIDI via verovio (kern/*.krn) =="
# verovio is a python dependency of the model package (see pyproject). Convert
# every kern/*.krn in each repo; failures are logged LOUDLY (no silent skips)
# and a per-source summary is printed. A source yielding ZERO MIDIs aborts.
uv run --project "$MODEL_DIR" python - "$RAW" "$CLONE" "${REPOS[@]}" <<'PY'
import base64
import sys
from pathlib import Path

import verovio

raw = Path(sys.argv[1])
clone = Path(sys.argv[2])
pairs = sys.argv[3:]

tk = verovio.toolkit()
tk.setInputFrom("humdrum")

total_ok = total_fail = 0
failures: list[str] = []
for pair in pairs:
    repo, dest = pair.split(":", 1)
    src_dir = clone / repo / "kern"
    out_dir = raw / dest
    out_dir.mkdir(parents=True, exist_ok=True)
    krns = sorted(src_dir.glob("*.krn"))
    if not krns:
        raise SystemExit(f"ABORT: no .krn found in {src_dir} (clone failed?)")
    ok = fail = 0
    for krn in krns:
        mid = out_dir / (krn.stem + ".mid")
        try:
            if not tk.loadFile(str(krn)):
                raise RuntimeError("verovio loadFile returned False")
            b64 = tk.renderToMIDI()
            mid.write_bytes(base64.b64decode(b64))
            ok += 1
        except Exception as e:  # noqa: BLE001 -- surface every conversion failure
            fail += 1
            failures.append(f"{repo}/{krn.name}: {e}")
            print(f"  FAIL {repo}/{krn.name}: {e}", file=sys.stderr)
    print(f"  {dest:20s} ok={ok:3d} fail={fail:2d}")
    if ok == 0:
        raise SystemExit(f"ABORT: source {dest} produced ZERO MIDIs")
    total_ok += ok
    total_fail += fail

print(f"\nTOTAL ok={total_ok} fail={total_fail}")
if failures:
    print("FAILED CONVERSIONS (upstream **kern that verovio could not render):")
    for f in failures:
        print(f"  - {f}")
PY

echo "Done. KernScores corpus MIDI in $RAW/{beethoven_sonatas,mozart_sonatas,scarlatti_sonatas,haydn_sonatas,joplin_rags,chopin_mazurkas}"
