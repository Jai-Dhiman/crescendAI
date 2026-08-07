#!/usr/bin/env bash
# #167 (#104 S2): render + transcribe both synthesis arms over the 900-piece
# eval set. ~15h on this machine at the measured 0.198x realtime (6 workers x 2
# threads). Resumable -- rerun after a kill and it picks up from the cached
# transcriptions.
#
#   model/scripts/run_s2_arms.sh [data-root] [out-root]
#
# Deliberately NOT `set -e` on the python calls: run_arm exits non-zero when
# any item failed, and that must be reported per arm rather than aborting the
# second arm.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${1:-$REPO_ROOT/model/data}"
OUT_ROOT="${2:-$DATA_ROOT/results/mirex_s2_shift}"

cd "$REPO_ROOT" || exit 1

status=0
for arm in expressive deadpan; do
  echo "=== arm=$arm  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run --project model python -m claim_measurement.difficulty.synth_arm \
    --arm "$arm" \
    --data-root "$DATA_ROOT" \
    --out-root "$OUT_ROOT" \
    --workers 6 \
    --threads-per-worker 2 \
    --timeout-s 7200
  rc=$?
  echo "=== arm=$arm exit=$rc  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  [ "$rc" -ne 0 ] && status=1
done

echo "=== both arms done, status=$status ==="
exit "$status"
