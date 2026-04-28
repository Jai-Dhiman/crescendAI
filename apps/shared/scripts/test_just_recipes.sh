#!/usr/bin/env bash
# scripts/test_just_recipes.sh
set -euo pipefail
cd "$(dirname "$0")/.."

rm -f apps/api/src/lib/playbook.json
just compile-playbook >/dev/null
test -f apps/api/src/lib/playbook.json || { echo "FAIL: compile did not produce playbook.json"; exit 1; }

just check-playbook-sync >/dev/null || { echo "FAIL: sync check failed after fresh compile"; exit 1; }

echo "{}" > apps/api/src/lib/playbook.json
if just check-playbook-sync >/dev/null 2>&1; then
  echo "FAIL: sync check should have failed on stale JSON"
  exit 1
fi

just compile-playbook >/dev/null
echo "PASS"
