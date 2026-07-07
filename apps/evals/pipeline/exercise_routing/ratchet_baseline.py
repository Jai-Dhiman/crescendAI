"""Ratchet script: promote last_run.json -> baseline.json for exercise-routing eval.

Called by `just exercise-routing-ratchet`. Not intended for direct invocation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parents[2] / "results" / "exercise_routing"
LAST_RUN_PATH = RESULTS_DIR / "last_run.json"
BASELINE_PATH = RESULTS_DIR / "baseline.json"

if not LAST_RUN_PATH.exists():
    print(f"ERROR: {LAST_RUN_PATH} not found -- run just exercise-routing-eval first", file=sys.stderr)
    sys.exit(1)

last = json.loads(LAST_RUN_PATH.read_text())
baseline = json.loads(BASELINE_PATH.read_text())

out = {
    "invocation_rate_floor": last["invocation_rate"],
    "kind_correctness_floor": last["kind_correctness_rate"],
    "dimension_match_floor": last["dimension_match_rate"],
    "bar_range_grounding_floor": last["bar_range_grounding_rate"],
    "tempo_sanity_floor": last["tempo_sanity_rate"],
    "notes": baseline.get("notes", ""),
}

# selector_relevance_at_1 is only present when the run included the judge pass;
# ratchet it when measured, otherwise carry the existing floor forward unchanged.
if "selector_relevance_at_1" in last and last.get("selector_relevance_n", 0) > 0:
    out["selector_relevance_at_1_floor"] = last["selector_relevance_at_1"]
elif "selector_relevance_at_1_floor" in baseline:
    out["selector_relevance_at_1_floor"] = baseline["selector_relevance_at_1_floor"]
BASELINE_PATH.write_text(json.dumps(out, indent=2))
print("ratcheted baseline.json from last_run.json")
