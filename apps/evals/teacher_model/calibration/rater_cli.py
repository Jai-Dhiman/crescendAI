"""Founder rating CLI for the rubric calibration protocol.

The CLI presents one synthesis at a time to the founder, captures 11 sub-score
ratings + evidence quote + reason per synthesis, and writes append-only jsonl.

This file starts with the security-critical redaction function. Later tasks
add: T8 rating capture loop, T11 session cap, T14 resume-from-crash.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Fields from the source row that the rater is allowed to see. Anything not
# in this allow-list is stripped. Allow-list (not deny-list) is the safer
# discipline: a future schema change in baseline_v1.jsonl that adds a new
# judge field stays redacted by default.
_RATER_VISIBLE_FIELDS: frozenset[str] = frozenset({
    "synth_id",
    "piece_slug",
    "recording_id",
    "skill_bucket",
    "composer",
    "title",
    "synthesis_text",
    "muq_means",
})


def redact_for_rater(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k in _RATER_VISIBLE_FIELDS}


PHASE_1_SUB_SCORES: list[str] = [
    "ascf_process",
    "concrete_artifact_process",
    "praise_process",
    "autonomy_process",
    "scaffolded_process",
    "style_process",
    "tone_process",
    "autonomy_outcome",
    "tone_outcome",
    "concrete_artifact_outcome",
    "praise_outcome",
]


def capture_synthesis_ratings(
    redacted_row: dict,
    sub_scores: list[str],
    session_id: str,
    session_idx_start: int,
    output_path: Path,
    input_provider: Callable[[dict, str], tuple[int, str, str]],
) -> int:
    n_written = 0
    with output_path.open("a") as out:
        for i, sub_score in enumerate(sub_scores):
            value, evidence, reason = input_provider(redacted_row, sub_score)
            event = {
                "event_type": "rating",
                "synth_id": redacted_row["synth_id"],
                "anchor_origin_id": redacted_row.get("anchor_origin_id"),
                "sub_score": sub_score,
                "value": value,
                "evidence": evidence,
                "reason": reason,
                "session_id": session_id,
                "session_idx": session_idx_start + i,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(event) + "\n")
            n_written += 1
    return n_written
