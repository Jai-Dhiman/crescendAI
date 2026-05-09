"""Founder rating CLI for the rubric calibration protocol.

The CLI presents one synthesis at a time to the founder, captures 11 sub-score
ratings + evidence quote + reason per synthesis, and writes append-only jsonl.

This file starts with the security-critical redaction function. Later tasks
add: T8 rating capture loop, T11 session cap, T14 resume-from-crash.
"""
from __future__ import annotations

from typing import Any

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
