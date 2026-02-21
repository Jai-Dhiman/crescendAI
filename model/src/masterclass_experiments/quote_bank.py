"""Quote bank construction from teaching moments."""

from __future__ import annotations

SEVERITY_ORDER = {"critical": 0, "significant": 1, "moderate": 2, "minor": 3}


def build_quote_bank(
    moments: list[dict],
    assignments: dict[str, str],
    max_per_dim: int = 10,
) -> dict[str, list[dict]]:
    """Build a quote bank organized by dimension.

    Args:
        moments: List of enriched moment dicts.
        assignments: {moment_id: dimension_name} mapping.
        max_per_dim: Max quotes per dimension.

    Returns:
        {dimension_name: [quote_entry, ...]} sorted by severity then feedback_type.
    """
    by_dim: dict[str, list[dict]] = {}

    for moment in moments:
        mid = moment["moment_id"]
        dim = assignments.get(mid)
        if dim is None:
            continue

        entry = {
            "moment_id": mid,
            "teacher": moment.get("teacher", "Unknown"),
            "feedback_summary": moment["feedback_summary"],
            "transcript_excerpt": _extract_excerpt(moment.get("transcript_text", "")),
            "severity": moment.get("severity", "moderate"),
            "feedback_type": moment.get("feedback_type", "suggestion"),
            "piece": moment.get("piece"),
            "composer": moment.get("composer"),
        }
        by_dim.setdefault(dim, []).append(entry)

    # Sort by severity (critical first) then feedback_type
    for dim in by_dim:
        by_dim[dim].sort(key=lambda e: SEVERITY_ORDER.get(e["severity"], 99))
        by_dim[dim] = by_dim[dim][:max_per_dim]

    return by_dim


def _extract_excerpt(transcript_text: str, max_chars: int = 500) -> str:
    """Extract a representative excerpt from the transcript context."""
    lines = transcript_text.strip().split("\n")
    # Skip music notation lines (just symbols)
    content_lines = [l for l in lines if not _is_music_notation(l)]
    excerpt = "\n".join(content_lines)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + "..."
    return excerpt


def _is_music_notation(line: str) -> bool:
    """Check if a transcript line is just music notation symbols."""
    stripped = line.strip()
    # Remove timestamp prefix like "[507.8s]"
    if stripped.startswith("[") and "]" in stripped:
        stripped = stripped[stripped.index("]") + 1 :].strip()
    # Lines that are empty or just music symbols
    return not stripped or all(c in "♪♫ \t" for c in stripped)
