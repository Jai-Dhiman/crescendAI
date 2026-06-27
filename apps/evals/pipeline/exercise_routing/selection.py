"""Python mirror of corpus-drill.ts selectPrimitive (deterministic top-1).

The serving layer's selection is pure and deterministic given the routing
decision + manifest: filter by target_dimension, sort by (suffixNum, id), take
top-1, widen to hanon_001 if the bucket is empty. Re-deriving it here lets the
relevance eval know WHICH drill is served WITHOUT consuming the pending exercise,
which matters because selection happens at consume-time, not synthesis-time.

This mirror is faithful ONLY to the current deterministic selector. Once cosine
selection lands (serve-time, embedding-dependent), the served primitive must be
surfaced from the system (eval_context) instead -- a Python re-derivation cannot
reproduce an embedding-conditioned choice. A parity test pins this mirror to the
TS behavior so drift fails loudly.
"""
from __future__ import annotations

import re

from pipeline.exercise_routing.relevance import DrillInfo, RelevanceCase

WIDEN_DEFAULT_PRIMITIVE = "hanon_001"
_SUFFIX_RE = re.compile(r"_(\d+)$")


def _suffix_num(primitive_id: str) -> float:
    m = _SUFFIX_RE.search(primitive_id)
    return float(int(m.group(1))) if m else float("inf")


def _stable_sorted(ids: list[str]) -> list[str]:
    # (suffixNum asc, id asc) -- exactly mirrors stableSorted in corpus-drill.ts.
    return sorted(ids, key=lambda i: (_suffix_num(i), i))


def select_primitive(routing: dict, manifest: dict) -> tuple[str, bool]:
    """Return (primitive_id, widened) for a routing decision. Mirrors TS.

    Raises:
        KeyError: if the bucket is empty AND the widen default is absent from the
            manifest (a build error, matching the TS throw).
    """
    explicit = routing.get("primitive_id")
    if explicit and explicit in manifest:
        return explicit, False

    target = routing.get("target_dimension")
    matches = _stable_sorted(
        [pid for pid, e in manifest.items() if target in e.get("dimensions", [])]
    )
    if matches:
        return matches[0], False

    if WIDEN_DEFAULT_PRIMITIVE not in manifest:
        raise KeyError(
            f"WIDEN_DEFAULT_PRIMITIVE {WIDEN_DEFAULT_PRIMITIVE!r} absent from manifest"
        )
    return WIDEN_DEFAULT_PRIMITIVE, True


def _weakness_context(teaching_moments: list[dict], target_dimension: str | None) -> str:
    """Pick the diagnosis text the drill is meant to address.

    Prefer the (negative) moment whose dimension matches the drill's target;
    fall back to any moment matching the target; then the top moment. The
    moment's `reasoning` is the human-readable description of what went wrong.
    """
    if not teaching_moments:
        return ""
    matching = [m for m in teaching_moments if m.get("dimension") == target_dimension]
    negative_matching = [m for m in matching if not m.get("is_positive", False)]
    chosen = (negative_matching or matching or teaching_moments)[0]
    return str(chosen.get("reasoning", "")).strip()


def build_relevance_case(capture, manifest: dict) -> RelevanceCase | None:
    """Build a RelevanceCase from a SessionCapture, or None if not judgeable.

    Relevance@1 judges CORPUS-DRILL selection: own_passage_loop replays the
    student's own bars (no catalog choice to judge), and null prescriptions have
    nothing to judge. Both return None.
    """
    ex = capture.prescribed_exercise
    if ex is None or ex.get("kind") != "corpus_drill":
        return None

    primitive_id, _widened = select_primitive(ex, manifest)
    entry = manifest[primitive_id]
    drill = DrillInfo(
        primitive_id=primitive_id,
        title=entry["title"],
        source=entry["source"],
        dimensions=entry["dimensions"],
        techniques=entry["techniques"],
    )

    target = ex.get("target_dimension")
    bar = ex.get("bar_range")
    bar_range = (int(bar[0]), int(bar[1])) if bar else None
    return RelevanceCase(
        weakness_dimension=target,
        weakness_context=_weakness_context(capture.teaching_moments, target),
        bar_range=bar_range,
        drill=drill,
    )


def _top_negative_moment(teaching_moments: list[dict], dimension: str) -> dict | None:
    matching = [
        m
        for m in teaching_moments
        if m.get("dimension") == dimension and not m.get("is_positive", False)
    ]
    return matching[0] if matching else None


def build_selector_case(capture, manifest: dict) -> RelevanceCase | None:
    """Build a COUNTERFACTUAL relevance case for the session's dominant weakness.

    Independent of the routed kind: it asks "if a corpus drill were prescribed for
    this weakness, would the deterministic selector's choice be appropriate?" This
    is the selector-quality signal Goal B (cosine selection) moves, and unlike the
    actual-corpus_drill subset it is powered across every invoked session (which
    are mostly own_passage_loop once piece-ID resolves).

    Returns None when there is no dominant weakness dimension to drill.
    """
    dim = capture.dominant_dimension
    if not dim:
        return None

    routing = {"kind": "corpus_drill", "target_dimension": dim}
    primitive_id, _widened = select_primitive(routing, manifest)
    entry = manifest[primitive_id]
    drill = DrillInfo(
        primitive_id=primitive_id,
        title=entry["title"],
        source=entry["source"],
        dimensions=entry["dimensions"],
        techniques=entry["techniques"],
    )

    moment = _top_negative_moment(capture.teaching_moments, dim)
    bar = moment.get("bar_range") if moment else None
    bar_range = (int(bar[0]), int(bar[1])) if bar else None
    return RelevanceCase(
        weakness_dimension=dim,
        weakness_context=_weakness_context(capture.teaching_moments, dim),
        bar_range=bar_range,
        drill=drill,
    )
