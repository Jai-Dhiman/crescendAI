"""The experimental piece-ID gate -- the SINGLE module an autoresearch loop edits.

This is the knob-board for hill-climbing piece-ID accuracy against the
comprehensive eval (pieceid_autoresearch_eval). Everything here defaults to a
faithful reproduction of the FROZEN production gate (chroma top-K whole-piece
shortlist -> pitch-only chord-Jaccard elastic-DTW -> margin accept), so a no-op
run reproduces the committed baseline metric. An autoresearch experiment changes
ONE lever, re-runs the verify command, and keeps the change only if the metric
improves and the false-accept guard holds.

HARD INVARIANT: this is an OFFLINE EVAL surface only. The production gate lives in
Rust/WASM (apps/api/src/wasm/piece-identify) and its golden parity fixture; NEVER
edit those to chase the eval. A winning lever here becomes a PROPOSAL to port, not
an automatic production change. Changing a lever must not require touching the
frozen `eval_query` or the parity test.

Levers (the search space):
  SHORTLIST_MODE   "whole_piece" (one chroma vector/piece) | "windowed" (several
                   overlapping note-window vectors/piece -> mid-piece queries can
                   match a local window instead of the whole-piece histogram; the
                   axis-B fix).
  WINDOW_NOTES     window length (notes) for the windowed shortlist.
  WINDOW_HOP       hop (notes) between windows.
  TOP_K            shortlist size handed to the elastic gate.
  MARGIN_THRESHOLD accept iff (2nd_cost - best_cost) >= this.
  ABS_COST_FLOOR   if not None, additionally require best_cost <= this (an absolute
                   alignment-quality floor on top of the relative margin).
  AMT_SOURCE_EXTRA_MARGIN  extra margin required to lock when the best candidate is
                   an AMT-transcribed distractor (giantmidi/pdmx) -- the corpus that
                   carries ~all genuine false-accepts. Source-aware acceptance.
"""
from __future__ import annotations

import os

import numpy as np

from piece_id_eval.note_chroma import chroma_vector
from piece_id_eval.notes import Note
from piece_id_eval.stage0c_elastic_dtwgate import ElasticGate, _notes_to_events
from score_library.bulk_ingest import W_PITCH, W_TIME, _RunningChromaIndex


# ---------------------------------------------------------------------------
# LEVERS  (autoresearch edits these)
# ---------------------------------------------------------------------------
# Each lever may be overridden for a sweep via an env var (PIECEID_<NAME>) so the
# autoresearch loop can try a value without rewriting this file; the literal
# defaults below remain the faithful no-op reproduction of the frozen gate. A
# KEPT win is written back into the literal default (the knob-board is the port
# proposal), not left as an env var.

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(f"PIECEID_{name}")
    return int(v) if v is not None else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(f"PIECEID_{name}")
    return float(v) if v is not None else default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(f"PIECEID_{name}", default)


def _env_opt_float(name: str, default: float | None) -> float | None:
    v = os.environ.get(f"PIECEID_{name}")
    if v is None:
        return default
    return None if v.lower() in ("", "none") else float(v)


# WINNING LEVER SET (autoresearch #96, 2026-06-27): an ADDITIVE hybrid shortlist
# -- whole-piece top-20 UNION windowed top-8 (400-note windows, hop 200) -- lifts
# held-out mid-piece recognition from 36.6% to 67.1% while genuine open-set
# false-accepts DROP (test 4.9%->2.4%). See the port proposal; defaults below are
# the proposed production operating point (NOT yet ported to Rust/WASM).
#
# PRODUCTION-FAITHFUL CAVEAT (full-piece re-tune, #96): the numbers above are on a
# catalog CAPPED at each piece's first 600 notes. Production build_piece_index
# fingerprints the FULL piece (no cap), and on full pieces the whole-piece chroma
# recall COLLAPSES (opening recog 80%->44%). The hybrid is still the right fix and
# recovers it (->62% at WINDOWED_K=40), and BEATS pure-windowed, but (a) full pieces
# need a much larger WINDOWED_K (~40, i.e. ~60 gate alignments => latency), and
# (b) recognition CEILINGS ~62% (a 12-dim-chroma information limit, not a bug). The
# exact (WINDOW_NOTES, WINDOWED_K) is a latency-vs-recall owner tradeoff; defaults
# below stay at the capped-eval win as the documented starting point. Higher
# recognition needs richer window features / a learned shortlist (a bigger call).
SHORTLIST_MODE: str = _env_str("SHORTLIST_MODE", "hybrid")  # "whole_piece" | "windowed" | "hybrid"
WINDOW_NOTES: int = _env_int("WINDOW_NOTES", 400)
WINDOW_HOP: int = _env_int("WINDOW_HOP", 200)
WINDOWED_K: int = _env_int("WINDOWED_K", 8)        # hybrid: # extra pieces from the windowed pass
TOP_K: int = _env_int("TOP_K", 20)
MARGIN_THRESHOLD: float = _env_float("MARGIN_THRESHOLD", 0.13)
ABS_COST_FLOOR: float | None = _env_opt_float("ABS_COST_FLOOR", None)
AMT_SOURCE_EXTRA_MARGIN: float = _env_float("AMT_SOURCE_EXTRA_MARGIN", 0.0)

_AMT_SOURCES = {"giantmidi", "pdmx"}


def _source(pid: str) -> str:
    return pid.split(".")[0]


# ---------------------------------------------------------------------------
# Shortlist index (whole-piece baseline OR windowed)
# ---------------------------------------------------------------------------

class WholePieceShortlist:
    """Baseline: one velocity-weighted chroma vector per catalog piece."""

    def __init__(self, catalog: dict[str, list[Note]]):
        self._chroma = _RunningChromaIndex(catalog)

    def shortlist(self, query_notes: list[Note], k: int) -> list[str]:
        return self._chroma.top_k(chroma_vector(query_notes), k)


class WindowedShortlist:
    """Index each catalog piece by SEVERAL overlapping note-window chroma vectors.

    A mid-piece query whose pitch distribution diverges from the whole-piece
    histogram can still match a local window of the right piece. The shortlist is
    the union of pieces whose ANY window is in the per-window top-(k*oversample),
    ordered by best (max) window cosine, truncated to k.
    """

    def __init__(self, catalog: dict[str, list[Note]], window_notes: int,
                 window_hop: int, oversample: int = 4):
        self._oversample = oversample
        vecs: list[np.ndarray] = []
        owners: list[str] = []
        for pid, notes in catalog.items():
            n = len(notes)
            if n <= window_notes:
                vecs.append(chroma_vector(notes))
                owners.append(pid)
                continue
            start = 0
            while start < n:
                w = notes[start:start + window_notes]
                if len(w) >= max(2, window_notes // 2):
                    vecs.append(chroma_vector(w))
                    owners.append(pid)
                start += window_hop
        self._mat = np.stack(vecs)              # (W, 12), each L2-normed
        self._owners = np.array(owners)

    def shortlist(self, query_notes: list[Note], k: int) -> list[str]:
        q = chroma_vector(query_notes)
        sims = self._mat @ q                    # cosine (all unit-norm)
        # Best window-sim per piece, then top-k pieces.
        best: dict[str, float] = {}
        # argpartition the top window candidates first to bound work.
        top_w = np.argpartition(-sims, min(k * self._oversample, len(sims) - 1))[: k * self._oversample]
        for idx in top_w:
            pid = self._owners[idx]
            s = float(sims[idx])
            if s > best.get(pid, -1.0):
                best[pid] = s
        return [pid for pid, _ in sorted(best.items(), key=lambda kv: -kv[1])[:k]]


class HybridShortlist:
    """ADDITIVE hybrid: whole-piece top-K  UNION  windowed top-WINDOWED_K.

    The whole-piece top-K is returned UNCHANGED (opening recognition is never
    sacrificed -- the failure mode of the naive windowed shortlist, which
    REPLACED the whole-piece set and let spurious short-window cosine hits crowd
    out the true opening candidates). A small number (WINDOWED_K) of extra pieces
    whose best local window matches the query are appended; the elastic-DTW gate
    then re-ranks the union by alignment cost, so a mid-piece query can recover
    its true piece via a window match while a spurious window candidate is either
    rejected on cost or only shrinks the margin (never a wrong lock for free).
    """

    def __init__(self, catalog: dict[str, list[Note]], window_notes: int,
                 window_hop: int, windowed_k: int):
        self._whole = _RunningChromaIndex(catalog)
        self._windowed_k = windowed_k
        vecs: list[np.ndarray] = []
        owners: list[str] = []
        for pid, notes in catalog.items():
            n = len(notes)
            if n <= window_notes:
                continue  # whole-piece vector already covers it (in self._whole)
            start = 0
            while start < n:
                w = notes[start:start + window_notes]
                if len(w) >= max(2, window_notes // 2):
                    vecs.append(chroma_vector(w))
                    owners.append(pid)
                start += window_hop
        self._mat = np.stack(vecs) if vecs else np.zeros((0, 12))
        self._owners = np.array(owners)

    def _windowed_top(self, query_vec: np.ndarray, k: int) -> list[str]:
        if self._mat.shape[0] == 0 or k <= 0:
            return []
        sims = self._mat @ query_vec
        oversample = 8
        n_take = min(k * oversample, len(sims))
        top_w = np.argpartition(-sims, n_take - 1)[:n_take]
        best: dict[str, float] = {}
        for idx in top_w:
            pid = str(self._owners[idx])
            s = float(sims[idx])
            if s > best.get(pid, -1.0):
                best[pid] = s
        return [pid for pid, _ in sorted(best.items(), key=lambda kv: -kv[1])[:k]]

    def shortlist(self, query_notes: list[Note], k: int) -> list[str]:
        q = chroma_vector(query_notes)
        whole = self._whole.top_k(q, k)
        seen = set(whole)
        out = list(whole)
        for pid in self._windowed_top(q, self._windowed_k + len(seen)):
            if pid not in seen:
                out.append(pid)
                seen.add(pid)
            if len(out) >= k + self._windowed_k:
                break
        return out


def build_shortlist_index(catalog: dict[str, list[Note]]):
    if SHORTLIST_MODE == "windowed":
        return WindowedShortlist(catalog, WINDOW_NOTES, WINDOW_HOP)
    if SHORTLIST_MODE == "hybrid":
        return HybridShortlist(catalog, WINDOW_NOTES, WINDOW_HOP, WINDOWED_K)
    return WholePieceShortlist(catalog)


# ---------------------------------------------------------------------------
# Acceptance decision (margin + optional cost floor + source-aware strictness)
# ---------------------------------------------------------------------------

def ranked_costs(query_notes: list[Note], cand_ids: list[str],
                 gate: ElasticGate) -> list[tuple[str, float]]:
    q_pc, q_li = _notes_to_events(query_notes)
    out: list[tuple[str, float]] = []
    for cid in cand_ids:
        c = gate.cost(q_pc, q_li, cid, W_PITCH, W_TIME)
        if c is not None and np.isfinite(c):
            out.append((cid, float(c)))
    out.sort(key=lambda x: x[1])
    return out


def accept(ranked: list[tuple[str, float]]) -> tuple[str | None, float, bool]:
    """Return (best_id, margin, locked) under the current acceptance levers.

    best_id/margin describe the lowest-cost candidate regardless of lock; locked
    is the accept decision (margin gate + optional absolute floor + source-aware
    extra margin for AMT distractors)."""
    if len(ranked) < 2:
        if ranked:
            return ranked[0][0], float("inf"), False
        return None, 0.0, False
    best_id, best_cost = ranked[0]
    margin = ranked[1][1] - best_cost
    required = MARGIN_THRESHOLD
    if _source(best_id) in _AMT_SOURCES:
        required += AMT_SOURCE_EXTRA_MARGIN
    locked = margin >= required
    if locked and ABS_COST_FLOOR is not None:
        locked = best_cost <= ABS_COST_FLOOR
    return best_id, margin, locked


def levers() -> dict:
    """Snapshot of the current lever settings (logged into every eval result)."""
    return {
        "SHORTLIST_MODE": SHORTLIST_MODE,
        "WINDOW_NOTES": WINDOW_NOTES,
        "WINDOW_HOP": WINDOW_HOP,
        "WINDOWED_K": WINDOWED_K,
        "TOP_K": TOP_K,
        "MARGIN_THRESHOLD": MARGIN_THRESHOLD,
        "ABS_COST_FLOOR": ABS_COST_FLOOR,
        "AMT_SOURCE_EXTRA_MARGIN": AMT_SOURCE_EXTRA_MARGIN,
    }
