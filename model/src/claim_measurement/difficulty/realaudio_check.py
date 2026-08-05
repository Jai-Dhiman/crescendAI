"""#138 Phase 1 real-audio second gate: 709 of 900 eval pieces have local
WAVs (re-fetched this session; see design spec). Resumable transcription
(`main`) plus MIDI drift (`midi_drift`) and per-fold audio scoring
(`score_audio_subset`) -- see this module's own docstring in the plan for the
deliberate scope split between what is CLI-wired here vs. what is a runbook
snippet over these tested primitives.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds, paired_boot, tau_c

N_FOLDS, SEED = 5, 2026
ALPHAS = np.logspace(-1, 5, 25)


def midi_drift(reference_notes: list, candidate_notes: list, onset_tolerance: float) -> dict:
    """note-count delta (candidate - reference) and onset F1: a candidate
    note matches a reference note when they share pitch and onsets differ by
    <= onset_tolerance seconds. Matching is greedy nearest-onset-first, and
    each reference/candidate note is used at most once."""
    pairs = []
    for ci, c in enumerate(candidate_notes):
        for ri, r in enumerate(reference_notes):
            if r["pitch"] != c["pitch"]:
                continue
            dt = abs(r["onset"] - c["onset"])
            if dt <= onset_tolerance:
                pairs.append((dt, ci, ri))
    pairs.sort(key=lambda p: p[0])

    matched_ref, matched_cand, tp = set(), set(), 0
    for _dt, ci, ri in pairs:
        if ci in matched_cand or ri in matched_ref:
            continue
        matched_cand.add(ci)
        matched_ref.add(ri)
        tp += 1

    precision = tp / len(candidate_notes) if candidate_notes else 0.0
    recall = tp / len(reference_notes) if reference_notes else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"note_count_delta": len(candidate_notes) - len(reference_notes), "onset_f1": f1}


if __name__ == "__main__":
    sys.exit(0)  # placeholder exit; main() is added in Task 17
