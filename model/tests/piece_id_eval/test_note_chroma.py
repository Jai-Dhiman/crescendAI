# model/tests/piece_id_eval/test_note_chroma.py
"""Verify chroma_vector and chroma_sequence through their public interfaces."""
from __future__ import annotations

import numpy as np
import pytest

from piece_id_eval.note_chroma import chroma_sequence, chroma_vector
from piece_id_eval.notes import Note


def _c_notes(n: int = 8) -> list[Note]:
    """n notes all on C (pitch 60, pc=0)."""
    return [Note(onset=i * 0.5, offset=i * 0.5 + 0.4, pitch=60, velocity=80) for i in range(n)]


def _mixed_notes() -> list[Note]:
    """Notes covering pitch classes 0 (C), 4 (E), 7 (G)."""
    return [
        Note(onset=0.0, offset=0.4, pitch=60, velocity=80),  # C
        Note(onset=0.5, offset=0.9, pitch=64, velocity=80),  # E
        Note(onset=1.0, offset=1.4, pitch=67, velocity=80),  # G
    ]


def test_chroma_vector_shape() -> None:
    cv = chroma_vector(_c_notes())
    assert cv.shape == (12,), f"expected (12,), got {cv.shape}"


def test_chroma_vector_l2_normalized() -> None:
    cv = chroma_vector(_c_notes())
    norm = float(np.linalg.norm(cv))
    assert abs(norm - 1.0) < 1e-6, f"expected unit vector, norm={norm}"


def test_chroma_vector_dominant_bin_for_c_notes() -> None:
    cv = chroma_vector(_c_notes())
    # C notes -> pitch class 0 should be dominant
    assert np.argmax(cv) == 0, f"expected bin 0 dominant, got bin {np.argmax(cv)}"


def test_chroma_vector_mixed_notes_has_three_nonzero_bins() -> None:
    cv = chroma_vector(_mixed_notes())
    nonzero = int(np.sum(cv > 0))
    assert nonzero == 3, f"expected 3 nonzero bins, got {nonzero}"


def test_chroma_sequence_shape() -> None:
    notes = [Note(onset=i * 0.25, offset=i * 0.25 + 0.2, pitch=60 + (i % 12), velocity=80) for i in range(40)]
    cs = chroma_sequence(notes, frame_seconds=0.5)
    assert cs.shape[0] == 12, f"expected 12 rows, got {cs.shape[0]}"
    assert cs.shape[1] > 0, "expected at least 1 frame"


def test_chroma_sequence_each_frame_normalized() -> None:
    notes = [Note(onset=i * 0.25, offset=i * 0.25 + 0.2, pitch=60, velocity=80) for i in range(40)]
    cs = chroma_sequence(notes, frame_seconds=0.5)
    norms = np.linalg.norm(cs, axis=0)
    # Every non-zero frame should be unit normalized
    for t, norm in enumerate(norms):
        if norm > 0:
            assert abs(norm - 1.0) < 1e-5, f"frame {t} norm={norm}"


def test_chroma_vector_empty_notes_raises() -> None:
    with pytest.raises(ValueError):
        chroma_vector([])
