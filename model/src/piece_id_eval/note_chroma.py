# model/src/piece_id_eval/note_chroma.py
"""Note-derived chroma features (key-dependent; no OTI/transposition invariance).

C1 (NoteChromaMatcher) uses chroma_vector for global bag-of-notes chroma.
C4 (ChromaSeqDtwMatcher) uses chroma_sequence for frame-level chroma.

Both functions are key-dependent: pitch-class 0 = C, no cyclic normalisation.
"""
from __future__ import annotations

import numpy as np

from piece_id_eval.notes import Note


def chroma_vector(notes: list[Note]) -> np.ndarray:
    """Aggregate all notes into a single 12-bin chroma vector, L2-normalised.

    Each note contributes its velocity (float) to its pitch-class bin.
    The result is the unit-norm version.

    Args:
        notes: non-empty list of Note.

    Returns:
        np.ndarray of shape (12,), float64, L2-normalised.

    Raises:
        ValueError: if notes is empty.
    """
    if not notes:
        raise ValueError("chroma_vector requires at least one note")
    cv = np.zeros(12, dtype=np.float64)
    for n in notes:
        cv[n.pitch % 12] += float(n.velocity)
    norm = np.linalg.norm(cv)
    if norm > 0:
        cv /= norm
    return cv


def chroma_sequence(notes: list[Note], frame_seconds: float) -> np.ndarray:
    """Compute a frame-level chroma matrix from notes.

    Time axis spans [notes[0].onset, notes[-1].offset) quantised to frame_seconds.
    Each note's velocity is added to all frames it overlaps (onset <= frame_start < offset).
    Each frame is L2-normalised independently; silent frames remain zero.

    Args:
        notes: sorted list of Note (ascending onset). May be empty.
        frame_seconds: frame hop and window length in seconds.

    Returns:
        np.ndarray of shape (12, T) where T = ceil(duration / frame_seconds).
        Returns shape (12, 0) for empty notes.
    """
    if not notes:
        return np.zeros((12, 0), dtype=np.float64)

    t_start = notes[0].onset
    t_end = max(n.offset for n in notes)
    duration = t_end - t_start
    n_frames = max(1, int(np.ceil(duration / frame_seconds)))

    cs = np.zeros((12, n_frames), dtype=np.float64)
    for note in notes:
        pc = note.pitch % 12
        f_start = int((note.onset - t_start) / frame_seconds)
        f_end = int(np.ceil((note.offset - t_start) / frame_seconds))
        f_start = max(0, f_start)
        f_end = min(n_frames, f_end)
        cs[pc, f_start:f_end] += float(note.velocity)

    # L2-normalise each frame independently
    norms = np.linalg.norm(cs, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    cs /= norms
    return cs
