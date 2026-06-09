"""Key-resolution helpers for the ADAPT layer.

Three pure/near-pure helpers:
  parse_key_to_pc   -- key signature string -> pitch class integer 0-11
  transpose_interval -- nearest-octave semitone shift from one pc to another
  load_passage_key  -- resolve key_signature from a piece JSON on disk

All enharmonic normalization and mode stripping live here, hidden from callers.
Default scores_dir is anchored to __file__, not CWD, so it survives `just`
recipe cwd shifts.
"""

import json
from pathlib import Path

# Pitch class lookup. Enharmonic pairs share a value.
# Keys are canonical tonic names (uppercase root + optional accidental).
_PC: dict[str, int] = {
    "C": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11,
}

# Anchored default: keys.py lives at model/src/exercise_corpus/keys.py,
# parents[2] is model/
_DEFAULT_SCORES_DIR = Path(__file__).resolve().parents[2] / "data" / "scores"


def parse_key_to_pc(key_signature: str) -> int:
    """Map a key signature string to a pitch class integer 0-11.

    Strips trailing mode tokens (" major", " minor", "m") so only the tonic
    matters. Raises ValueError on unrecognizable input.

    Examples:
        "C major" -> 0
        "Am" -> 9
        "Eb" -> 3
        "C#m" -> 1
        "F#" -> 6
    """
    s = key_signature.strip()
    # Strip trailing " major" / " minor" (case-insensitive)
    for suffix in (" major", " minor"):
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    # Strip trailing "m" (minor shorthand) -- only if not the entire string
    if s.endswith("m") and len(s) > 1:
        s = s[:-1]
    if s in _PC:
        return _PC[s]
    raise ValueError(f"unparseable key signature: {key_signature!r}")


def transpose_interval(from_pc: int, to_pc: int) -> int:
    """Nearest-octave semitone shift from from_pc to to_pc.

    Returns an integer in [-5, +6]. Tritone (d=6) resolves to +6 by convention.

    Examples:
        (0, 0) -> 0     (same key, no shift)
        (0, 3) -> +3    (C -> Eb)
        (0, 9) -> -3    (C -> A, nearest is down 3, not up 9)
        (0, 6) -> +6    (tritone, +6 by convention)
    """
    d = (to_pc - from_pc) % 12
    if d > 6:
        d -= 12
    return d


def load_passage_key(piece_id: str, scores_dir: Path | None = None) -> str | None:
    """Resolve the key signature string for a piece from its score JSON.

    Args:
        piece_id: identifier used as the JSON filename stem.
        scores_dir: directory containing <piece_id>.json files. Defaults to
            model/data/scores/ anchored to this file's location.

    Returns:
        The `key_signature` string from the JSON, or None if the field is null.

    Raises:
        FileNotFoundError: if <scores_dir>/<piece_id>.json does not exist.
    """
    if scores_dir is None:
        scores_dir = _DEFAULT_SCORES_DIR
    path = Path(scores_dir) / f"{piece_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"score JSON not found for piece_id {piece_id!r}: {path}"
        )
    with open(path) as f:
        data = json.load(f)
    return data.get("key_signature")
