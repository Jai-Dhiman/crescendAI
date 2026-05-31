import shutil
import time
from pathlib import Path

import partitura
import pytest

from chroma_dtw_eval.gold_truth_builder import GoldMapMissingDataError, build_gold_map


REPO = Path(__file__).resolve().parents[2]
ASAP_ROOT = REPO / "data" / "raw" / "asap"


def _find_asap_pair() -> tuple[Path, Path] | None:
    if not ASAP_ROOT.exists():
        return None
    for midi in ASAP_ROOT.rglob("*.mid"):
        # Look for a sibling MusicXML or .musicxml that parangonar can read as the score.
        for ext in ("xml_score.musicxml", "musicxml", "xml"):
            candidate = midi.with_name(f"{ext}")
            if candidate.exists():
                return midi, candidate
        # Fallback: try first .musicxml in the same folder.
        cands = list(midi.parent.glob("*.musicxml"))
        if cands:
            return midi, cands[0]
    return None


def test_build_gold_map_caches_and_supports_lookup(tmp_path):
    pair = _find_asap_pair()
    if pair is None:
        pytest.skip("no ASAP midi+musicxml pair on disk")
    midi_path, score_path = pair
    cache_root = tmp_path / "gold_cache"
    t0 = time.monotonic()
    gm = build_gold_map(midi_path, score_path, cache_root=cache_root)
    t_first = time.monotonic() - t0
    t1 = time.monotonic()
    gm2 = build_gold_map(midi_path, score_path, cache_root=cache_root)
    t_second = time.monotonic() - t1
    assert t_second * 4 < t_first + 1e-6, f"cache miss: {t_first}s vs {t_second}s"

    frame_a = gm.audio_seconds_to_score_frame(1.0, frame_rate_hz=50.0)
    frame_b = gm2.audio_seconds_to_score_frame(1.0, frame_rate_hz=50.0)
    assert frame_a == frame_b
    assert isinstance(frame_a, int)


def test_build_gold_map_raises_when_inputs_missing(tmp_path):
    with pytest.raises(GoldMapMissingDataError):
        build_gold_map(tmp_path / "nope.mid", tmp_path / "nope.musicxml", cache_root=tmp_path)
