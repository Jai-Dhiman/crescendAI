"""Three behaviors locked in one suite:
1. round-trip writer -> loader equality + monotone interpolation
2. missing cache file raises PseudoTruthMissingError with usable path
3. any-of-four key-field mismatch raises PseudoTruthMismatchError
"""
from pathlib import Path

import numpy as np
import pytest

from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthMismatchError,
    PseudoTruthMissingError,
    PseudoTruthPayload,
    cache_path,
    load_pseudo_truth,
    write_pseudo_truth,
)


def _payload() -> PseudoTruthPayload:
    return PseudoTruthPayload(
        perf_audio_sec=np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float64),
        score_audio_sec=np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64),
        measure_table=[{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}],
        audio_sha256="a" * 16,
        amt_checkpoint_hash="b" * 16,
        score_sha256="c" * 16,
        parangonar_version="3.3.2",
        regen_source="local:test",
    )


def test_roundtrip_and_interpolation(tmp_path: Path) -> None:
    written = write_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="VID000",
        payload=_payload(), cache_root=tmp_path,
    )
    assert written.exists()
    assert written == cache_path(tmp_path, "bach_prelude_c_wtc1", "VID000")

    loaded = load_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="VID000",
        audio_sha256="a" * 16, amt_checkpoint_hash="b" * 16,
        score_sha256="c" * 16, parangonar_version="3.3.2",
        cache_root=tmp_path,
    )
    np.testing.assert_array_equal(loaded.perf_audio_sec, _payload().perf_audio_sec)
    np.testing.assert_array_equal(loaded.score_audio_sec, _payload().score_audio_sec)
    assert loaded.measure_table == _payload().measure_table
    # Monotone linear interpolation between anchors.
    assert loaded.audio_sec_to_score_sec(0.5) == pytest.approx(0.25)
    assert loaded.audio_sec_to_score_sec(3.0) == pytest.approx(1.5)
    # Inverse is consistent.
    assert loaded.score_sec_to_audio_sec(1.0) == pytest.approx(2.0)


def test_missing_file_raises_with_path(tmp_path: Path) -> None:
    with pytest.raises(PseudoTruthMissingError) as exc:
        load_pseudo_truth(
            piece_id="nope", video_id="zzz",
            audio_sha256="x" * 16, amt_checkpoint_hash="y" * 16,
            score_sha256="w" * 16, parangonar_version="3.3.2",
            cache_root=tmp_path,
        )
    msg = str(exc.value)
    assert "nope" in msg and "zzz" in msg
    assert str(tmp_path) in msg


@pytest.mark.parametrize("field,bad", [
    ("audio_sha256", "z" * 16),
    ("amt_checkpoint_hash", "z" * 16),
    ("score_sha256", "z" * 16),
    ("parangonar_version", "9.9.9"),
])
def test_key_mismatch_raises(tmp_path: Path, field: str, bad: str) -> None:
    write_pseudo_truth("p", "v", _payload(), tmp_path)
    kwargs = {
        "audio_sha256": "a" * 16,
        "amt_checkpoint_hash": "b" * 16,
        "score_sha256": "c" * 16,
        "parangonar_version": "3.3.2",
    }
    kwargs[field] = bad
    with pytest.raises(PseudoTruthMismatchError) as exc:
        load_pseudo_truth("p", "v", cache_root=tmp_path, **kwargs)
    assert field in str(exc.value)
