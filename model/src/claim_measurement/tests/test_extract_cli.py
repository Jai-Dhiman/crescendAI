"""Unit tests for the bundle-extraction runner's clip resolution (no live AMT server)."""
from __future__ import annotations

from pathlib import Path

from claim_measurement import extract_cli


def _make_pseudo_layout(tmp_path: Path) -> tuple[Path, Path]:
    pseudo_root = tmp_path / "practice_eval_pseudo"
    practice_eval_root = tmp_path / "practice_eval"
    # Two pieces; first has a score mapping, second does not. Each has clip subdirs.
    for piece, vids in {"bach_invention_1": ["vidA", "vidB"], "chopin_x": ["vidC"]}.items():
        for v in vids:
            (pseudo_root / piece / v).mkdir(parents=True)
            audio_dir = practice_eval_root / piece / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            (audio_dir / f"{v}.wav").write_bytes(b"stub")
    return pseudo_root, practice_eval_root


def test_resolve_clips_maps_audio_and_score(tmp_path) -> None:
    pseudo_root, practice_eval_root = _make_pseudo_layout(tmp_path)
    score = tmp_path / "bach.inventions.1.json"
    score.write_text("{}")
    score_by_piece = {"bach_invention_1": score}

    clips = extract_cli._resolve_clips(pseudo_root, practice_eval_root, score_by_piece)

    by_key = {(c.piece_id, c.video_id): c for c in clips}
    assert set(by_key) == {
        ("bach_invention_1", "vidA"),
        ("bach_invention_1", "vidB"),
        ("chopin_x", "vidC"),
    }
    a = by_key[("bach_invention_1", "vidA")]
    assert a.audio_path == practice_eval_root / "bach_invention_1" / "audio" / "vidA.wav"
    assert a.score_path == score
    # Unmapped piece -> score_path None (reported as no_score, not a silent skip).
    assert by_key[("chopin_x", "vidC")].score_path is None


def test_run_records_no_score_and_raises_on_missing_audio(tmp_path, monkeypatch) -> None:
    pseudo_root, practice_eval_root = _make_pseudo_layout(tmp_path)
    score = tmp_path / "bach.inventions.1.json"
    score.write_text("{}")
    score_by_piece = {"bach_invention_1": score}
    clips = extract_cli._resolve_clips(pseudo_root, practice_eval_root, score_by_piece)

    # Stub extract_bundle so no AMT server is needed; write a bundle with pedal_events.
    def fake_extract(piece_id, video_id, *, audio_path, score_path, cache_root,
                     bundle_root, amt_url, force):
        out = bundle_root / piece_id / f"{video_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"pedal_events": [{"time": 1.0, "value": 127}]}')
        return out

    monkeypatch.setattr(extract_cli, "extract_bundle", fake_extract)

    results = extract_cli.run(clips, bundle_root=tmp_path / "bundles",
                              amt_url="http://x", force=True)
    by_key = {(r["piece"], r["video"]): r for r in results}
    assert by_key[("bach_invention_1", "vidA")]["status"] == "ok"
    assert by_key[("bach_invention_1", "vidA")]["n_pedal_events"] == 1
    assert by_key[("chopin_x", "vidC")]["status"] == "no_score"

    # Missing audio must fail loudly, not skip silently.
    (practice_eval_root / "bach_invention_1" / "audio" / "vidA.wav").unlink()
    import pytest
    with pytest.raises(extract_cli.BundleExtractionError):
        extract_cli.run(clips, bundle_root=tmp_path / "bundles2",
                        amt_url="http://x", force=True)
