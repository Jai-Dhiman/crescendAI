# model/tests/follower_eval/test_build_corpus.py
"""Unit tests for the corpus builder's OWN logic (issue #133 Slice 2): approved
filtering, idempotent skip, loud empty/failure handling. Network (yt-dlp) and
Transkun are stubbed -- the real download+transcribe is exercised by the smoke
run, not the unit suite."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from follower_eval import build_corpus as bc


def _write_candidates(root: Path, piece: str, recs: list[dict]) -> None:
    d = root / piece
    d.mkdir(parents=True)
    (d / "candidates.yaml").write_text(yaml.safe_dump({"piece": piece, "recordings": recs}))


def test_approved_videos_filters_unapproved(tmp_path: Path):
    _write_candidates(tmp_path, "fur_elise", [
        {"video_id": "a", "approved": True},
        {"video_id": "b", "approved": False},
        {"video_id": "c"},  # unreviewed -> not approved
    ])
    _write_candidates(tmp_path, "bach_invention_1", [{"video_id": "d", "approved": True}])
    got = bc.approved_videos(tmp_path)
    assert set(got) == {"fur_elise", "bach_invention_1"}
    assert [r["video_id"] for r in got["fur_elise"]] == ["a"]


def test_approved_videos_pieces_filter(tmp_path: Path):
    _write_candidates(tmp_path, "fur_elise", [{"video_id": "a", "approved": True}])
    _write_candidates(tmp_path, "bach_invention_1", [{"video_id": "d", "approved": True}])
    assert set(bc.approved_videos(tmp_path, pieces=["fur_elise"])) == {"fur_elise"}


def test_build_one_skips_existing_bundle(tmp_path: Path):
    bundle = tmp_path / "b" / "vid.json"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("{}")
    def _fail_transcribe(_):  # must NOT be called on skip
        raise AssertionError("transcribe should not run when bundle exists")
    out = bc.build_one("p", {"video_id": "vid"}, tmp_path / "audio", bundle, _fail_transcribe)
    assert out.status == "skip"


def test_build_one_empty_notes_is_loud(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bc, "acquire_audio", lambda vid, d: tmp_path / f"{vid}.wav")
    out = bc.build_one("p", {"video_id": "v"}, tmp_path / "audio",
                       tmp_path / "b" / "v.json", lambda _wav: ([], []))
    assert out.status == "empty"
    assert not (tmp_path / "b" / "v.json").exists()  # never wrote an empty bundle


def test_build_one_ok_writes_bundle(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bc, "acquire_audio", lambda vid, d: tmp_path / f"{vid}.wav")
    notes = [{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 70}]
    bundle_path = tmp_path / "p" / "v.json"
    out = bc.build_one("p", {"video_id": "v", "title": "T"}, tmp_path / "audio",
                       bundle_path, lambda _wav: (notes, []))
    assert out.status == "ok" and out.n_notes == 1
    body = json.loads(bundle_path.read_text())
    assert body["piece_id"] == "p" and body["notes"] == notes
    assert body["substrate_versions"]["transcriber"] == "transkun"


def test_build_one_download_failure_recorded(tmp_path: Path, monkeypatch):
    def _boom(vid, d):
        raise bc.AcquireError("yt-dlp 403")
    monkeypatch.setattr(bc, "acquire_audio", _boom)
    out = bc.build_one("p", {"video_id": "v"}, tmp_path / "audio",
                       tmp_path / "b" / "v.json", lambda _wav: ([], []))
    assert out.status == "download_fail" and "403" in out.error
