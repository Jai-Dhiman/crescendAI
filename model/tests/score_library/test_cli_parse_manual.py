"""Test that the parse-manual CLI wiring ingests a tiny manifest end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import mido

from score_library.cli import cmd_parse_manual

PPB = 480
BAR_TICKS = PPB * 4
SIXTEENTH = PPB // 4


def _clean_c_major_bytes() -> bytes:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    events: list[tuple[int, str, int]] = []
    for b in range(3):
        for i, pitch in enumerate(pitches):
            on = b * BAR_TICKS + i * (2 * SIXTEENTH)
            events.append((on, "on", pitch))
            events.append((on + SIXTEENTH, "off", pitch))
    events.sort(key=lambda e: e[0])
    prev = 0
    for abs_tick, kind, pitch in events:
        delta = abs_tick - prev
        prev = abs_tick
        msg_type = "note_on" if kind == "on" else "note_off"
        vel = 80 if kind == "on" else 0
        track.append(mido.Message(msg_type, note=pitch, velocity=vel, time=delta))
    mid = mido.MidiFile(ticks_per_beat=PPB)
    mid.tracks.append(track)
    buf = BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def test_cmd_parse_manual_writes_score(tmp_path: Path, monkeypatch) -> None:
    scores_dir = tmp_path / "scores"
    manifest_path = tmp_path / "manifest.json"
    lock_path = tmp_path / "lock.json"

    clean = _clean_c_major_bytes()
    url = "https://example.org/cmaj.mid"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "slug": "test_slug",
                    "piece_id": "test.cmajor",
                    "composer": "Test",
                    "title": "C Major",
                    "expected_key": "C major",
                    "expected_bars": 3,
                    "license": "PD",
                    "sources": [url],
                }
            ]
        )
    )

    # Redirect Scores.root to the temp dir and inject a fake fetch.
    import score_library.cli as cli_mod
    monkeypatch.setattr(cli_mod.Scores, "root", scores_dir)

    import score_library.manual as manual_mod
    monkeypatch.setattr(manual_mod, "_http_fetch", lambda u: clean)

    args = argparse.Namespace(manifest=str(manifest_path), lock=str(lock_path))
    cmd_parse_manual(args)

    out = scores_dir / "test.cmajor.json"
    assert out.exists()
    lock = json.loads(lock_path.read_text())
    assert lock["test.cmajor"]["sha256"] == hashlib.sha256(clean).hexdigest()
