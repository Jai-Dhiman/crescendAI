"""Tests for the ranked-source manual ingestion driver."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import mido
import pytest

from score_library.manual import (
    IngestReport,
    SourceResolutionError,
    ingest_manifest,
)

PPB = 480
BAR_TICKS = PPB * 4
SIXTEENTH = PPB // 4


def _scale_track(pitches: list[int], n_bars: int, offset: int = 0) -> mido.MidiTrack:
    """Build a track: each bar plays `pitches` on every other 16th (8 notes/bar).

    `offset` shifts EVERY onset by a FIXED tick amount (not alternating, not
    random). A +60-tick offset at 480 tpq / 120-tick sixteenth puts every note
    exactly half a sixteenth off-grid (0.5 sixteenths), the deterministic maximum
    the median-deviation metric can reach.
    """
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    # Build absolute-tick (note_on, note_off) events, then delta-encode.
    events: list[tuple[int, str, int]] = []
    for b in range(n_bars):
        for i, pitch in enumerate(pitches):
            base = b * BAR_TICKS + i * (2 * SIXTEENTH)
            on = base + offset
            off = on + SIXTEENTH
            events.append((on, "on", pitch))
            events.append((off, "off", pitch))
    events.sort(key=lambda e: e[0])
    prev = 0
    for abs_tick, kind, pitch in events:
        delta = abs_tick - prev
        prev = abs_tick
        if kind == "on":
            track.append(mido.Message("note_on", note=pitch, velocity=80, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
    return track


def _midi_bytes(track: mido.MidiTrack) -> bytes:
    mid = mido.MidiFile(ticks_per_beat=PPB)
    mid.tracks.append(track)
    buf = BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def build_clean_c_major_bytes() -> bytes:
    """>= 20 notes on a clean 4/4 16th grid in C major, several bars, monotonic."""
    return _midi_bytes(_scale_track([60, 62, 64, 65, 67, 69, 71, 72], n_bars=3))


def build_performance_timed_bytes() -> bytes:
    """Same C-major notes but EVERY onset shifted a fixed +60 ticks (half a 16th).

    Deterministic median grid deviation = 0.5 sixteenths > 0.4 -> quantization fails.
    """
    return _midi_bytes(_scale_track([60, 62, 64, 65, 67, 69, 71, 72], n_bars=3, offset=60))


def build_transposed_bytes() -> bytes:
    """Clean grid but transposed to F# major (key-agreement vs C major fails)."""
    return _midi_bytes(_scale_track([66, 68, 70, 71, 73, 75, 77, 78], n_bars=3))


def _make_fetch(mapping: dict[str, bytes]):
    def fetch(url: str) -> bytes:
        if url not in mapping:
            raise KeyError(f"no fixture for {url}")
        return mapping[url]
    return fetch


def _write_manifest(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries))


class TestIngestHappyPath:
    def test_clean_source_writes_json_and_lockfile(self, tmp_path: Path) -> None:
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        manifest_path = tmp_path / "manifest.json"
        lock_path = tmp_path / "lock.json"

        clean = build_clean_c_major_bytes()
        url = "https://example.org/cmaj.mid"
        _write_manifest(
            manifest_path,
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
            ],
        )

        report = ingest_manifest(
            manifest_path, scores_dir, lock_path, fetch_fn=_make_fetch({url: clean})
        )

        assert isinstance(report, IngestReport)
        # Score JSON written.
        out = scores_dir / "test.cmajor.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["piece_id"] == "test.cmajor"
        # Lockfile written with correct sha256.
        lock = json.loads(lock_path.read_text())
        assert lock["test.cmajor"]["resolved_url"] == url
        assert lock["test.cmajor"]["sha256"] == hashlib.sha256(clean).hexdigest()


class TestIngestHashMismatch:
    def test_pinned_wrong_sha_rejects_only_candidate_then_halts(self, tmp_path: Path) -> None:
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        manifest_path = tmp_path / "manifest.json"
        lock_path = tmp_path / "lock.json"

        clean = build_clean_c_major_bytes()
        url = "https://example.org/cmaj.mid"
        # Pre-seed the lockfile pinning this piece to a DIFFERENT sha.
        lock_path.write_text(json.dumps({"test.cmajor": {"resolved_url": url, "sha256": "deadbeef"}}))
        _write_manifest(
            manifest_path,
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
            ],
        )

        with pytest.raises(SourceResolutionError) as exc:
            ingest_manifest(
                manifest_path, scores_dir, lock_path, fetch_fn=_make_fetch({url: clean})
            )
        assert "hash_mismatch" in str(exc.value)
        # The only candidate was rejected by the pin, so no JSON was written.
        assert not (scores_dir / "test.cmajor.json").exists()


class TestIngestRankedFallback:
    def test_first_fails_second_wins(self, tmp_path: Path) -> None:
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        manifest_path = tmp_path / "manifest.json"
        lock_path = tmp_path / "lock.json"

        bad = build_performance_timed_bytes()
        good = build_clean_c_major_bytes()
        url_bad = "https://example.org/perf.mid"
        url_good = "https://example.org/clean.mid"
        _write_manifest(
            manifest_path,
            [
                {
                    "slug": "test_slug",
                    "piece_id": "test.cmajor",
                    "composer": "Test",
                    "title": "C Major",
                    "expected_key": "C major",
                    "expected_bars": 3,
                    "license": "PD",
                    "sources": [url_bad, url_good],
                }
            ],
        )

        report = ingest_manifest(
            manifest_path,
            scores_dir,
            lock_path,
            fetch_fn=_make_fetch({url_bad: bad, url_good: good}),
        )

        assert (scores_dir / "test.cmajor.json").exists()
        lock = json.loads(lock_path.read_text())
        # Second (clean) source won; lockfile records its URL.
        assert lock["test.cmajor"]["resolved_url"] == url_good
        assert report.resolved["test.cmajor"]["resolved_url"] == url_good


class TestIngestAllFailHalt:
    def test_all_sources_fail_raises_with_table(self, tmp_path: Path) -> None:
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        manifest_path = tmp_path / "manifest.json"
        lock_path = tmp_path / "lock.json"

        perf = build_performance_timed_bytes()
        transposed = build_transposed_bytes()
        url_a = "https://example.org/perf.mid"
        url_b = "https://example.org/transposed.mid"
        _write_manifest(
            manifest_path,
            [
                {
                    "slug": "test_slug",
                    "piece_id": "test.cmajor",
                    "composer": "Test",
                    "title": "C Major",
                    "expected_key": "C major",
                    "expected_bars": 3,
                    "license": "PD",
                    "sources": [url_a, url_b],
                }
            ],
        )

        with pytest.raises(SourceResolutionError) as exc:
            ingest_manifest(
                manifest_path,
                scores_dir,
                lock_path,
                fetch_fn=_make_fetch({url_a: perf, url_b: transposed}),
            )
        msg = str(exc.value)
        assert "test.cmajor" in msg
        assert url_a in msg and url_b in msg
        # No JSON written for the unresolved piece.
        assert not (scores_dir / "test.cmajor.json").exists()


class TestLocalFileSource:
    def test_local_file_source_resolves(self, tmp_path: Path) -> None:
        """A non-http(s) source is resolved as a path relative to the manifest file."""
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        lock_path = tmp_path / "lock.json"

        # Write a valid MIDI to tmp_path/manual_midis/x.mid
        midi_dir = tmp_path / "manual_midis"
        midi_dir.mkdir()
        midi_bytes = build_clean_c_major_bytes()
        (midi_dir / "x.mid").write_bytes(midi_bytes)

        # Manifest lives at tmp_path/m.json; source is a relative path
        manifest_path = tmp_path / "m.json"
        manifest_path.write_text(
            json.dumps(
                [
                    {
                        "slug": "test_local",
                        "piece_id": "test.cmajor",
                        "composer": "Test",
                        "title": "C Major",
                        "expected_key": "C major",
                        "expected_bars": 3,
                        "license": "PD",
                        "sources": ["manual_midis/x.mid"],
                    }
                ]
            )
        )

        # NO fetch_fn override: exercises the real local-read path
        ingest_manifest(manifest_path, scores_dir, lock_path)

        # Score JSON lands in scores_dir
        assert (scores_dir / "test.cmajor.json").exists()
        data = json.loads((scores_dir / "test.cmajor.json").read_text())
        assert data["piece_id"] == "test.cmajor"

        # Lockfile records the relative source string (not an absolute path) + sha256
        lock = json.loads(lock_path.read_text())
        assert lock["test.cmajor"]["resolved_url"] == "manual_midis/x.mid"
        assert lock["test.cmajor"]["sha256"] == hashlib.sha256(midi_bytes).hexdigest()

    def test_local_source_path_traversal_rejected(self, tmp_path: Path) -> None:
        """A manifest source that escapes manifest_dir via ../ raises SourceResolutionError."""
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        lock_path = tmp_path / "manifest_subdir" / "lock.json"

        # Manifest lives in a subdirectory; source tries to escape with ../
        manifest_subdir = tmp_path / "manifest_subdir"
        manifest_subdir.mkdir()
        manifest_path = manifest_subdir / "m.json"
        manifest_path.write_text(
            json.dumps(
                [
                    {
                        "slug": "test_escape",
                        "piece_id": "test.escape",
                        "composer": "Test",
                        "title": "Escape",
                        "expected_key": "C major",
                        "expected_bars": 3,
                        "license": "PD",
                        "sources": ["../escape.mid"],
                    }
                ]
            )
        )

        with pytest.raises(SourceResolutionError) as exc:
            ingest_manifest(manifest_path, scores_dir, lock_path)
        assert "path traversal" in str(exc.value).lower()
