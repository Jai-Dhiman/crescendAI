"""Shared generated fixtures for audio_teacher tests.

Everything is generated under tmp_path -- no binary fixture files are
committed to the repo.
"""
from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def wav_factory(tmp_path):
    """Write a PCM-16 silence WAV under tmp_path and return its path."""

    def _write(
        rel: str,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        seconds: float = 1.0,
    ) -> Path:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        nframes = int(sample_rate * seconds)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(
                struct.pack(f"<{nframes * channels}h", *([0] * (nframes * channels)))
            )
        return path

    return _write


@pytest.fixture
def manifest_factory(tmp_path, wav_factory):
    """Build a loadable probe manifest YAML rooted at tmp_path.

    Each entry in `pairs` is a dict with keys id / axis / population /
    degraded (all optional except id); the two clip WAVs are generated
    automatically. Returns the manifest path. Load with
    load_manifest(path, repo_root=tmp_path).
    """

    def _build(pairs: list[dict], *, sample_rate: int = 16000) -> Path:
        entries = []
        for p in pairs:
            a = wav_factory(f"clips/{p['id']}_a.wav", sample_rate=sample_rate)
            b = wav_factory(f"clips/{p['id']}_b.wav", sample_rate=sample_rate)
            entries.append(
                {
                    "id": p["id"],
                    "axis": p.get("axis", "pedaling"),
                    "population": p.get("population", "real"),
                    "clip_a": str(a.relative_to(tmp_path)),
                    "clip_b": str(b.relative_to(tmp_path)),
                    "degraded": p.get("degraded", "a"),
                    "description": "test contrast",
                }
            )
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            yaml.safe_dump(
                {"schema_version": 1, "sample_rate": sample_rate, "pairs": entries},
                sort_keys=False,
            )
        )
        return manifest_path

    return _build
