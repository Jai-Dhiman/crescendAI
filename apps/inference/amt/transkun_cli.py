"""Shared Transkun transcription helper.

Deliberately does NOT import `transkun`: it shells out to an isolated env
(`uv run --no-project --with transkun --python 3.11 transkun IN OUT --device cpu`),
so this module is import-safe from BOTH the service env and model/.venv (whose
torch deps conflict with Transkun). Parses the output MIDI with pretty_midi.

Returns the exact dict shapes both surfaces already expect:
  notes:  {"pitch": int, "onset": float, "offset": float, "velocity": int}
  pedals: {"time": float, "value": int}   (CC64 >= 64 -> value 127 "on", else 0)
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import soundfile as sf

SAMPLE_RATE = 16000
_TRANSKUN_TIMEOUT_S = 900


class TranskunError(RuntimeError):
    """Raised when Transkun transcription fails. Never return empty notes on error."""


def midi_to_notes_and_pedals(
    midi_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a Transkun-produced MIDI file into note and pedal-event lists."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    notes: list[dict[str, Any]] = []
    pedals: list[dict[str, Any]] = []  # CC64 parsing added in T2
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append({
                "pitch": int(n.pitch),
                "onset": round(float(n.start), 4),
                "offset": round(float(n.end), 4),
                "velocity": int(n.velocity),
            })
        for cc in inst.control_changes:
            if int(cc.number) != 64:
                continue
            pedals.append({
                "time": round(float(cc.time), 4),
                "value": 127 if int(cc.value) >= 64 else 0,
            })

    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    pedals.sort(key=lambda e: e["time"])
    return notes, pedals


def _run_transkun(in_wav: Path, out_mid: Path) -> None:
    """Shell out to Transkun in an isolated env. Raise TranskunError on any failure."""
    # setuptools is required: transkun's transcribe.py does `import pkg_resources`, which lives
    # in setuptools. uv's isolated env does not include it unless a dep pulls it in, so without
    # this the CLI dies with ModuleNotFoundError: No module named 'pkg_resources' (intermittent,
    # depending on uv-cache resolution). Pinning it explicitly makes the env deterministic.
    #
    # The <81 bound is NOT cosmetic (#166): setuptools REMOVED
    # pkg_resources in 81, so an unpinned `--with setuptools` resolves to
    # a version without it on any machine with a cold uv cache, and
    # Transkun dies exactly as it did before this arg existed. Found when
    # the MIREX container build -- a guaranteed-cold cache -- hit it; a
    # warm local cache still holding 80.9.0 hides it completely.
    cmd = [
        "uv", "run", "--no-project", "--with", "transkun", "--with", "setuptools<81",
        "--python", "3.11",
        "transkun", str(in_wav), str(out_mid), "--device", "cpu",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=_TRANSKUN_TIMEOUT_S
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise TranskunError(f"transkun subprocess failed to run: {exc}") from exc
    if proc.returncode != 0:
        raise TranskunError(
            f"transkun exited {proc.returncode}: "
            f"{proc.stderr.decode('utf-8', errors='replace')[-2000:]}"
        )
    if not out_mid.exists():
        raise TranskunError(
            f"transkun exited 0 but produced no MIDI at {out_mid}"
        )


def transcribe_wav(
    wav_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transcribe a WAV file to (notes, pedals) via Transkun. Raise TranskunError on failure."""
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise TranskunError(f"input WAV does not exist: {wav_path}")
    with tempfile.TemporaryDirectory() as td:
        out_mid = Path(td) / "out.mid"
        _run_transkun(wav_path, out_mid)
        return midi_to_notes_and_pedals(out_mid)


def transcribe_pcm(
    pcm_16k: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transcribe a 16kHz mono float32 PCM array to (notes, pedals) via Transkun."""
    pcm = np.ascontiguousarray(np.asarray(pcm_16k, dtype=np.float32))
    if pcm.size == 0:
        raise TranskunError("transcribe_pcm received empty PCM")
    with tempfile.TemporaryDirectory() as td:
        in_wav = Path(td) / "in.wav"
        sf.write(str(in_wav), pcm, SAMPLE_RATE, format="WAV", subtype="FLOAT")
        return transcribe_wav(in_wav)
