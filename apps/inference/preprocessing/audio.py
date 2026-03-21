"""Audio download and preprocessing for D9c inference."""

import tempfile
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import requests

# Default sample rate for MERT/MuQ (hardcoded to avoid import issues)
TARGET_SR = 24000


class AudioDownloadError(Exception):
    """Raised when audio download fails."""

    pass


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""

    pass


def download_and_preprocess_audio(
    audio_url: str,
    target_sr: int = TARGET_SR,
    max_duration: int = 300,
    timeout: int = 60,
) -> Tuple[np.ndarray, float]:
    """Download audio from URL and preprocess for MERT/MuQ.

    Args:
        audio_url: URL to download audio from
        target_sr: Target sample rate (24kHz for MERT/MuQ)
        max_duration: Maximum audio duration in seconds
        timeout: Download timeout in seconds

    Returns:
        Tuple of (audio_array, duration_seconds)

    Raises:
        AudioDownloadError: If download fails
        AudioProcessingError: If audio processing fails
    """
    try:
        response = requests.get(audio_url, timeout=timeout, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        raise AudioDownloadError(f"Failed to download audio: {e}")

    # Determine file extension from content-type or URL
    content_type = response.headers.get("content-type", "")
    if "mpeg" in content_type or audio_url.endswith(".mp3"):
        suffix = ".mp3"
    elif "wav" in content_type or audio_url.endswith(".wav"):
        suffix = ".wav"
    elif "flac" in content_type or audio_url.endswith(".flac"):
        suffix = ".flac"
    else:
        suffix = ".mp3"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
        temp_path = Path(f.name)

    try:
        audio, sr = librosa.load(temp_path, sr=target_sr, mono=True)
        duration = len(audio) / sr

        if duration > max_duration:
            raise AudioProcessingError(
                f"Audio too long: {duration:.1f}s > {max_duration}s limit"
            )

        if duration < 1.0:
            raise AudioProcessingError(
                f"Audio too short: {duration:.1f}s < 1.0s minimum"
            )

        return audio, duration

    except AudioProcessingError:
        raise
    except Exception as e:
        raise AudioProcessingError(f"Failed to process audio: {e}")

    finally:
        temp_path.unlink(missing_ok=True)


def load_audio_from_file(
    audio_path: Path,
    target_sr: int = TARGET_SR,
) -> Tuple[np.ndarray, float]:
    """Load audio from local file."""
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    duration = len(audio) / sr
    return audio, duration


def _decode_with_torchaudio(audio_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode audio bytes (WebM/Opus, etc.) to numpy via torchaudio.

    torchaudio uses ffmpeg as a backend and handles WebM/Opus containers
    that soundfile/libsndfile cannot decode.
    """
    import io

    import torchaudio

    buf = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(buf)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    return waveform.squeeze(0).numpy()


def preprocess_audio_from_bytes(
    audio_bytes: bytes,
    target_sr: int = TARGET_SR,
    max_duration: int = 300,
) -> Tuple[np.ndarray, float]:
    """Preprocess audio from raw bytes (e.g., base64 decoded).

    Tries librosa/soundfile first (WAV, FLAC, etc.).
    Falls back to torchaudio for formats soundfile can't handle (WebM/Opus).
    """
    import io

    try:
        try:
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
        except Exception:
            # soundfile can't decode this format (e.g., WebM/Opus) -- use torchaudio
            audio = _decode_with_torchaudio(audio_bytes, target_sr)

        duration = len(audio) / target_sr

        if duration > max_duration:
            raise AudioProcessingError(
                f"Audio too long: {duration:.1f}s > {max_duration}s limit"
            )

        if duration < 1.0:
            raise AudioProcessingError(
                f"Audio too short: {duration:.1f}s < 1.0s minimum"
            )

        return audio, duration

    except AudioProcessingError:
        raise
    except Exception as e:
        raise AudioProcessingError(f"Failed to process audio bytes: {e}")
