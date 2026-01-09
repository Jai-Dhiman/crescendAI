"""Audio download and preprocessing."""

import tempfile
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import requests

from ..constants import MERT_CONFIG


class AudioDownloadError(Exception):
    """Raised when audio download fails."""

    pass


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""

    pass


def download_and_preprocess_audio(
    audio_url: str,
    target_sr: int = MERT_CONFIG["target_sr"],
    max_duration: int = 300,
    timeout: int = 60,
) -> Tuple[np.ndarray, float]:
    """Download audio from URL and preprocess for MERT.

    Args:
        audio_url: URL to download audio from
        target_sr: Target sample rate (24kHz for MERT)
        max_duration: Maximum audio duration in seconds
        timeout: Download timeout in seconds

    Returns:
        Tuple of (audio_array, duration_seconds)

    Raises:
        AudioDownloadError: If download fails
        AudioProcessingError: If audio processing fails
    """
    # Download audio to temp file
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
        suffix = ".mp3"  # Default to mp3

    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
        temp_path = Path(f.name)

    try:
        # Load and resample audio
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

    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        raise AudioProcessingError(f"Failed to process audio: {e}")

    finally:
        # Clean up temp file
        temp_path.unlink(missing_ok=True)


def load_audio_from_file(
    audio_path: Path,
    target_sr: int = MERT_CONFIG["target_sr"],
) -> Tuple[np.ndarray, float]:
    """Load audio from local file.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Tuple of (audio_array, duration_seconds)
    """
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    duration = len(audio) / sr
    return audio, duration


def preprocess_audio_from_bytes(
    audio_bytes: bytes,
    target_sr: int = MERT_CONFIG["target_sr"],
    max_duration: int = 300,
) -> Tuple[np.ndarray, float]:
    """Preprocess audio from raw bytes (e.g., base64 decoded).

    Args:
        audio_bytes: Raw audio file bytes (any format librosa supports)
        target_sr: Target sample rate (24kHz for MERT)
        max_duration: Maximum audio duration in seconds

    Returns:
        Tuple of (audio_array, duration_seconds)

    Raises:
        AudioProcessingError: If audio processing fails
    """
    import io

    try:
        # Load from bytes buffer - librosa handles format detection
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
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
        raise AudioProcessingError(f"Failed to process audio bytes: {e}")
