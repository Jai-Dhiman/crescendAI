import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Union


def load_audio(
    path: Union[str, Path],
    sr: int = 44100,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file (WAV, MP3, FLAC, etc.)
        sr: Target sample rate (default: 44100 Hz)
        mono: Convert to mono if True
        duration: Maximum duration in seconds (None = load all)
        offset: Start reading after this time (in seconds)

    Returns:
        Tuple of (audio array, sample rate)
    """
    audio, original_sr = librosa.load(
        path,
        sr=sr,
        mono=mono,
        duration=duration,
        offset=offset
    )

    return audio, sr


def compute_cqt(
    audio: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512,
    n_bins: int = 168,
    bins_per_octave: int = 24,
    fmin: Optional[float] = None,
) -> np.ndarray:
    """
    Compute Constant-Q Transform spectrogram.

    CQT parameters optimized for piano (C1 to C8, 7 octaves):
    - 24 bins per octave captures piano harmonics
    - Hop length 512 samples = ~11.6ms @ 44.1kHz
    - Logarithmic frequency spacing matches musical pitch

    Args:
        audio: Audio signal
        sr: Sample rate
        hop_length: Number of samples between frames
        n_bins: Total number of frequency bins (default: 168 = 24 bins × 7 octaves)
        bins_per_octave: Frequency bins per octave
        fmin: Minimum frequency (default: C1 = 32.7 Hz)

    Returns:
        CQT spectrogram of shape [n_bins, time_frames]
    """
    if fmin is None:
        fmin = librosa.note_to_hz('C1')  # 32.7 Hz

    cqt = librosa.cqt(
        audio,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin
    )

    # Convert to magnitude (dB scale)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    return cqt_db


def segment_audio(
    audio: np.ndarray,
    sr: int,
    segment_len: float = 10.0,
    overlap: float = 0.5,
) -> list:
    """
    Split audio into fixed-length segments with overlap.

    Args:
        audio: Audio signal
        sr: Sample rate
        segment_len: Segment length in seconds
        overlap: Overlap ratio (0.0 to 1.0)

    Returns:
        List of audio segments
    """
    segment_samples = int(segment_len * sr)
    hop_samples = int(segment_samples * (1 - overlap))

    segments = []
    start = 0

    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        start += hop_samples

    # Handle last segment if remaining audio is significant
    if start < len(audio) and len(audio) - start > segment_samples * 0.5:
        # Pad the last segment to full length
        last_segment = audio[start:]
        pad_length = segment_samples - len(last_segment)
        last_segment = np.pad(last_segment, (0, pad_length), mode='constant')
        segments.append(last_segment)

    return segments


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -3.0,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Peak normalize audio to target dB level.

    Args:
        audio: Audio signal
        target_db: Target peak level in dB (default: -3 dB)
        eps: Small constant to avoid division by zero

    Returns:
        Normalized audio
    """
    # Find current peak
    peak = np.abs(audio).max()

    if peak < eps:
        return audio

    # Calculate gain to reach target
    target_linear = 10 ** (target_db / 20.0)
    gain = target_linear / peak

    # Apply gain
    normalized = audio * gain

    return normalized


def get_audio_duration(path: Union[str, Path]) -> float:
    """
    Get audio file duration without loading entire file.

    Args:
        path: Path to audio file

    Returns:
        Duration in seconds
    """
    info = sf.info(str(path))
    return info.duration


def preprocess_audio_file(
    path: Union[str, Path],
    sr: int = 44100,
    segment_len: float = 10.0,
    overlap: float = 0.5,
    normalize: bool = True,
    compute_cqt_specs: bool = True,
) -> dict:
    """
    Complete preprocessing pipeline for a single audio file.

    Args:
        path: Path to audio file
        sr: Target sample rate
        segment_len: Segment length in seconds
        overlap: Overlap ratio for segmentation
        normalize: Whether to normalize audio
        compute_cqt_specs: Whether to compute CQT spectrograms

    Returns:
        Dictionary containing:
            - 'audio': Full audio array
            - 'segments': List of audio segments
            - 'cqt_segments': List of CQT spectrograms (if compute_cqt_specs=True)
            - 'sr': Sample rate
            - 'duration': Duration in seconds
    """
    # Load audio
    audio, sr = load_audio(path, sr=sr)

    # Normalize if requested
    if normalize:
        audio = normalize_audio(audio)

    # Segment audio
    segments = segment_audio(audio, sr, segment_len, overlap)

    result = {
        'audio': audio,
        'segments': segments,
        'sr': sr,
        'duration': len(audio) / sr,
    }

    # Compute CQT for each segment
    if compute_cqt_specs:
        cqt_segments = []
        for segment in segments:
            cqt = compute_cqt(segment, sr=sr)
            cqt_segments.append(cqt)
        result['cqt_segments'] = cqt_segments

    return result


if __name__ == "__main__":
    # Example usage
    print("Audio processing module loaded successfully")
    print(f"Example: compute CQT for piano range (C1 to C8)")
    print(f"- Frequency bins: 168 (24 bins/octave × 7 octaves)")
    print(f"- Hop length: 512 samples (~11.6ms @ 44.1kHz)")
