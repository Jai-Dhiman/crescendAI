import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Cache for GPU resamplers (expensive to create)
_GPU_RESAMPLERS: dict = {}


def _get_resampler(
    orig_freq: int,
    new_freq: int,
    device: torch.device,
) -> torchaudio.transforms.Resample:
    """Get or create a cached resampler for the given frequencies."""
    key = (orig_freq, new_freq, str(device))
    if key not in _GPU_RESAMPLERS:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_freq,
            new_freq=new_freq,
        ).to(device)
        _GPU_RESAMPLERS[key] = resampler
    return _GPU_RESAMPLERS[key]


def load_audio_torchaudio(
    path: Union[str, Path],
    sr: int = 24000,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Load audio using torchaudio (3-10x faster than librosa).

    Modern torchaudio API (2.0+) auto-detects the best backend.
    GPU-accelerated resampling available for significant speedup.

    Args:
        path: Path to audio file
        sr: Target sample rate
        mono: Convert to mono if True
        duration: Maximum duration in seconds (None = load all)
        offset: Start reading after this time (in seconds)
        use_gpu: Use GPU for resampling (faster for large batches)

    Returns:
        Tuple of (audio array, sample rate)

    Raises:
        RuntimeError: If audio file cannot be loaded
    """
    path_str = str(path)

    # Modern torchaudio.load() - no backend parameter needed
    # It auto-detects: uses ffmpeg/soundfile based on installation
    if offset > 0 or duration is not None:
        # Get file info first for offset/duration calculation
        info = torchaudio.info(path_str)
        original_sr = info.sample_rate

        frame_offset = int(offset * original_sr) if offset > 0 else 0
        num_frames = int(duration * original_sr) if duration is not None else -1

        waveform, original_sr = torchaudio.load(
            path_str,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
    else:
        waveform, original_sr = torchaudio.load(path_str)

    # Convert to mono if needed
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed (GPU-accelerated if requested)
    if sr != original_sr:
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            waveform = waveform.to(device)
            resampler = _get_resampler(original_sr, sr, device)
            waveform = resampler(waveform)
            waveform = waveform.cpu()
        else:
            resampler = _get_resampler(original_sr, sr, torch.device("cpu"))
            waveform = resampler(waveform)

    # Convert to numpy
    audio = waveform.squeeze().numpy()

    return audio, sr


def load_audio(
    path: Union[str, Path],
    sr: int = 24000,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 0.5,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file using torchaudio with retry logic.

    PRODUCTION: Uses torchaudio exclusively (no librosa fallback).
    Default is 24kHz to match MERT-v1-95M requirements.

    Includes retry logic for Google Drive I/O errors (common in Colab).

    Performance:
    - torchaudio: 3-10x faster than librosa
    - GPU resampling: Additional 2-5x speedup on resampling

    Args:
        path: Path to audio file (WAV, MP3, FLAC, etc.)
        sr: Target sample rate (default: 24000 Hz for MERT)
        mono: Convert to mono if True
        duration: Maximum duration in seconds (None = load all)
        offset: Start reading after this time (in seconds)
        max_retries: Maximum retry attempts for I/O errors (default: 3)
        retry_delay: Delay between retries in seconds (default: 0.5)
        use_gpu: Use GPU for resampling (default: False)

    Returns:
        Tuple of (audio array, sample rate)

    Raises:
        OSError: If file cannot be loaded after all retries
        RuntimeError: For torchaudio errors (corrupted file, unsupported format)
    """
    import time

    last_error = None
    for attempt in range(max_retries):
        try:
            audio, sr_out = load_audio_torchaudio(
                path, sr, mono, duration, offset, use_gpu=use_gpu
            )
            return audio, sr_out
        except OSError as e:
            # I/O error - retry (common with Google Drive mounts)
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                raise OSError(
                    f"Failed to load audio after {max_retries} attempts: {path}"
                ) from e
        except Exception as e:
            # Other errors (corrupted file, unsupported format) - don't retry
            raise RuntimeError(f"Failed to load audio file: {path}") from e

    # Should never reach here
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unexpected error loading audio: {path}")


def compute_cqt(
    audio: np.ndarray,
    sr: int = 24000,
    hop_length: int = 512,
    n_bins: int = 168,
    bins_per_octave: int = 24,
    fmin: Optional[float] = None,
) -> np.ndarray:
    """
    Compute Constant-Q Transform spectrogram.

    NOTE: CQT is NOT used by MERT (which processes raw waveforms).
    This function is kept for:
    - Visualization purposes
    - Alternative models that may use spectrograms
    - Analysis and debugging

    CQT parameters optimized for piano (C1 to C8, 7 octaves):
    - 24 bins per octave captures piano harmonics
    - Hop length 512 samples = ~11.6ms @ 44.1kHz, ~21.3ms @ 24kHz
    - Logarithmic frequency spacing matches musical pitch

    Args:
        audio: Audio signal
        sr: Sample rate (default: 24000 Hz)
        hop_length: Number of samples between frames
        n_bins: Total number of frequency bins (default: 168 = 24 bins × 7 octaves)
        bins_per_octave: Frequency bins per octave
        fmin: Minimum frequency (default: C1 = 32.7 Hz)

    Returns:
        CQT spectrogram of shape [n_bins, time_frames]
    """
    # Import librosa only when CQT is needed (not for audio loading)
    import librosa

    if fmin is None:
        fmin = librosa.note_to_hz("C1")  # 32.7 Hz

    cqt = librosa.cqt(
        audio,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
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
        segment = audio[start : start + segment_samples]
        segments.append(segment)
        start += hop_samples

    # Handle last segment if remaining audio is significant
    if start < len(audio) and len(audio) - start > segment_samples * 0.5:
        # Pad the last segment to full length
        last_segment = audio[start:]
        pad_length = segment_samples - len(last_segment)
        last_segment = np.pad(last_segment, (0, pad_length), mode="constant")
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
    sr: int = 24000,
    segment_len: float = 10.0,
    overlap: float = 0.5,
    normalize: bool = True,
    compute_cqt_specs: bool = False,
) -> dict:
    """
    Complete preprocessing pipeline for a single audio file.

    PRODUCTION: Returns raw audio waveforms for MERT processing.
    CQT computation is disabled by default (not needed for MERT).

    Args:
        path: Path to audio file
        sr: Target sample rate (default: 24000 Hz for MERT)
        segment_len: Segment length in seconds
        overlap: Overlap ratio for segmentation
        normalize: Whether to normalize audio
        compute_cqt_specs: Whether to compute CQT spectrograms (default: False)
                          Only enable for visualization/debugging

    Returns:
        Dictionary containing:
            - 'audio': Full audio array (raw waveform)
            - 'segments': List of audio segments (raw waveforms)
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
        "audio": audio,
        "segments": segments,
        "sr": sr,
        "duration": len(audio) / sr,
    }

    # Compute CQT for each segment (optional, for visualization only)
    if compute_cqt_specs:
        cqt_segments = []
        for segment in segments:
            cqt = compute_cqt(segment, sr=sr)
            cqt_segments.append(cqt)
        result["cqt_segments"] = cqt_segments

    return result


if __name__ == "__main__":
    # Example usage and diagnostics
    print("Audio Processing Module")
    print("=" * 50)
    print(f"torchaudio version: {torchaudio.__version__}")
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    print("PRODUCTION CONFIGURATION:")
    print("- Audio loading: torchaudio (no librosa fallback)")
    print("- Sample rate: 24kHz (MERT requirement)")
    print("- Format: Raw audio waveforms (not spectrograms)")
    print("- Resampling: GPU-accelerated when use_gpu=True")
    print("- CQT: Available for visualization only (uses librosa)")
    print("- Piano range: C1 to C8 (7 octaves, 168 frequency bins)")
