import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Union
import warnings

# Try to import torchaudio (faster than librosa)
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


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

    Uses soundfile backend explicitly to avoid TorchCodec dependency.
    GPU-accelerated resampling available.

    Args:
        path: Path to audio file
        sr: Target sample rate
        mono: Convert to mono if True
        duration: Maximum duration in seconds (None = load all)
        offset: Start reading after this time (in seconds)
        use_gpu: Use GPU for resampling (faster for large batches)

    Returns:
        Tuple of (audio array, sample rate)
    """
    import torch

    # Load audio using soundfile backend (avoids TorchCodec dependency)
    waveform, original_sr = torchaudio.load(str(path), backend="soundfile")

    # Apply offset and duration
    if offset > 0 or duration is not None:
        start_frame = int(offset * original_sr)
        if duration is not None:
            num_frames = int(duration * original_sr)
            waveform = waveform[:, start_frame:start_frame + num_frames]
        else:
            waveform = waveform[:, start_frame:]

    # Convert to mono if needed
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed (GPU-accelerated if requested)
    if sr != original_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sr,
            new_freq=sr,
        )
        if use_gpu and torch.cuda.is_available():
            waveform = waveform.cuda()
            resampler = resampler.cuda()
            waveform = resampler(waveform)
            waveform = waveform.cpu()
        else:
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
    prefer_torchaudio: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate with retry logic.

    PRODUCTION: Default is 24kHz to match MERT-v1-95M requirements.
    MERT expects raw audio waveforms at 24kHz sampling rate.

    Includes retry logic for Google Drive I/O errors (common in Colab).

    Performance:
    - torchaudio (preferred): 3-10x faster than librosa, GPU-accelerated
    - librosa (fallback): Slower but more compatible

    Args:
        path: Path to audio file (WAV, MP3, FLAC, etc.)
        sr: Target sample rate (default: 24000 Hz for MERT)
        mono: Convert to mono if True
        duration: Maximum duration in seconds (None = load all)
        offset: Start reading after this time (in seconds)
        max_retries: Maximum retry attempts for I/O errors (default: 3)
        retry_delay: Delay between retries in seconds (default: 0.5)
        prefer_torchaudio: Use torchaudio if available (default: True)

    Returns:
        Tuple of (audio array, sample rate)

    Raises:
        OSError: If file cannot be loaded after all retries
        Exception: For other errors (corrupted file, unsupported format)
    """
    import time

    # Try torchaudio first if available and preferred
    if prefer_torchaudio and TORCHAUDIO_AVAILABLE:
        last_error = None
        for attempt in range(max_retries):
            try:
                audio, sr_out = load_audio_torchaudio(
                    path, sr, mono, duration, offset
                )
                return audio, sr_out
            except OSError as e:
                # I/O error (retry)
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    # Fall through to librosa
                    warnings.warn(
                        f"torchaudio failed after {max_retries} attempts, "
                        f"trying librosa: {e}"
                    )
                    break
            except Exception as e:
                # Other errors, fall through to librosa
                warnings.warn(f"torchaudio failed, trying librosa: {e}")
                break

    # Fallback to librosa
    last_error = None
    for attempt in range(max_retries):
        try:
            audio, original_sr = librosa.load(
                path,
                sr=sr,
                mono=mono,
                duration=duration,
                offset=offset
            )
            return audio, sr
        except OSError as e:
            # I/O error (common with Google Drive mounts in Colab)
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                # All retries exhausted
                raise OSError(
                    f"Failed to load audio after {max_retries} attempts: {path}"
                ) from e
        except Exception as e:
            # Other errors (corrupted file, unsupported format) - don't retry
            raise Exception(f"Failed to load audio file: {path}") from e

    # Should never reach here, but just in case
    if last_error is not None:
        raise last_error
    raise Exception(f"Unexpected error loading audio: {path}")


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
        'audio': audio,
        'segments': segments,
        'sr': sr,
        'duration': len(audio) / sr,
    }

    # Compute CQT for each segment (optional, for visualization only)
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
    print("PRODUCTION CONFIGURATION:")
    print(f"- Sample rate: 24kHz (MERT requirement)")
    print(f"- Format: Raw audio waveforms (not spectrograms)")
    print(f"- CQT: Available for visualization only (disabled by default)")
    print(f"- Piano range: C1 to C8 (7 octaves, 168 frequency bins)")
