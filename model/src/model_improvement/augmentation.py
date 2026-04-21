"""Audio augmentation pipeline for domain robustness training."""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class AudioAugmentor:
    """Pedalboard-based audio augmentation pipeline for training.

    Applies a chain of independent augmentations, each with its own probability,
    gated by a global augment_prob. Uses Pedalboard (JUCE-backed) for all DSP
    effects. Optionally convolves with a practice-room impulse response (real
    IR WAV files or a synthetic exponential-decay approximation). Tensor-in,
    tensor-out: conversion to/from numpy is handled internally.

    Args:
        augment_prob: Global gate probability. If random draw exceeds this,
            the waveform is returned unchanged.
        ir_dir: Optional directory containing .wav impulse response files.
            If provided, a random IR is loaded and convolved at IR_CONV_PROB.
            If None and IR_CONV_PROB triggers, a synthetic IR is generated
            instead (adequate for training; replace with real IRs before eval).
    """

    # Per-augmentation probabilities
    REVERB_PROB = 0.3
    NOISE_PROB = 0.3
    PHONE_SIM_PROB = 0.2
    PITCH_SHIFT_PROB = 0.1
    EQ_PROB = 0.2
    IR_CONV_PROB = 0.35

    def __init__(
        self,
        augment_prob: float = 0.5,
        ir_dir: Optional[Path] = None,
    ) -> None:
        if not 0.0 <= augment_prob <= 1.0:
            raise ValueError(
                f"augment_prob must be in [0, 1], got {augment_prob}"
            )
        self.augment_prob = augment_prob
        self.ir_dir = Path(ir_dir) if ir_dir is not None else None
        self._ir_paths: Optional[list[Path]] = None
        if self.ir_dir is not None and self.ir_dir.exists():
            self._ir_paths = sorted(self.ir_dir.glob("*.wav"))

    def __call__(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        """Apply random augmentations to waveform.

        Args:
            waveform: Audio tensor of shape [C, T].
            sample_rate: Sample rate of the waveform in Hz.

        Returns:
            Augmented waveform of the same shape [C, T].
        """
        if waveform.ndim != 2:
            raise ValueError(
                f"Expected waveform of shape [C, T], got {waveform.shape}"
            )

        # Global gate: if random draw exceeds augment_prob, skip all augmentation
        if random.random() >= self.augment_prob:
            return waveform

        from pedalboard import (
            Compressor,
            LowpassFilter,
            PeakFilter,
            Pedalboard,
            PitchShift,
            Reverb,
        )

        effects = []

        # Reverb (parametric, no WAV files needed)
        if random.random() < self.REVERB_PROB:
            effects.append(Reverb(
                room_size=random.uniform(0.1, 0.7),
                wet_level=random.uniform(0.05, 0.3),
            ))

        # Phone simulation (low-pass + compression)
        if random.random() < self.PHONE_SIM_PROB:
            effects.append(LowpassFilter(
                cutoff_frequency_hz=min(8000.0, sample_rate / 2.0 - 1.0),
            ))
            effects.append(Compressor(
                threshold_db=-20.0,
                ratio=4.0,
            ))

        # Pitch shift (in semitones; -50 to +50 cents = -0.42 to +0.42 semitones)
        if random.random() < self.PITCH_SHIFT_PROB:
            semitones = random.uniform(-0.42, 0.42)
            if abs(semitones) > 0.01:
                effects.append(PitchShift(semitones=semitones))

        # 3-band parametric EQ
        if random.random() < self.EQ_PROB:
            nyquist = sample_rate / 2.0
            bands = [
                (80.0, 300.0),
                (300.0, 3000.0),
                (3000.0, min(10000.0, nyquist - 1.0)),
            ]
            for low_f, high_f in bands:
                if low_f >= nyquist:
                    continue
                effects.append(PeakFilter(
                    cutoff_frequency_hz=random.uniform(low_f, min(high_f, nyquist - 1.0)),
                    gain_db=random.uniform(-6.0, 6.0),
                    q=random.uniform(0.5, 2.0),
                ))

        # Convert torch tensor -> numpy, apply Pedalboard, convert back
        device = waveform.device
        audio_np = waveform.cpu().numpy().astype(np.float32)

        if effects:
            board = Pedalboard(effects)
            audio_np = board(audio_np, sample_rate=sample_rate)

        # Additive pink noise (post-Pedalboard, no Pedalboard noise plugin)
        if random.random() < self.NOISE_PROB:
            audio_np = self._mix_pink_noise(audio_np, sample_rate)

        # Practice-room IR convolution (after EQ/noise so the room coloration
        # sits on top of the mic chain, matching the real signal path)
        if random.random() < self.IR_CONV_PROB:
            audio_np = self._apply_room_ir(audio_np, sample_rate)

        result = torch.from_numpy(audio_np).to(device)
        return result

    def _apply_room_ir(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Convolve audio with a practice-room impulse response.

        Uses a real IR loaded from self._ir_paths if available; otherwise
        generates a synthetic exponential-decay IR on the fly. The result
        is length-matched to the input and normalised to the input peak level
        so downstream loudness statistics are preserved.

        Args:
            audio: [C, T] float32 array.
            sample_rate: Sample rate in Hz.

        Returns:
            Convolved [C, T] float32 array, same shape as input.
        """
        from scipy.signal import fftconvolve

        if self._ir_paths:
            ir_path = random.choice(self._ir_paths)
            ir = self._load_ir(ir_path, sample_rate)
        else:
            rt60 = random.uniform(0.2, 0.7)
            ir = self._generate_synthetic_room_ir(sample_rate, rt60=rt60)

        peak_in = np.max(np.abs(audio)) + 1e-8
        C, T = audio.shape
        out = np.zeros_like(audio)
        for ch in range(C):
            convolved = fftconvolve(audio[ch], ir)[:T]
            out[ch] = convolved.astype(np.float32)

        # Normalise to input peak so loudness is preserved
        peak_out = np.max(np.abs(out)) + 1e-8
        out = out * (peak_in / peak_out)
        return out

    @staticmethod
    def _load_ir(path: Path, target_sr: int) -> np.ndarray:
        """Load an IR WAV file and resample to target_sr if needed.

        Returns a 1D float32 array normalised to peak 1.0.
        """
        from pedalboard.io import AudioFile

        with AudioFile(str(path)) as f:
            ir_audio = f.read(f.frames)
            ir_sr = int(f.samplerate)

        # Mono-mix if multi-channel
        if ir_audio.ndim == 2:
            ir = ir_audio.mean(axis=0).astype(np.float32)
        else:
            ir = ir_audio.astype(np.float32)

        # Resample if needed using scipy
        if ir_sr != target_sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(target_sr, ir_sr)
            ir = resample_poly(ir, target_sr // g, ir_sr // g).astype(np.float32)

        peak = np.max(np.abs(ir))
        if peak > 0:
            ir /= peak
        return ir

    @staticmethod
    def _generate_synthetic_room_ir(
        sample_rate: int,
        rt60: float = 0.4,
        pre_delay_ms: float = 3.0,
    ) -> np.ndarray:
        """Generate a synthetic practice-room impulse response.

        Models a small untreated room: direct sound + sparse early reflections
        + diffuse reverberant tail decaying at the given RT60.

        Args:
            sample_rate: Output sample rate in Hz.
            rt60: Reverberation time in seconds (time to decay 60 dB).
            pre_delay_ms: Pre-delay before first reflection in ms.

        Returns:
            1D float32 IR normalised to peak 1.0.
        """
        duration = rt60 * 2.5
        n = int(duration * sample_rate)
        t = np.arange(n, dtype=np.float32) / sample_rate

        # Exponential decay envelope: reaches -60 dB at rt60
        decay_rate = np.log(1e-3) / rt60  # negative
        envelope = np.exp(decay_rate * t)

        # Diffuse tail: shaped noise
        tail = np.random.randn(n).astype(np.float32) * envelope

        # Direct sound at t=0
        ir = tail.copy()
        ir[0] += 1.0

        # Early reflections: a few attenuated copies at small delays
        pre_delay = int(pre_delay_ms * sample_rate / 1000)
        reflection_delays_ms = [7.0, 14.0, 22.0, 35.0]
        reflection_gains = [0.7, 0.5, 0.4, 0.25]
        for delay_ms, gain in zip(reflection_delays_ms, reflection_gains):
            d = int(delay_ms * sample_rate / 1000) + pre_delay
            if d < n:
                ir[d] += gain * np.exp(decay_rate * d / sample_rate)

        # Normalise
        peak = np.max(np.abs(ir))
        if peak > 0:
            ir /= peak
        return ir

    @staticmethod
    def _mix_pink_noise(
        audio: np.ndarray,
        sample_rate: int,
        snr_range: tuple[float, float] = (10.0, 30.0),
    ) -> np.ndarray:
        """Mix pink noise into audio at a random SNR.

        Args:
            audio: Audio array of shape [C, T].
            sample_rate: Sample rate in Hz.
            snr_range: (min_snr_db, max_snr_db).

        Returns:
            Noisy audio of the same shape.
        """
        snr_db = random.uniform(snr_range[0], snr_range[1])
        C, T = audio.shape
        result = audio.copy()

        for ch in range(C):
            # Generate pink noise via spectral shaping of white noise
            white = np.random.randn(T).astype(np.float32)
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(T)
            freqs[0] = 1.0  # avoid division by zero at DC
            fft *= 1.0 / np.sqrt(freqs)
            noise = np.fft.irfft(fft, n=T).astype(np.float32)

            signal_power = np.mean(audio[ch] ** 2)
            noise_power = np.mean(noise ** 2)
            if noise_power == 0:
                continue

            scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            result[ch] = audio[ch] + scale * noise

        return result


# ---------------------------------------------------------------------------
# Pedalboard-based augmentation for T4 offline data pipeline
# ---------------------------------------------------------------------------


def create_augmentation_chain(seed: int | None = None):
    """Create a Pedalboard augmentation chain for piano audio.

    Applies a random subset of:
    - Reverb (room IR simulation)
    - Compression (dynamic range reduction)
    - Low-pass filter (phone mic simulation)
    - Pitch shift (slight tuning variation)
    - Parametric EQ (3-band: 80-300, 300-3000, 3000-10000 Hz)
    - Noise mixing (pink noise at 10-30 dB SNR)

    Args:
        seed: Random seed for reproducible augmentation.

    Returns:
        Callable that takes (audio_ndarray, sample_rate) and returns augmented audio.
    """
    from pedalboard import (
        Compressor,
        LowpassFilter,
        PeakFilter,
        Pedalboard,
        PitchShift,
        Reverb,
    )

    import numpy as np

    rng = np.random.RandomState(seed)

    # Randomly sample augmentation parameters
    effects = []

    # Reverb (60% chance) -- room acoustics variation
    if rng.random() < 0.6:
        effects.append(Reverb(
            room_size=rng.uniform(0.1, 0.7),
            wet_level=rng.uniform(0.05, 0.3),
        ))

    # Compression (50% chance) -- dynamic range reduction
    if rng.random() < 0.5:
        effects.append(Compressor(
            threshold_db=rng.uniform(-30, -10),
            ratio=rng.uniform(2.0, 6.0),
        ))

    # Low-pass filter (40% chance) -- phone/laptop mic simulation
    if rng.random() < 0.4:
        effects.append(LowpassFilter(
            cutoff_frequency_hz=rng.uniform(4000, 12000),
        ))

    # Pitch shift (30% chance) -- slight tuning variation
    if rng.random() < 0.3:
        effects.append(PitchShift(
            semitones=rng.uniform(-0.5, 0.5),
        ))

    # Parametric EQ (70% chance) -- 3-band EQ matching torchaudio implementation
    if rng.random() < 0.7:
        bands = [
            (80.0, 300.0),     # low
            (300.0, 3000.0),   # mid
            (3000.0, 10000.0), # high
        ]
        for low_f, high_f in bands:
            effects.append(PeakFilter(
                cutoff_frequency_hz=rng.uniform(low_f, high_f),
                gain_db=rng.uniform(-6.0, 6.0),
                q=rng.uniform(0.5, 2.0),
            ))

    board = Pedalboard(effects)

    # Pre-generate noise mixing parameters
    apply_noise = rng.random() < 0.5
    noise_snr_db = rng.uniform(10.0, 30.0)
    noise_seed = int(rng.randint(0, 2**31))

    def augment(audio: "np.ndarray", sample_rate: int) -> "np.ndarray":
        # Pedalboard expects (channels, samples)
        audio_2d = audio.reshape(1, -1).astype(np.float32)
        result = board(audio_2d, sample_rate=sample_rate)
        result = result.squeeze(0).astype(np.float32)

        # Noise mixing (post-Pedalboard, since Pedalboard has no noise plugin)
        if apply_noise:
            result = _mix_noise(result, snr_db=noise_snr_db, seed=noise_seed)

        return result

    return augment


def _mix_noise(
    audio: "np.ndarray",
    snr_db: float = 20.0,
    seed: int = 0,
) -> "np.ndarray":
    """Mix pink noise into audio at a target SNR.

    Args:
        audio: 1D float32 audio array.
        snr_db: Target signal-to-noise ratio in dB.
        seed: Random seed for noise generation.

    Returns:
        Noisy audio as 1D float32 ndarray.
    """
    import numpy as np

    rng = np.random.RandomState(seed)

    # Generate pink noise via spectral shaping of white noise
    white = rng.randn(len(audio)).astype(np.float32)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white))
    # Avoid division by zero at DC
    freqs[0] = 1.0
    # Pink noise: 1/sqrt(f) power spectrum
    fft *= 1.0 / np.sqrt(freqs)
    noise = np.fft.irfft(fft, n=len(audio)).astype(np.float32)

    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return audio

    # Scale noise to achieve target SNR
    scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return (audio + scale * noise).astype(np.float32)


def augment_audio(
    audio: "np.ndarray",
    sr: int = 24000,
    seed: int | None = None,
) -> "np.ndarray":
    """Apply random Pedalboard augmentation to audio.

    Args:
        audio: 1D float32 audio array.
        sr: Sample rate.
        seed: Random seed.

    Returns:
        Augmented audio as 1D float32 ndarray (same shape as input).
    """
    chain = create_augmentation_chain(seed=seed)
    return chain(audio, sample_rate=sr)


def augment_and_embed_piano(
    cache_dir: "Path",
    segment_duration: float = 30.0,
) -> int:
    """Generate augmented audio and extract MuQ embeddings for T4 invariance.

    For each segment in cache_dir/metadata.jsonl:
    1. Load the corresponding audio segment from the full recording
    2. Apply random augmentation
    3. Extract MuQ embedding from augmented audio
    4. Save to cache_dir/muq_embeddings_augmented/{segment_id}.pt

    Args:
        cache_dir: YouTube piano cache directory.
        segment_duration: Segment duration used during segmentation.

    Returns:
        Count of newly processed augmented segments.
    """
    import logging
    from collections import defaultdict
    from pathlib import Path

    import jsonlines
    import torch
    from audio_experiments.extractors.muq import MuQExtractor
    from model_improvement.audio_utils import load_audio

    logger = logging.getLogger(__name__)

    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "metadata.jsonl"
    audio_dir = cache_dir / "audio"
    aug_emb_dir = cache_dir / "muq_embeddings_augmented"

    if not metadata_path.exists():
        logger.warning("No segment metadata at %s", metadata_path)
        return 0

    aug_emb_dir.mkdir(parents=True, exist_ok=True)

    # Load segment metadata
    with jsonlines.open(metadata_path) as reader:
        segments = list(reader)

    # Check which augmented embeddings already exist
    existing = {p.stem for p in aug_emb_dir.glob("*.pt")}

    to_process = [s for s in segments if s["segment_id"] not in existing]
    if not to_process:
        logger.info("All %d augmented embeddings already cached", len(segments))
        return 0

    logger.info("Augmenting %d segments (%d already cached)", len(to_process), len(existing))

    extractor = MuQExtractor(cache_dir=aug_emb_dir)
    new_count = 0

    # Group segments by video_id to load audio once per recording
    video_segments: dict[str, list[dict]] = defaultdict(list)
    for seg in to_process:
        video_segments[seg["video_id"]].append(seg)

    for video_id, segs in video_segments.items():
        wav_path = audio_dir / f"{video_id}.wav"
        if not wav_path.exists():
            logger.warning("Audio not found: %s", wav_path)
            continue

        audio, sr = load_audio(wav_path, target_sr=24000)

        for seg in segs:
            segment_id = seg["segment_id"]
            start_sample = int(seg["segment_start"] * sr)
            end_sample = int(seg["segment_end"] * sr)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) < sr:  # less than 1 second
                continue

            # Apply augmentation with segment-specific seed for reproducibility
            seed = hash(segment_id) % (2**31)
            augmented = augment_audio(segment_audio, sr=sr, seed=seed)

            # Extract MuQ embedding from augmented audio
            audio_tensor = torch.from_numpy(augmented).float()
            embedding = extractor.extract_from_audio(audio_tensor)

            torch.save(embedding, aug_emb_dir / f"{segment_id}.pt")  # nosemgrep
            new_count += 1

    del extractor
    logger.info("Generated %d augmented embeddings", new_count)
    return new_count
