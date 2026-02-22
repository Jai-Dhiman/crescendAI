"""Audio augmentation pipeline for domain robustness training."""

import random

import numpy as np
import torch


class AudioAugmentor:
    """Pedalboard-based audio augmentation pipeline for training.

    Applies a chain of independent augmentations, each with its own probability,
    gated by a global augment_prob. Uses Pedalboard (JUCE-backed) for all DSP
    effects. Tensor-in, tensor-out: conversion to/from numpy is handled internally.

    Args:
        augment_prob: Global gate probability. If random draw exceeds this,
            the waveform is returned unchanged.
    """

    # Per-augmentation probabilities
    REVERB_PROB = 0.3
    NOISE_PROB = 0.3
    PHONE_SIM_PROB = 0.2
    PITCH_SHIFT_PROB = 0.1
    EQ_PROB = 0.2

    def __init__(self, augment_prob: float = 0.5) -> None:
        if not 0.0 <= augment_prob <= 1.0:
            raise ValueError(
                f"augment_prob must be in [0, 1], got {augment_prob}"
            )
        self.augment_prob = augment_prob

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

        result = torch.from_numpy(audio_np).to(device)
        return result

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

            torch.save(embedding, aug_emb_dir / f"{segment_id}.pt")
            new_count += 1

    del extractor
    logger.info("Generated %d augmented embeddings", new_count)
    return new_count
