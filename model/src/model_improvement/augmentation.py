"""Audio augmentation pipeline for domain robustness training."""

import random
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import torchaudio.functional as F


class AudioAugmentor:
    """GPU-compatible audio augmentation pipeline using torchaudio.functional.

    Applies a chain of independent augmentations, each with its own probability,
    gated by a global augment_prob. Designed for training domain-robust piano
    performance evaluation models.

    Args:
        room_irs_dir: Directory containing room impulse response WAV files.
            If None, room IR augmentation is skipped.
        noise_dir: Directory containing noise WAV files for additive noise.
            If None, additive noise augmentation is skipped.
        augment_prob: Global gate probability. If random draw exceeds this,
            the waveform is returned unchanged.
    """

    # Per-augmentation probabilities
    ROOM_IR_PROB = 0.3
    NOISE_PROB = 0.3
    PHONE_SIM_PROB = 0.2
    PITCH_SHIFT_PROB = 0.1
    EQ_PROB = 0.2

    def __init__(
        self,
        room_irs_dir: Optional[Path],
        noise_dir: Optional[Path],
        augment_prob: float = 0.5,
    ) -> None:
        if not 0.0 <= augment_prob <= 1.0:
            raise ValueError(
                f"augment_prob must be in [0, 1], got {augment_prob}"
            )

        self.room_irs_dir = Path(room_irs_dir) if room_irs_dir is not None else None
        self.noise_dir = Path(noise_dir) if noise_dir is not None else None
        self.augment_prob = augment_prob

        # Cache file lists for room IRs and noise clips
        self._room_ir_files: list[Path] = []
        self._noise_files: list[Path] = []

        if self.room_irs_dir is not None:
            if not self.room_irs_dir.is_dir():
                raise FileNotFoundError(
                    f"Room IRs directory not found: {self.room_irs_dir}"
                )
            self._room_ir_files = sorted(self.room_irs_dir.glob("*.wav"))
            if not self._room_ir_files:
                raise FileNotFoundError(
                    f"No WAV files found in room IRs directory: {self.room_irs_dir}"
                )

        if self.noise_dir is not None:
            if not self.noise_dir.is_dir():
                raise FileNotFoundError(
                    f"Noise directory not found: {self.noise_dir}"
                )
            self._noise_files = sorted(self.noise_dir.glob("*.wav"))
            if not self._noise_files:
                raise FileNotFoundError(
                    f"No WAV files found in noise directory: {self.noise_dir}"
                )

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

        augmented = waveform

        # Room impulse response convolution
        if self._room_ir_files and random.random() < self.ROOM_IR_PROB:
            augmented = self._apply_room_ir(augmented, sample_rate)

        # Additive noise
        if self._noise_files and random.random() < self.NOISE_PROB:
            augmented = self._apply_additive_noise(augmented, sample_rate)

        # Phone simulation (low-pass + compression)
        if random.random() < self.PHONE_SIM_PROB:
            augmented = self._apply_phone_simulation(augmented, sample_rate)

        # Pitch shift
        if random.random() < self.PITCH_SHIFT_PROB:
            augmented = self._apply_pitch_shift(augmented, sample_rate)

        # EQ variation
        if random.random() < self.EQ_PROB:
            augmented = self._apply_eq_variation(augmented, sample_rate)

        return augmented

    def _apply_room_ir(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        """Convolve with a random room impulse response.

        Args:
            waveform: Audio tensor of shape [C, T].
            sample_rate: Sample rate in Hz.

        Returns:
            Convolved waveform of shape [C, T] (trimmed to original length).
        """
        ir_path = random.choice(self._room_ir_files)
        ir_waveform, ir_sr = torchaudio.load(ir_path)

        # Resample IR if needed
        if ir_sr != sample_rate:
            ir_waveform = F.resample(ir_waveform, ir_sr, sample_rate)

        # Use mono IR for convolution
        if ir_waveform.shape[0] > 1:
            ir_waveform = ir_waveform.mean(dim=0, keepdim=True)

        # Move IR to same device as waveform
        ir_waveform = ir_waveform.to(waveform.device)

        # Normalize IR
        ir_max = ir_waveform.abs().max()
        if ir_max > 0:
            ir_waveform = ir_waveform / ir_max

        original_length = waveform.shape[-1]

        # Convolve each channel with the IR
        convolved = F.fftconvolve(waveform, ir_waveform.expand_as(waveform))

        # Trim to original length
        convolved = convolved[..., :original_length]

        # Normalize to prevent clipping
        conv_max = convolved.abs().max()
        if conv_max > 0:
            convolved = convolved * (waveform.abs().max() / conv_max)

        return convolved

    def _apply_additive_noise(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        snr_range: Tuple[float, float] = (10, 30),
    ) -> torch.Tensor:
        """Mix with a random noise clip at a random SNR.

        Args:
            waveform: Audio tensor of shape [C, T].
            sample_rate: Sample rate in Hz.
            snr_range: Tuple of (min_snr_db, max_snr_db).

        Returns:
            Noisy waveform of shape [C, T].
        """
        noise_path = random.choice(self._noise_files)
        noise, noise_sr = torchaudio.load(noise_path)

        # Resample noise if needed
        if noise_sr != sample_rate:
            noise = F.resample(noise, noise_sr, sample_rate)

        # Match channels
        if noise.shape[0] != waveform.shape[0]:
            if noise.shape[0] > 1:
                noise = noise.mean(dim=0, keepdim=True)
            noise = noise.expand(waveform.shape[0], -1)

        # Ensure noise is at least as long as waveform by looping
        target_length = waveform.shape[-1]
        if noise.shape[-1] < target_length:
            repeats = (target_length // noise.shape[-1]) + 1
            noise = noise.repeat(1, repeats)
        noise = noise[..., :target_length]

        noise = noise.to(waveform.device)

        # Compute random SNR
        snr_db = random.uniform(snr_range[0], snr_range[1])

        # Compute signal and noise power
        signal_power = (waveform**2).mean()
        noise_power = (noise**2).mean()

        if noise_power == 0:
            return waveform

        # Scale noise to achieve target SNR
        # SNR_dB = 10 * log10(signal_power / noise_power_scaled)
        scale = torch.sqrt(
            signal_power / (noise_power * 10 ** (snr_db / 10))
        )
        mixed = waveform + scale * noise

        return mixed

    def _apply_phone_simulation(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        """Simulate phone recording: low-pass filter at 8kHz + dynamic range compression.

        Args:
            waveform: Audio tensor of shape [C, T].
            sample_rate: Sample rate in Hz.

        Returns:
            Filtered and compressed waveform of shape [C, T].
        """
        # Low-pass filter at 8kHz (phone bandwidth)
        # Ensure cutoff doesn't exceed Nyquist
        cutoff_freq = min(8000.0, sample_rate / 2.0 - 1.0)
        filtered = F.lowpass_biquad(waveform, sample_rate, cutoff_freq, Q=0.707)

        # Simple dynamic range compression via soft clipping (tanh)
        # Scale up, apply tanh, scale back down for mild compression
        compression_gain = 2.0
        compressed = torch.tanh(filtered * compression_gain) / compression_gain

        # Normalize to preserve approximate original level
        orig_rms = torch.sqrt((waveform**2).mean())
        comp_rms = torch.sqrt((compressed**2).mean())
        if comp_rms > 0:
            compressed = compressed * (orig_rms / comp_rms)

        return compressed

    def _apply_pitch_shift(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        cents_range: Tuple[int, int] = (-50, 50),
    ) -> torch.Tensor:
        """Apply a small pitch shift in cents.

        Args:
            waveform: Audio tensor of shape [C, T].
            sample_rate: Sample rate in Hz.
            cents_range: Tuple of (min_cents, max_cents) for random shift.

        Returns:
            Pitch-shifted waveform of shape [C, T].
        """
        n_cents = random.randint(cents_range[0], cents_range[1])
        if n_cents == 0:
            return waveform

        # Use bins_per_octave=1200 so each n_step corresponds to 1 cent
        shifted = F.pitch_shift(
            waveform,
            sample_rate,
            n_steps=n_cents,
            bins_per_octave=1200,
        )
        return shifted

    def _apply_eq_variation(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        """Apply random 3-band parametric EQ using biquad peaking filters.

        Three bands with random center frequencies and gains:
        - Low band:  80-300 Hz
        - Mid band:  300-3000 Hz
        - High band: 3000-10000 Hz

        Args:
            waveform: Audio tensor of shape [C, T].
            sample_rate: Sample rate in Hz.

        Returns:
            EQ'd waveform of shape [C, T].
        """
        nyquist = sample_rate / 2.0

        # Define band ranges and select random center frequencies
        bands = [
            (80.0, 300.0),    # low
            (300.0, 3000.0),  # mid
            (3000.0, min(10000.0, nyquist - 1.0)),  # high
        ]

        result = waveform
        for low_f, high_f in bands:
            # Skip band if it would exceed Nyquist
            if low_f >= nyquist:
                continue

            center_freq = random.uniform(low_f, min(high_f, nyquist - 1.0))
            # Random gain in dB, mild range for subtle variation
            gain_db = random.uniform(-6.0, 6.0)
            q_factor = random.uniform(0.5, 2.0)

            result = F.equalizer_biquad(
                result, sample_rate, center_freq, gain_db, Q=q_factor
            )

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
    - Gain (volume variation)

    Args:
        seed: Random seed for reproducible augmentation.

    Returns:
        Callable that takes (audio_ndarray, sample_rate) and returns augmented audio.
    """
    from pedalboard import (
        Compressor,
        Gain,
        LowpassFilter,
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

    # Gain (70% chance) -- volume variation
    if rng.random() < 0.7:
        effects.append(Gain(
            gain_db=rng.uniform(-6, 3),
        ))

    board = Pedalboard(effects)

    def augment(audio: "np.ndarray", sample_rate: int) -> "np.ndarray":
        # Pedalboard expects (channels, samples)
        audio_2d = audio.reshape(1, -1).astype(np.float32)
        result = board(audio_2d, sample_rate=sample_rate)
        return result.squeeze(0).astype(np.float32)

    return augment


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
