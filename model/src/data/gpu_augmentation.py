"""
GPU-accelerated audio augmentation using torchaudio.

Unlike CPU augmentation (librosa-based), these transforms run on GPU tensors
and can be applied during training without CPU-GPU transfer overhead.

Usage:
    augmentor = GPUAudioAugmentation(device='cuda')
    augmented_batch = augmentor(audio_batch)  # [batch, samples]
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import random

# Check for torchaudio availability
try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


class GPUAudioAugmentation(nn.Module):
    """
    GPU-accelerated audio augmentation pipeline.

    All transforms operate on GPU tensors for maximum throughput.
    Designed to be called during training after data is on GPU.

    Supported augmentations:
    - Gain variation: Random volume changes
    - Noise injection: Additive Gaussian noise
    - Time masking: SpecAugment-style time masking
    - Frequency masking: SpecAugment-style frequency masking (if using spectrograms)
    - Pitch shift: Resampling-based pitch shift (approximate)
    - Time stretch: Resampling-based time stretch (approximate)
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        device: str = "cuda",
        # Probabilities
        gain_prob: float = 0.4,
        noise_prob: float = 0.3,
        time_mask_prob: float = 0.3,
        pitch_shift_prob: float = 0.3,
        time_stretch_prob: float = 0.3,
        # Parameters
        gain_db_range: tuple = (-6.0, 6.0),
        noise_snr_range: tuple = (25.0, 40.0),
        time_mask_max_len: int = 2400,  # 100ms at 24kHz
        pitch_shift_range: tuple = (-2, 2),  # semitones
        time_stretch_range: tuple = (0.9, 1.1),
        max_augmentations: int = 3,
    ):
        """
        Initialize GPU augmentation pipeline.

        Args:
            sample_rate: Audio sample rate (default: 24kHz for MERT)
            device: Target device ('cuda' or 'cpu')
            gain_prob: Probability of applying gain variation
            noise_prob: Probability of adding noise
            time_mask_prob: Probability of time masking
            pitch_shift_prob: Probability of pitch shift
            time_stretch_prob: Probability of time stretch
            gain_db_range: Min/max gain in dB
            noise_snr_range: Min/max SNR in dB for noise
            time_mask_max_len: Maximum time mask length in samples
            pitch_shift_range: Min/max pitch shift in semitones
            time_stretch_range: Min/max time stretch ratio
            max_augmentations: Maximum augmentations to apply per sample
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.device = device
        self.max_augmentations = max_augmentations

        # Store probabilities
        self.gain_prob = gain_prob
        self.noise_prob = noise_prob
        self.time_mask_prob = time_mask_prob
        self.pitch_shift_prob = pitch_shift_prob
        self.time_stretch_prob = time_stretch_prob

        # Store parameters
        self.gain_db_range = gain_db_range
        self.noise_snr_range = noise_snr_range
        self.time_mask_max_len = time_mask_max_len
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_range = time_stretch_range

        # Pre-compute pitch shift resamplers (expensive to create)
        # Cache resamplers for common semitone shifts
        self._resamplers = {}

    def _get_resampler(self, ratio: float) -> nn.Module:
        """Get or create a resampler for the given ratio."""
        # Quantize ratio to avoid too many cached resamplers
        quantized = round(ratio, 2)
        if quantized not in self._resamplers:
            orig_freq = self.sample_rate
            new_freq = int(self.sample_rate * quantized)
            if TORCHAUDIO_AVAILABLE:
                resampler = T.Resample(orig_freq, new_freq).to(self.device)
            else:
                resampler = None
            self._resamplers[quantized] = resampler
        return self._resamplers[quantized]

    @torch.no_grad()
    def apply_gain(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random gain variation."""
        db = torch.empty(audio.shape[0], 1, device=audio.device).uniform_(
            self.gain_db_range[0], self.gain_db_range[1]
        )
        gain = 10 ** (db / 20.0)
        return audio * gain

    @torch.no_grad()
    def apply_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at random SNR."""
        snr_db = torch.empty(audio.shape[0], 1, device=audio.device).uniform_(
            self.noise_snr_range[0], self.noise_snr_range[1]
        )

        # Calculate signal power per sample
        signal_power = (audio ** 2).mean(dim=-1, keepdim=True)

        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        # Generate and add noise
        noise = torch.randn_like(audio) * torch.sqrt(noise_power)
        return audio + noise

    @torch.no_grad()
    def apply_time_mask(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply time masking (set random segment to zero)."""
        batch_size, num_samples = audio.shape

        for i in range(batch_size):
            mask_len = random.randint(0, self.time_mask_max_len)
            if mask_len > 0 and mask_len < num_samples:
                start = random.randint(0, num_samples - mask_len)
                audio[i, start:start + mask_len] = 0

        return audio

    @torch.no_grad()
    def apply_pitch_shift(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply pitch shift via resampling (approximate).

        This is faster than librosa but less accurate.
        For precise pitch shift, use CPU augmentation.
        """
        if not TORCHAUDIO_AVAILABLE:
            return audio

        batch_size, num_samples = audio.shape
        semitones = random.uniform(*self.pitch_shift_range)

        # Pitch shift ratio (2^(semitones/12))
        ratio = 2 ** (semitones / 12.0)

        # Resample to shift pitch
        resampler_up = self._get_resampler(ratio)
        resampler_down = self._get_resampler(1.0 / ratio)

        if resampler_up is None or resampler_down is None:
            return audio

        # Resample up then down to shift pitch while maintaining duration
        shifted = resampler_up(audio)
        shifted = resampler_down(shifted)

        # Ensure output matches input length
        if shifted.shape[-1] > num_samples:
            shifted = shifted[:, :num_samples]
        elif shifted.shape[-1] < num_samples:
            pad_len = num_samples - shifted.shape[-1]
            shifted = torch.nn.functional.pad(shifted, (0, pad_len))

        return shifted

    @torch.no_grad()
    def apply_time_stretch(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply time stretch via resampling.

        Changes tempo without changing pitch (approximately).
        """
        if not TORCHAUDIO_AVAILABLE:
            return audio

        batch_size, num_samples = audio.shape
        rate = random.uniform(*self.time_stretch_range)

        # Resample to stretch time
        resampler = self._get_resampler(rate)
        if resampler is None:
            return audio

        stretched = resampler(audio)

        # Crop or pad to match original length
        if stretched.shape[-1] > num_samples:
            stretched = stretched[:, :num_samples]
        elif stretched.shape[-1] < num_samples:
            pad_len = num_samples - stretched.shape[-1]
            stretched = torch.nn.functional.pad(stretched, (0, pad_len))

        return stretched

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Apply random augmentations to audio batch.

        Args:
            audio: Audio tensor [batch, samples] on GPU
            config: Optional config to override probabilities

        Returns:
            Augmented audio tensor [batch, samples]
        """
        if not self.training:
            return audio

        # Override probabilities from config if provided
        if config is not None:
            for key, value in config.items():
                if isinstance(value, dict) and value.get('enabled', True):
                    prob_attr = f"{key}_prob"
                    if hasattr(self, prob_attr) and 'probability' in value:
                        setattr(self, prob_attr, value['probability'])

        # Build list of augmentations to apply
        augmentations = []

        if random.random() < self.gain_prob:
            augmentations.append(self.apply_gain)
        if random.random() < self.noise_prob:
            augmentations.append(self.apply_noise)
        if random.random() < self.time_mask_prob:
            augmentations.append(self.apply_time_mask)
        if random.random() < self.pitch_shift_prob:
            augmentations.append(self.apply_pitch_shift)
        if random.random() < self.time_stretch_prob:
            augmentations.append(self.apply_time_stretch)

        # Limit number of augmentations
        if len(augmentations) > self.max_augmentations:
            augmentations = random.sample(augmentations, self.max_augmentations)

        # Apply augmentations
        augmented = audio
        for aug_fn in augmentations:
            augmented = aug_fn(augmented)

        # Prevent clipping
        max_val = augmented.abs().max()
        if max_val > 1.0:
            augmented = augmented / max_val * 0.99

        return augmented


class GPUMixup(nn.Module):
    """
    GPU-accelerated mixup augmentation.

    Applies mixup directly on GPU tensors for efficiency.
    """

    def __init__(self, alpha: float = 0.2, probability: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            probability: Probability of applying mixup
        """
        super().__init__()
        self.alpha = alpha
        self.probability = probability

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple:
        """
        Apply mixup to batch.

        Args:
            audio: Audio tensor [batch, samples]
            labels: Label tensor [batch, num_dims]

        Returns:
            Tuple of (mixed_audio, mixed_labels)
        """
        if not self.training or random.random() >= self.probability:
            return audio, labels

        batch_size = audio.shape[0]
        if batch_size < 2:
            return audio, labels

        # Sample mixing coefficient
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()

        # Random permutation for pairing
        index = torch.randperm(batch_size, device=audio.device)

        # Mix
        mixed_audio = lam * audio + (1 - lam) * audio[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_audio, mixed_labels


if __name__ == "__main__":
    print("GPU Audio Augmentation Module")
    print("=" * 50)
    print(f"torchaudio available: {TORCHAUDIO_AVAILABLE}")
    print("\nSupported augmentations (all GPU-accelerated):")
    print("- Gain variation: Random volume changes")
    print("- Noise injection: Additive Gaussian noise")
    print("- Time masking: Zero out random segments")
    print("- Pitch shift: Resampling-based (approximate)")
    print("- Time stretch: Resampling-based")
    print("\nUsage:")
    print("  augmentor = GPUAudioAugmentation(device='cuda')")
    print("  augmented = augmentor(audio_batch)  # [batch, samples]")
