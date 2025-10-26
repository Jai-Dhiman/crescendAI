import numpy as np
import librosa
import scipy.signal
from typing import Optional, Dict, Any
import random


class AudioAugmentation:
    """
    Audio augmentation pipeline for piano performance evaluation.

    Augmentations preserve performance characteristics while increasing robustness:
    - Pitch shift: ±2 semitones
    - Time stretch: 0.85-1.15× speed
    - Gaussian noise: SNR 25-40dB
    - Gain variation: ±6dB
    - Room acoustics: Convolution with impulse responses
    - MP3 compression simulation: Quality degradation

    Max 3 simultaneous augmentations applied.
    """

    def __init__(
        self,
        pitch_shift_prob: float = 0.3,
        time_stretch_prob: float = 0.3,
        noise_prob: float = 0.2,
        gain_prob: float = 0.3,
        room_acoustics_prob: float = 0.4,
        compression_prob: float = 0.2,
        max_augmentations: int = 3,
        sr: int = 44100,
    ):
        """
        Initialize augmentation pipeline.

        Args:
            pitch_shift_prob: Probability of applying pitch shift
            time_stretch_prob: Probability of applying time stretch
            noise_prob: Probability of adding noise
            gain_prob: Probability of gain variation
            room_acoustics_prob: Probability of room acoustics
            compression_prob: Probability of compression
            max_augmentations: Maximum number of simultaneous augmentations
            sr: Sample rate
        """
        self.pitch_shift_prob = pitch_shift_prob
        self.time_stretch_prob = time_stretch_prob
        self.noise_prob = noise_prob
        self.gain_prob = gain_prob
        self.room_acoustics_prob = room_acoustics_prob
        self.compression_prob = compression_prob
        self.max_augmentations = max_augmentations
        self.sr = sr

    def pitch_shift(self, audio: np.ndarray, semitones: Optional[float] = None) -> np.ndarray:
        """
        Shift pitch by specified semitones.

        Args:
            audio: Input audio signal
            semitones: Number of semitones to shift (default: random ±2)

        Returns:
            Pitch-shifted audio
        """
        if semitones is None:
            semitones = np.random.uniform(-2.0, 2.0)

        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=semitones)

    def time_stretch(self, audio: np.ndarray, rate: Optional[float] = None) -> np.ndarray:
        """
        Stretch or compress audio in time.

        Args:
            audio: Input audio signal
            rate: Stretch rate (default: random 0.85-1.15×)

        Returns:
            Time-stretched audio
        """
        if rate is None:
            rate = np.random.uniform(0.85, 1.15)

        return librosa.effects.time_stretch(audio, rate=rate)

    def add_noise(
        self,
        audio: np.ndarray,
        snr_db: Optional[float] = None
    ) -> np.ndarray:
        """
        Add Gaussian noise at specified SNR.

        Args:
            audio: Input audio signal
            snr_db: Signal-to-noise ratio in dB (default: random 25-40dB)

        Returns:
            Audio with added noise
        """
        if snr_db is None:
            snr_db = np.random.uniform(25.0, 40.0)

        # Calculate signal power
        signal_power = np.mean(audio ** 2)

        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        # Generate and add noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))

        return audio + noise

    def apply_room_acoustics(
        self,
        audio: np.ndarray,
        impulse_response: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply room acoustics via impulse response convolution.

        Args:
            audio: Input audio signal
            impulse_response: Room impulse response (default: simple synthetic)

        Returns:
            Audio with room acoustics
        """
        if impulse_response is None:
            # Generate simple synthetic impulse response
            impulse_response = self._generate_simple_ir()

        # Convolve with impulse response
        return scipy.signal.fftconvolve(audio, impulse_response, mode='same')

    def _generate_simple_ir(self) -> np.ndarray:
        """
        Generate simple synthetic impulse response for room acoustics.

        Returns:
            Synthetic impulse response
        """
        # Room parameters
        delay_ms = np.random.uniform(10, 50)  # Early reflection delay
        decay_time = np.random.uniform(0.2, 0.8)  # Reverb tail

        delay_samples = int(delay_ms * self.sr / 1000)
        ir_length = int(decay_time * self.sr)

        # Create impulse response
        ir = np.zeros(ir_length)
        ir[0] = 1.0  # Direct sound

        # Early reflections
        for i in range(5):
            reflection_time = delay_samples + int(np.random.exponential(delay_samples))
            if reflection_time < ir_length:
                ir[reflection_time] = np.random.uniform(0.1, 0.3)

        # Exponential decay tail
        tail_start = delay_samples * 2
        if tail_start < ir_length:
            decay_envelope = np.exp(-np.arange(ir_length - tail_start) / (decay_time * self.sr))
            tail = np.random.randn(ir_length - tail_start) * decay_envelope * 0.1
            ir[tail_start:] += tail

        return ir

    def compress_audio(
        self,
        audio: np.ndarray,
        quality: Optional[str] = None
    ) -> np.ndarray:
        """
        Simulate MP3 compression artifacts.

        This is a simplified simulation using low-pass filtering and bit reduction.
        Real MP3 compression would require additional libraries.

        Args:
            audio: Input audio signal
            quality: 'low' (128kbps) or 'medium' (192kbps) or 'high' (320kbps)

        Returns:
            Compressed audio simulation
        """
        if quality is None:
            quality = random.choice(['low', 'medium', 'high'])

        # Simulate compression with low-pass filter and bit reduction
        if quality == 'low':
            cutoff = 12000  # Hz
            bits = 12
        elif quality == 'medium':
            cutoff = 16000
            bits = 14
        else:  # high
            cutoff = 18000
            bits = 15

        # Low-pass filter
        nyquist = self.sr / 2
        normalized_cutoff = cutoff / nyquist
        b, a = scipy.signal.butter(4, normalized_cutoff, btype='low')
        filtered = scipy.signal.filtfilt(b, a, audio)

        # Bit reduction (quantization)
        max_val = 2 ** (bits - 1)
        quantized = np.round(filtered * max_val) / max_val

        return quantized

    def gain_variation(
        self,
        audio: np.ndarray,
        db_range: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply random gain variation.

        Args:
            audio: Input audio signal
            db_range: Gain variation range in dB (default: random ±6dB)

        Returns:
            Audio with gain applied
        """
        if db_range is None:
            db_range = np.random.uniform(-6.0, 6.0)

        gain_linear = 10 ** (db_range / 20.0)
        return audio * gain_linear

    def augment_pipeline(
        self,
        audio: np.ndarray,
        config: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Apply random subset of augmentations (max 3).

        Args:
            audio: Input audio signal
            config: Optional configuration dict to override probabilities

        Returns:
            Augmented audio
        """
        if config is not None:
            # Override probabilities if provided
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # List of available augmentations
        augmentations = []

        if random.random() < self.pitch_shift_prob:
            augmentations.append(('pitch_shift', {}))
        if random.random() < self.time_stretch_prob:
            augmentations.append(('time_stretch', {}))
        if random.random() < self.noise_prob:
            augmentations.append(('add_noise', {}))
        if random.random() < self.gain_prob:
            augmentations.append(('gain_variation', {}))
        if random.random() < self.room_acoustics_prob:
            augmentations.append(('apply_room_acoustics', {}))
        if random.random() < self.compression_prob:
            augmentations.append(('compress_audio', {}))

        # Limit to max_augmentations
        if len(augmentations) > self.max_augmentations:
            augmentations = random.sample(augmentations, self.max_augmentations)

        # Apply augmentations
        augmented = audio.copy()
        for aug_name, aug_params in augmentations:
            aug_func = getattr(self, aug_name)
            augmented = aug_func(augmented, **aug_params)

        # Ensure no clipping
        max_val = np.abs(augmented).max()
        if max_val > 1.0:
            augmented = augmented / max_val * 0.99

        return augmented


def create_augmentation_pipeline(
    sr: int = 44100,
    training: bool = True
) -> Optional[AudioAugmentation]:
    """
    Create augmentation pipeline for training or inference.

    Args:
        sr: Sample rate
        training: Whether this is for training (True) or inference (False)

    Returns:
        AudioAugmentation instance or None (for inference)
    """
    if not training:
        return None

    return AudioAugmentation(
        pitch_shift_prob=0.3,
        time_stretch_prob=0.3,
        noise_prob=0.2,
        gain_prob=0.3,
        room_acoustics_prob=0.4,
        compression_prob=0.2,
        max_augmentations=3,
        sr=sr,
    )


if __name__ == "__main__":
    print("Audio augmentation module loaded successfully")
    print("Available augmentations:")
    print("- Pitch shift: ±2 semitones (p=0.3)")
    print("- Time stretch: 0.85-1.15× (p=0.3)")
    print("- Noise: SNR 25-40dB (p=0.2)")
    print("- Gain: ±6dB (p=0.3)")
    print("- Room acoustics: IR convolution (p=0.4)")
    print("- Compression: Simulated MP3 (p=0.2)")
    print("- Max 3 simultaneous augmentations")
