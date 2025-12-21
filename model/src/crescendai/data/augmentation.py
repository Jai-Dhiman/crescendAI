import numpy as np
import librosa
import scipy.signal
from typing import Optional, Dict, Any
import random
import tempfile
import os
from pydub import AudioSegment
import soundfile as sf


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
        compression_prob: float = 0.0,  # Disabled: requires ffmpeg
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
        Apply real MP3 compression using pydub and ffmpeg.

        Compresses audio to MP3 format at specified bitrate and decodes back,
        introducing authentic lossy compression artifacts.

        Args:
            audio: Input audio signal (mono numpy array)
            quality: 'low' (128kbps) or 'medium' (192kbps) or 'high' (320kbps)

        Returns:
            Audio with MP3 compression artifacts

        Requires:
            ffmpeg must be installed and available in system PATH
        """
        if quality is None:
            quality = random.choice(['low', 'medium', 'high'])

        # Map quality levels to bitrates
        bitrate_map = {
            'low': '128k',      # 128 kbps
            'medium': '192k',   # 192 kbps
            'high': '320k'      # 320 kbps
        }
        bitrate = bitrate_map[quality]

        # Create temporary files for MP3 encoding/decoding
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                tmp_wav_path = tmp_wav.name
                tmp_mp3_path = tmp_mp3.name

        try:
            # Write audio to temporary WAV file
            sf.write(tmp_wav_path, audio, self.sr, subtype='PCM_16')

            # Load with pydub and export to MP3
            audio_segment = AudioSegment.from_wav(tmp_wav_path)
            audio_segment.export(
                tmp_mp3_path,
                format='mp3',
                bitrate=bitrate,
                parameters=['-q:a', '2']  # Good quality encoder settings
            )

            # Load compressed MP3 back as numpy array
            compressed_segment = AudioSegment.from_mp3(tmp_mp3_path)

            # Convert to numpy array
            samples = np.array(compressed_segment.get_array_of_samples(), dtype=np.float32)

            # Normalize to [-1, 1] range
            if compressed_segment.sample_width == 1:
                samples = samples / (2**7)
            elif compressed_segment.sample_width == 2:
                samples = samples / (2**15)
            elif compressed_segment.sample_width == 4:
                samples = samples / (2**31)

            # Handle stereo to mono conversion if needed
            if compressed_segment.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)

            # Resample if sample rate changed during compression
            if compressed_segment.frame_rate != self.sr:
                samples = librosa.resample(
                    samples,
                    orig_sr=compressed_segment.frame_rate,
                    target_sr=self.sr
                )

            # Ensure output length matches input (crop or pad)
            if len(samples) > len(audio):
                samples = samples[:len(audio)]
            elif len(samples) < len(audio):
                samples = np.pad(samples, (0, len(audio) - len(samples)), mode='constant')

            return samples

        finally:
            # Clean up temporary files
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)
            if os.path.exists(tmp_mp3_path):
                os.unlink(tmp_mp3_path)

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
                   Supports both flat format (pitch_shift_prob: 0.3) and
                   nested format (pitch_shift: {probability: 0.3, enabled: True})

        Returns:
            Augmented audio
        """
        if config is not None:
            # Parse config to override probabilities
            # Handle both flat and nested config formats
            for key, value in config.items():
                if isinstance(value, dict):
                    # Nested format: pitch_shift: {enabled: True, probability: 0.3}
                    if value.get('enabled', True):
                        prob_key = f"{key}_prob"
                        if 'probability' in value and hasattr(self, prob_key):
                            setattr(self, prob_key, value['probability'])
                elif key.endswith('_prob') and hasattr(self, key):
                    # Flat format: pitch_shift_prob: 0.3
                    setattr(self, key, value)
                elif key == 'max_transforms' and hasattr(self, 'max_augmentations'):
                    # Handle max_transforms alias
                    setattr(self, 'max_augmentations', value)
                elif hasattr(self, key):
                    # Direct attribute
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
