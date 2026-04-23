"""GPU-accelerated mel spectrogram extraction using nnAudio."""

from pathlib import Path
from typing import List, Optional

import librosa
import torch
from nnAudio.features import MelSpectrogram
from tqdm.auto import tqdm


class MelExtractor:
    """GPU-accelerated mel spectrogram extraction using nnAudio."""

    def __init__(self, cache_dir: Optional[Path] = None, sr: int = 24000):
        self.sr = sr
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mel_spec = MelSpectrogram(
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            fmin=20,
            fmax=8000,
            trainable_mel=False,
            trainable_STFT=False,
        ).to(self.device)

    def extract_from_file(self, audio_path: Path, use_cache: bool = True) -> torch.Tensor:
        audio_path = Path(audio_path)
        key = audio_path.stem

        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{key}.pt"
            if cache_path.exists():
                return torch.load(cache_path, weights_only=True)

        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            mel = self.mel_spec(audio_tensor)  # [1, n_mels, time]
            mel = mel.squeeze(0).cpu()  # [n_mels, time]

        if use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(mel, self.cache_dir / f"{key}.pt")

        return mel


def extract_mel_spectrograms(
    audio_dir: Path,
    cache_dir: Path,
    keys: List[str],
) -> int:
    """Extract mel spectrograms for all keys.

    Returns count of newly extracted spectrograms.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = {p.stem for p in cache_dir.glob("*.pt")}
    to_extract = [k for k in keys if k not in cached]

    if not to_extract:
        print(f"All {len(keys)} mel spectrograms already cached.")
        return 0

    print(f"Extracting {len(to_extract)} mel spectrograms...")
    extractor = MelExtractor(cache_dir)

    for key in tqdm(to_extract, desc="Mel extraction"):
        audio_path = Path(audio_dir) / f"{key}.wav"
        if audio_path.exists():
            extractor.extract_from_file(audio_path)

    return len(to_extract)
