"""
MERT-330M Feature Extraction for Audio.

Extracts embeddings from the MERT (Music Understanding Model) for piano audio,
using weighted average of layers 12-24 following SUPERB/MARBLE protocols.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor


class MERT330MExtractor:
    """
    Extract embeddings from MERT-330M for audio segments.

    Uses average of layers 12-24 following SUPERB/MARBLE protocols.
    Supports caching embeddings to disk to avoid re-extraction.

    Attributes:
        model: MERT-330M model
        processor: MERT audio processor
        device: Device to run model on
        target_sr: Target sample rate (24000 for MERT)
        use_layers: Tuple of (start, end) layer indices to average
        hidden_size: Embedding dimension (1024 for MERT-330M)
        cache_dir: Optional directory for caching embeddings
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-330M",
        device: str = "auto",
        cache_dir: Optional[Path] = None,
        target_sr: int = 24000,
        use_layers: Tuple[int, int] = (12, 25),
    ):
        """
        Initialize MERT extractor.

        Args:
            model_name: HuggingFace model name
            device: Device ("auto", "cuda", "cpu", or specific GPU)
            cache_dir: Optional directory for caching embeddings
            target_sr: Target sample rate (24000 is MERT native)
            use_layers: Tuple of (start, end) layer indices to average (0-indexed, exclusive end)
        """
        self.target_sr = target_sr
        self.use_layers = use_layers
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading MERT model: {model_name}")
        print(f"Device: {self.device}")

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        # Get hidden size
        self.hidden_size = self.model.config.hidden_size
        print(f"Hidden size: {self.hidden_size}")
        print(f"Using layers: {use_layers[0]}-{use_layers[1]-1}")

    def load_audio(self, audio_path: Path) -> torch.Tensor:
        """
        Load and resample audio to target sample rate.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio tensor [num_samples]
        """
        audio, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)
        return torch.from_numpy(audio).float()

    @torch.no_grad()
    def extract(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract MERT embeddings from audio tensor.

        Args:
            audio: Audio tensor [num_samples] at target_sr

        Returns:
            Embeddings tensor [num_frames, hidden_size]
        """
        # Process audio
        inputs = self.processor(
            audio.numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)

        # Get hidden states from specified layers
        hidden_states = outputs.hidden_states[self.use_layers[0] : self.use_layers[1]]

        # Stack and average across layers
        stacked = torch.stack(hidden_states, dim=0)  # [num_layers, B, T, H]
        embeddings = stacked.mean(dim=0).squeeze(0)  # [T, H]

        return embeddings.cpu()

    def extract_from_file(
        self,
        audio_path: Path,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Extract embeddings from audio file, with optional caching.

        Args:
            audio_path: Path to audio file
            use_cache: Whether to use/save cached embeddings

        Returns:
            Embeddings tensor [num_frames, hidden_size]
        """
        audio_path = Path(audio_path)

        # Check cache
        if use_cache and self.cache_dir is not None:
            cache_path = self.cache_dir / f"{audio_path.stem}.pt"
            if cache_path.exists():
                return torch.load(cache_path, weights_only=True)

        # Load and extract
        audio = self.load_audio(audio_path)
        embeddings = self.extract(audio)

        # Cache
        if use_cache and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings, cache_path)

        return embeddings


def batch_extract_mert(
    audio_dir: Path,
    cache_dir: Path,
    extractor: Optional[MERT330MExtractor] = None,
    skip_existing: bool = True,
) -> Tuple[int, int]:
    """
    Batch extract MERT embeddings for all audio files.

    Args:
        audio_dir: Directory containing WAV files
        cache_dir: Directory for caching embeddings
        extractor: Optional pre-initialized extractor (creates new if None)
        skip_existing: Skip files that already have cached embeddings

    Returns:
        Tuple of (successful, failed) counts
    """
    audio_dir = Path(audio_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor if not provided
    if extractor is None:
        extractor = MERT330MExtractor(cache_dir=cache_dir)

    # Get audio files
    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"Audio files found: {len(audio_files)}")

    # Filter to files needing extraction
    if skip_existing:
        to_extract = [
            f for f in audio_files if not (cache_dir / f"{f.stem}.pt").exists()
        ]
        already_cached = len(audio_files) - len(to_extract)
        print(f"Already cached: {already_cached}")
    else:
        to_extract = audio_files

    print(f"Files to extract: {len(to_extract)}")

    if not to_extract:
        return len(audio_files), 0

    # Extract embeddings
    successful = len(audio_files) - len(to_extract)  # Count already cached
    failed: List[Tuple[str, str]] = []

    for audio_path in tqdm(to_extract, desc="Extracting MERT embeddings"):
        try:
            extractor.extract_from_file(audio_path, use_cache=True)
            successful += 1
        except Exception as e:
            failed.append((audio_path.stem, str(e)))

    # Report failures
    if failed:
        print(f"\nFailed files: {len(failed)}")
        for name, error in failed[:10]:
            print(f"  {name}: {error}")

    return successful, len(failed)


def get_embedding_stats(cache_dir: Path) -> dict:
    """
    Get statistics about cached embeddings.

    Args:
        cache_dir: Directory containing cached embeddings

    Returns:
        Dictionary with statistics
    """
    cache_dir = Path(cache_dir)
    cached_files = list(cache_dir.glob("*.pt"))

    if not cached_files:
        return {"count": 0}

    # Sample a few files for stats
    sample = torch.load(cached_files[0], weights_only=True)

    # Get frame counts
    frame_counts = []
    for f in cached_files[:100]:  # Sample first 100
        emb = torch.load(f, weights_only=True)
        frame_counts.append(emb.shape[0])

    return {
        "count": len(cached_files),
        "hidden_size": sample.shape[1],
        "sample_frames": sample.shape[0],
        "mean_frames": sum(frame_counts) / len(frame_counts),
        "min_frames": min(frame_counts),
        "max_frames": max(frame_counts),
    }
