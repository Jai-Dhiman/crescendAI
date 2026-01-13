"""MuQ (Music Understanding Quantized) feature extractor.

MuQ is a music understanding model from ByteDance/OpenMuQ that uses quantized
representations for music audio. Similar to MERT but with different training
objectives.

Model: OpenMuQ/MuQ-large-msd-iter (~300M params)
Sample rate: 24,000 Hz
Hidden size: 1024 (same as MERT-330M)
"""

from pathlib import Path
from typing import List, Optional

import librosa
import torch
from tqdm.auto import tqdm


class MuQExtractor:
    """MuQ feature extractor with optional layer selection.

    Extracts embeddings from the MuQ model, which outputs hidden states
    similar to MERT. Can extract from last hidden state or average across
    a range of layers.
    """

    def __init__(
        self,
        layer_start: Optional[int] = None,
        layer_end: Optional[int] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize MuQ extractor.

        Args:
            layer_start: Starting layer index for averaging (inclusive).
                If None, uses last_hidden_state directly.
            layer_end: Ending layer index for averaging (exclusive).
                If None, uses last_hidden_state directly.
            cache_dir: Directory to cache extracted embeddings.
        """
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.use_layer_range = layer_start is not None and layer_end is not None
        self.target_sr = 24000  # MuQ uses 24kHz like MERT
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model()

    def _load_model(self):
        """Load MuQ model from HuggingFace."""
        try:
            from muq import MuQ
        except ImportError as e:
            raise ImportError(
                "MuQ library not found. Install with: pip install muq"
            ) from e

        layer_info = (
            f"layers {self.layer_start}-{self.layer_end - 1}"
            if self.use_layer_range
            else "last hidden state"
        )
        print(f"Loading MuQ-large-msd-iter ({layer_info}) on {self.device}...")

        self.model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get hidden size from model config
        self.hidden_size = getattr(self.model.config, "hidden_size", 1024)

        # Get number of layers from config
        self.num_layers = getattr(self.model.config, "num_hidden_layers", None)
        layer_info_str = f", {self.num_layers} layers" if self.num_layers else ""
        print(f"Model loaded. Hidden size: {self.hidden_size}{layer_info_str}")

        # Validate layer range against model architecture if we know num_layers
        if self.use_layer_range and self.num_layers is not None:
            # hidden_states has num_layers + 1 elements (includes initial embedding)
            max_layer_idx = self.num_layers + 1
            if self.layer_end > max_layer_idx:
                raise ValueError(
                    f"Requested layers {self.layer_start}-{self.layer_end - 1} but MuQ only has "
                    f"{self.num_layers} transformer layers (hidden_states indices 0-{self.num_layers}). "
                    f"Use layer_end <= {max_layer_idx}."
                )

    def get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        if self.cache_dir is None:
            raise ValueError("cache_dir not set")
        return self.cache_dir / f"{key}.pt"

    @torch.no_grad()
    def extract_from_file(self, audio_path: Path, use_cache: bool = True) -> torch.Tensor:
        """Extract MuQ embeddings from an audio file.

        Args:
            audio_path: Path to audio file.
            use_cache: Whether to use cached embeddings if available.

        Returns:
            Tensor of shape [time_steps, hidden_size].
        """
        audio_path = Path(audio_path)
        key = audio_path.stem

        # Check cache
        if use_cache and self.cache_dir:
            cache_path = self.get_cache_path(key)
            if cache_path.exists():
                return torch.load(cache_path, weights_only=True)

        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)
        wavs = torch.tensor(audio).unsqueeze(0).to(self.device)

        # Extract embeddings
        output = self.model(wavs, output_hidden_states=True)

        if self.use_layer_range:
            # Validate hidden_states exists and has expected layers
            if output.hidden_states is None:
                raise RuntimeError(
                    "MuQ model did not return hidden_states. "
                    "Ensure the model supports output_hidden_states=True."
                )

            num_layers = len(output.hidden_states)
            if self.layer_end > num_layers:
                raise RuntimeError(
                    f"Requested layers {self.layer_start}-{self.layer_end - 1} but MuQ only has "
                    f"{num_layers} hidden states (indices 0-{num_layers - 1}). "
                    f"Use layer_end <= {num_layers}."
                )

            # Average across specified layer range
            hidden_states = output.hidden_states[self.layer_start : self.layer_end]
            embeddings = torch.stack(hidden_states, dim=0).mean(dim=0).squeeze(0).cpu()
        else:
            # Use last hidden state directly
            embeddings = output.last_hidden_state.squeeze(0).cpu()

        # Cache embeddings
        if use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings, self.get_cache_path(key))

        return embeddings

    @torch.no_grad()
    def extract_from_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract MuQ embeddings from audio tensor.

        Args:
            audio: Audio tensor of shape [samples] or [batch, samples] at 24kHz.

        Returns:
            Tensor of shape [time_steps, hidden_size] or [batch, time_steps, hidden_size].
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        wavs = audio.to(self.device)
        output = self.model(wavs, output_hidden_states=True)

        if self.use_layer_range:
            # Validate layer range (should be caught at init, but double-check)
            if output.hidden_states is None:
                raise RuntimeError("MuQ model did not return hidden_states.")
            num_layers = len(output.hidden_states)
            if self.layer_end > num_layers:
                raise RuntimeError(
                    f"Requested layers {self.layer_start}-{self.layer_end - 1} but MuQ only has "
                    f"{num_layers} hidden states (indices 0-{num_layers - 1})."
                )

            hidden_states = output.hidden_states[self.layer_start : self.layer_end]
            embeddings = torch.stack(hidden_states, dim=0).mean(dim=0).cpu()
        else:
            embeddings = output.last_hidden_state.cpu()

        return embeddings.squeeze(0) if audio.shape[0] == 1 else embeddings


def extract_muq_embeddings(
    audio_dir: Path,
    cache_dir: Path,
    keys: List[str],
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
) -> int:
    """Extract MuQ embeddings for a list of audio files.

    Args:
        audio_dir: Directory containing audio files.
        cache_dir: Directory to cache embeddings.
        keys: List of audio file keys (without extension).
        layer_start: Starting layer index for averaging (optional).
        layer_end: Ending layer index for averaging (optional).

    Returns:
        Count of newly extracted embeddings.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = {p.stem for p in cache_dir.glob("*.pt")}
    to_extract = [k for k in keys if k not in cached]

    if not to_extract:
        print(f"All {len(keys)} MuQ embeddings already cached.")
        return 0

    layer_info = (
        f"layers {layer_start}-{layer_end - 1}"
        if layer_start is not None and layer_end is not None
        else "last hidden state"
    )
    print(f"Extracting {len(to_extract)} MuQ embeddings ({layer_info})...")

    extractor = MuQExtractor(layer_start, layer_end, cache_dir)

    for key in tqdm(to_extract, desc="MuQ extraction"):
        audio_path = Path(audio_dir) / f"{key}.wav"
        if audio_path.exists():
            extractor.extract_from_file(audio_path)

    del extractor
    torch.cuda.empty_cache()
    return len(to_extract)
