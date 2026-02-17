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

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm.auto import tqdm


class MuQExtractor:
    """MuQ feature extractor with optional layer selection.

    Extracts embeddings from the MuQ model, which outputs hidden states
    similar to MERT. Can extract from last hidden state, average across
    a range of layers, or concatenate layers for higher-dimensional output.
    """

    def __init__(
        self,
        layer_start: Optional[int] = None,
        layer_end: Optional[int] = None,
        layer_aggregation: str = "mean",
        cache_dir: Optional[Path] = None,
    ):
        """Initialize MuQ extractor.

        Args:
            layer_start: Starting layer index for aggregation (inclusive).
                If None, uses last_hidden_state directly.
            layer_end: Ending layer index for aggregation (exclusive).
                If None, uses last_hidden_state directly.
            layer_aggregation: How to aggregate layers when using layer range.
                "mean": Average across layers (output dim = 1024).
                "concat": Concatenate layers (output dim = 1024 * n_layers).
            cache_dir: Directory to cache extracted embeddings.
        """
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.layer_aggregation = layer_aggregation
        self.use_layer_range = layer_start is not None and layer_end is not None
        self.target_sr = 24000  # MuQ uses 24kHz like MERT
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Device selection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self._load_model()

    def _load_model(self):
        """Load MuQ model from HuggingFace."""
        try:
            from muq import MuQ
        except ImportError as e:
            raise ImportError(
                "MuQ library not found. Install with: pip install muq"
            ) from e

        if self.use_layer_range:
            agg_info = f"{self.layer_aggregation} of layers {self.layer_start}-{self.layer_end - 1}"
        else:
            agg_info = "last hidden state"
        print(f"Loading MuQ-large-msd-iter ({agg_info}) on {self.device}...")

        self.model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        self.model = self.model.to(self.device)

        # Use FP16 on CUDA for ~2x speedup
        # MPS segfaults with FP16 on many transformer ops
        if self.device.type == "cuda":
            self.model = self.model.half()

        self.model.eval()

        # Compile model for additional speedup (PyTorch 2.x, CUDA only)
        # MPS Metal shaders fail with torch.compile on many ops
        if hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled with torch.compile()")
            except Exception:
                pass  # Compilation not supported for this model/device

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

    # 30 seconds at 24kHz — keeps MPS memory well under limit
    MAX_CHUNK_SAMPLES = 30 * 24000
    # MuQ's STFT uses n_fft=2048 with padding of 1024 on each side
    MIN_CHUNK_SAMPLES = 2048

    def _extract_chunk(self, wavs: torch.Tensor) -> torch.Tensor:
        """Run model on a single [1, samples] tensor, return [T, hidden_size] on CPU."""
        output = self.model(wavs, output_hidden_states=True)

        if self.use_layer_range:
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

            hidden_states = output.hidden_states[self.layer_start : self.layer_end]

            if self.layer_aggregation == "concat":
                return torch.cat(hidden_states, dim=-1).squeeze(0).cpu()
            else:
                return torch.stack(hidden_states, dim=0).mean(dim=0).squeeze(0).cpu()
        else:
            return output.last_hidden_state.squeeze(0).cpu()

    @torch.no_grad()
    def extract_from_file(self, audio_path: Path, use_cache: bool = True) -> torch.Tensor:
        """Extract MuQ embeddings from an audio file.

        Long files are processed in 30-second chunks to avoid OOM on MPS/GPU,
        then the frame-level embeddings are concatenated.

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

        # Load audio with soundfile (torchaudio 2.9 has broken torchcodec default)
        data, sr = sf.read(audio_path, dtype="float32")
        # soundfile returns (samples, channels) numpy array
        audio = torch.from_numpy(data)
        if audio.dim() == 2:
            audio = audio.mean(dim=1)  # Convert to mono
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if audio.shape[0] <= self.MAX_CHUNK_SAMPLES:
            # Short file — single forward pass
            wavs = audio.unsqueeze(0).to(self.device, dtype=dtype)
            embeddings = self._extract_chunk(wavs)
        else:
            # Long file — process in chunks and concatenate
            chunks = audio.split(self.MAX_CHUNK_SAMPLES)
            parts = []
            for chunk in chunks:
                if chunk.shape[0] < self.MIN_CHUNK_SAMPLES:
                    continue  # skip runt tail chunk — too short for STFT
                wavs = chunk.unsqueeze(0).to(self.device, dtype=dtype)
                parts.append(self._extract_chunk(wavs))
                del wavs
            embeddings = torch.cat(parts, dim=0)

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

        # Match model dtype (FP16 on GPU/MPS)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        wavs = audio.to(self.device, dtype=dtype)
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

            if self.layer_aggregation == "concat":
                # Concatenate layers: [B, T, 1024 * n_layers]
                embeddings = torch.cat(hidden_states, dim=-1).cpu()
            else:
                # Average across specified layer range: [B, T, 1024]
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
    layer_aggregation: str = "mean",
) -> int:
    """Extract MuQ embeddings for a list of audio files.

    Args:
        audio_dir: Directory containing audio files.
        cache_dir: Directory to cache embeddings.
        keys: List of audio file keys (without extension).
        layer_start: Starting layer index for aggregation (optional).
        layer_end: Ending layer index for aggregation (optional).
        layer_aggregation: How to aggregate layers ("mean" or "concat").

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

    if layer_start is not None and layer_end is not None:
        layer_info = f"{layer_aggregation} of layers {layer_start}-{layer_end - 1}"
    else:
        layer_info = "last hidden state"
    print(f"Extracting {len(to_extract)} MuQ embeddings ({layer_info})...")

    extractor = MuQExtractor(layer_start, layer_end, layer_aggregation, cache_dir)

    for key in tqdm(to_extract, desc="MuQ extraction"):
        audio_path = Path(audio_dir) / f"{key}.wav"
        if audio_path.exists():
            extractor.extract_from_file(audio_path)

    del extractor
    torch.cuda.empty_cache()
    return len(to_extract)
