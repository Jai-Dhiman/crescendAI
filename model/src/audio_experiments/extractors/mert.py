"""MERT-330M feature extractor with configurable layer selection."""

from pathlib import Path
from typing import List, Optional

import librosa
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor


class MERTLayerExtractor:
    """MERT-330M extractor with configurable layer range."""

    def __init__(
        self,
        layer_start: int = 13,
        layer_end: int = 25,
        cache_dir: Optional[Path] = None,
    ):
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.target_sr = 24000
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading MERT-v1-330M (layers {layer_start}-{layer_end-1}) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M",
            output_hidden_states=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        print(f"Model loaded. Hidden size: {self.model.config.hidden_size}")

    def get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pt"

    @torch.no_grad()
    def extract_from_file(self, audio_path: Path, use_cache: bool = True) -> torch.Tensor:
        audio_path = Path(audio_path)
        key = audio_path.stem

        if use_cache and self.cache_dir:
            cache_path = self.get_cache_path(key)
            if cache_path.exists():
                return torch.load(cache_path, weights_only=True)

        audio, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)
        inputs = self.processor(audio, sampling_rate=self.target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states[self.layer_start : self.layer_end]
        embeddings = torch.stack(hidden_states, dim=0).mean(dim=0).squeeze(0).cpu()

        if use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings, self.get_cache_path(key))

        return embeddings


def extract_mert_for_layer_range(
    layer_start: int,
    layer_end: int,
    audio_dir: Path,
    cache_dir: Path,
    keys: List[str],
) -> int:
    """Extract MERT embeddings for a specific layer range.

    Returns count of newly extracted embeddings.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = {p.stem for p in cache_dir.glob("*.pt")}
    to_extract = [k for k in keys if k not in cached]

    if not to_extract:
        print(f"All {len(keys)} embeddings already cached.")
        return 0

    print(f"Extracting {len(to_extract)} embeddings (layers {layer_start}-{layer_end-1})...")
    extractor = MERTLayerExtractor(layer_start, layer_end, cache_dir)

    for key in tqdm(to_extract, desc=f"MERT L{layer_start}-{layer_end-1}"):
        audio_path = Path(audio_dir) / f"{key}.wav"
        if audio_path.exists():
            extractor.extract_from_file(audio_path)

    del extractor
    torch.cuda.empty_cache()
    return len(to_extract)
