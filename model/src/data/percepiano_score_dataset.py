"""
PercePiano Dataset with Score Alignment for piano performance evaluation.

Extends the base PercePiano dataset to include score alignment features
that compare performance MIDI to reference MusicXML scores.

Score files location: PercePiano/virtuoso/data/all_2rounds/*.musicxml
"""

import json
import logging
import numpy as np
import pretty_midi
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from .midi_processing import OctupleMIDITokenizer
from .score_alignment import (
    MusicXMLParser,
    ScorePerformanceAligner,
    ScoreAlignmentFeatureExtractor,
    NUM_NOTE_FEATURES,
)

logger = logging.getLogger(__name__)

# All 19 PercePiano dimensions (in order matching percepiano_scores list)
PERCEPIANO_DIMENSIONS = [
    "timing",
    "articulation_length",
    "articulation_touch",
    "pedal_amount",
    "pedal_clarity",
    "timbre_variety",
    "timbre_depth",
    "timbre_brightness",
    "timbre_loudness",
    "dynamic_range",
    "tempo",
    "space",
    "balance",
    "drama",
    "mood_valence",
    "mood_energy",
    "mood_imagination",
    "sophistication",
    "interpretation",
]

# Custom 8 dimensions (alternative format)
CUSTOM_DIMENSIONS = [
    "timing_stability",
    "note_accuracy",
    "dynamic_range",
    "articulation",
    "pedal_technique",
    "expression",
    "tone_quality",
    "overall",
]


class PercePianoScoreDataset(Dataset):
    """
    PyTorch Dataset for PercePiano with score alignment features.

    Loads:
    - Performance MIDI files
    - Reference MusicXML scores
    - Expert annotations

    Extracts:
    - OctupleMIDI tokens from performance
    - Score alignment features (timing, dynamics, tempo deviations)
    """

    def __init__(
        self,
        data_file: Path,
        score_dir: Optional[Path] = None,
        max_midi_seq_length: int = 1024,
        max_score_notes: int = 1024,
        max_tempo_segments: int = 256,
        segment_seconds: float = 30.0,
        augment: bool = False,
        cache_midi: bool = True,
        cache_scores: bool = True,
        default_tempo: float = 120.0,
        use_percepiano_dims: bool = True,
    ):
        """
        Args:
            data_file: Path to JSON file with sample list
            score_dir: Directory containing MusicXML score files
            max_midi_seq_length: Maximum number of MIDI tokens per sample
            max_score_notes: Maximum number of score notes for alignment features
            max_tempo_segments: Maximum number of tempo curve segments
            segment_seconds: Target segment length in seconds
            augment: Whether to apply data augmentation
            cache_midi: Whether to cache loaded MIDI in memory
            cache_scores: Whether to cache parsed scores in memory
            default_tempo: Default tempo if not specified in score
            use_percepiano_dims: If True, use 19 PercePiano dimensions from
                percepiano_scores list. If False, use 8 custom dimensions from
                scores dict.
        """
        self.data_file = Path(data_file)
        self.score_dir = Path(score_dir) if score_dir else None
        self.max_midi_seq_length = max_midi_seq_length
        self.max_score_notes = max_score_notes
        self.max_tempo_segments = max_tempo_segments
        self.segment_seconds = segment_seconds
        self.augment = augment
        self.cache_midi = cache_midi
        self.cache_scores = cache_scores
        self.default_tempo = default_tempo
        self.use_percepiano_dims = use_percepiano_dims

        # Load sample list
        with open(self.data_file, "r") as f:
            self.samples = json.load(f)

        # Detect and set dimensions based on data format
        self._detect_dimensions()

        # Initialize components
        self.tokenizer = OctupleMIDITokenizer()
        self.feature_extractor = ScoreAlignmentFeatureExtractor()

        # Caches
        self._midi_cache: Dict[str, np.ndarray] = {}
        self._score_cache: Dict[str, Dict] = {}

        # Track score feature loading stats
        self._score_load_stats = {"loaded": 0, "empty": 0}

    def _detect_dimensions(self) -> None:
        """Detect which dimension format to use based on data."""
        if not self.samples:
            raise ValueError(f"No samples found in {self.data_file}")

        sample = self.samples[0]

        # Check if percepiano_scores list is available and valid
        has_percepiano = (
            "percepiano_scores" in sample
            and isinstance(sample["percepiano_scores"], list)
            and len(sample["percepiano_scores"]) >= 19
        )

        # Check if scores dict is available
        has_custom = "scores" in sample and isinstance(sample["scores"], dict)

        if self.use_percepiano_dims and has_percepiano:
            self.dimensions = PERCEPIANO_DIMENSIONS
            self._score_format = "percepiano"
            logger.info(
                f"Using 19 PercePiano dimensions from percepiano_scores list"
            )
        elif has_custom:
            # Use custom dimensions from scores dict
            self.dimensions = list(sample["scores"].keys())
            self._score_format = "custom"
            logger.info(
                f"Using {len(self.dimensions)} custom dimensions from scores dict: "
                f"{self.dimensions}"
            )
        elif has_percepiano:
            # Fallback to percepiano if custom requested but not available
            self.dimensions = PERCEPIANO_DIMENSIONS
            self._score_format = "percepiano"
            logger.warning(
                "Custom dimensions requested but scores dict not found. "
                "Falling back to PercePiano dimensions."
            )
        else:
            raise ValueError(
                f"No valid score format found in data. Sample keys: {sample.keys()}"
            )

        self.num_dimensions = len(self.dimensions)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load and tokenize performance MIDI
        midi_tokens = self._load_midi(sample["midi_path"])
        midi_tokens = self._prepare_midi_tokens(midi_tokens)

        # Load score alignment features
        score_features, score_loaded = self._get_score_features(sample)

        # Track score loading stats
        if score_loaded:
            self._score_load_stats["loaded"] += 1
        else:
            self._score_load_stats["empty"] += 1

        # Log periodically
        total = self._score_load_stats["loaded"] + self._score_load_stats["empty"]
        if total == 1 or total % 100 == 0:
            logger.info(
                f"Score feature loading: {self._score_load_stats['loaded']}/{total} "
                f"loaded ({100*self._score_load_stats['loaded']/total:.1f}%)"
            )

        # Get scores as tensor based on format
        scores = self._get_scores_tensor(sample)

        # Create attention masks
        midi_attention_mask = torch.ones(self.max_midi_seq_length, dtype=torch.float32)
        score_attention_mask = torch.ones(self.max_score_notes, dtype=torch.float32)

        # Get note_locations for hierarchical processing
        note_locations = score_features.get("note_locations")
        if note_locations is None:
            note_locations = self._get_empty_note_locations()

        return {
            "midi_tokens": midi_tokens,
            "midi_attention_mask": midi_attention_mask,
            "score_note_features": score_features["note_features"],
            "score_global_features": score_features["global_features"],
            "score_tempo_curve": score_features["tempo_curve"],
            "score_attention_mask": score_attention_mask,
            "note_locations_beat": note_locations["beat"],
            "note_locations_measure": note_locations["measure"],
            "note_locations_voice": note_locations["voice"],
            "scores": scores,
            "name": sample["name"],
        }

    def _get_score_features(
        self, sample: Dict
    ) -> Tuple[Dict[str, torch.Tensor], bool]:
        """Load score alignment features with tracking."""
        score_path = sample.get("score_path")
        if score_path and self.score_dir:
            full_score_path = self.score_dir / score_path
            if full_score_path.exists():
                try:
                    features = self._load_score_features(
                        sample["midi_path"],
                        full_score_path,
                        sample.get("tempo", self.default_tempo),
                    )
                    # Verify features are not all zeros
                    if self._features_are_valid(features):
                        return features, True
                    else:
                        logger.warning(
                            f"Score features are all zeros for {sample['name']}"
                        )
                        return self._get_empty_score_features(), False
                except Exception as e:
                    logger.warning(
                        f"Failed to load score features for {sample['name']}: {e}"
                    )
                    return self._get_empty_score_features(), False
            else:
                logger.debug(f"Score file not found: {full_score_path}")
        return self._get_empty_score_features(), False

    def _features_are_valid(self, features: Dict[str, torch.Tensor]) -> bool:
        """Check if score features are valid (not all zeros)."""
        note_features = features["note_features"]
        global_features = features["global_features"]

        # Check if note features have any non-zero values
        if isinstance(note_features, torch.Tensor):
            note_nonzero = note_features.abs().sum() > 0.01
        else:
            note_nonzero = np.abs(note_features).sum() > 0.01

        # Check if global features have any non-zero values
        if isinstance(global_features, torch.Tensor):
            global_nonzero = global_features.abs().sum() > 0.01
        else:
            global_nonzero = np.abs(global_features).sum() > 0.01

        return note_nonzero or global_nonzero

    def _get_scores_tensor(self, sample: Dict) -> torch.Tensor:
        """Get scores tensor based on the detected format."""
        if self._score_format == "percepiano":
            # Use percepiano_scores list (first 19 values)
            scores_list = sample["percepiano_scores"][:19]
            return torch.tensor(scores_list, dtype=torch.float32)
        else:
            # Use scores dict with custom dimensions
            scores_list = [
                sample["scores"].get(dim, 0.0) / 100.0  # Normalize to 0-1
                for dim in self.dimensions
            ]
            return torch.tensor(scores_list, dtype=torch.float32)

    def _load_midi(self, midi_path: str) -> np.ndarray:
        """Load and tokenize a MIDI file."""
        if self.cache_midi and midi_path in self._midi_cache:
            tokens = self._midi_cache[midi_path].copy()
        else:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            tokens = self.tokenizer.encode(midi_data)

            if self.cache_midi:
                self._midi_cache[midi_path] = tokens.copy()

        if self.augment:
            tokens = self._augment_midi(tokens)

        return tokens

    def _prepare_midi_tokens(self, tokens: np.ndarray) -> torch.Tensor:
        """Prepare MIDI tokens: truncate or pad to max length."""
        seq_len = len(tokens)

        if seq_len > self.max_midi_seq_length:
            if self.augment:
                start = np.random.randint(0, seq_len - self.max_midi_seq_length)
            else:
                start = (seq_len - self.max_midi_seq_length) // 2
            tokens = tokens[start : start + self.max_midi_seq_length]
        elif seq_len < self.max_midi_seq_length:
            padding = np.zeros((self.max_midi_seq_length - seq_len, 8), dtype=np.int32)
            tokens = np.concatenate([tokens, padding], axis=0)

        return torch.tensor(tokens, dtype=torch.long)

    def _load_score_features(
        self,
        midi_path: str,
        score_path: Path,
        tempo: float,
    ) -> Dict[str, torch.Tensor]:
        """Load and compute score alignment features."""
        cache_key = f"{midi_path}:{score_path}"

        if self.cache_scores and cache_key in self._score_cache:
            cached = self._score_cache[cache_key]
            return {
                "note_features": torch.tensor(cached["note_features"], dtype=torch.float32),
                "global_features": torch.tensor(cached["global_features"], dtype=torch.float32),
                "tempo_curve": torch.tensor(cached["tempo_curve"], dtype=torch.float32),
                "note_locations": {
                    "beat": torch.tensor(cached["note_locations"]["beat"], dtype=torch.long),
                    "measure": torch.tensor(cached["note_locations"]["measure"], dtype=torch.long),
                    "voice": torch.tensor(cached["note_locations"]["voice"], dtype=torch.long),
                },
            }

        # Load performance MIDI
        perf_midi = pretty_midi.PrettyMIDI(midi_path)

        # Extract features
        features = self.feature_extractor.extract_features(
            perf_midi,
            score_path,
            tempo_bpm=tempo,
        )

        # Prepare features
        note_features = self._prepare_note_features(features["note_features"])
        global_features = features["global_features"]
        tempo_curve = self._prepare_tempo_curve(features["tempo_curve"])
        note_locations = self._prepare_note_locations(features["note_locations"])

        # Cache
        if self.cache_scores:
            self._score_cache[cache_key] = {
                "note_features": note_features.numpy(),
                "global_features": global_features,
                "tempo_curve": tempo_curve.numpy(),
                "note_locations": {
                    "beat": note_locations["beat"].numpy(),
                    "measure": note_locations["measure"].numpy(),
                    "voice": note_locations["voice"].numpy(),
                },
            }

        return {
            "note_features": torch.tensor(note_features, dtype=torch.float32),
            "global_features": torch.tensor(global_features, dtype=torch.float32),
            "tempo_curve": torch.tensor(tempo_curve, dtype=torch.float32),
            "note_locations": note_locations,
        }

    def _prepare_note_features(self, note_features: np.ndarray) -> np.ndarray:
        """Prepare note features: truncate or pad to max length."""
        num_notes = len(note_features)

        if num_notes > self.max_score_notes:
            if self.augment:
                start = np.random.randint(0, num_notes - self.max_score_notes)
            else:
                start = (num_notes - self.max_score_notes) // 2
            note_features = note_features[start : start + self.max_score_notes]
        elif num_notes < self.max_score_notes:
            padding = np.zeros(
                (self.max_score_notes - num_notes, note_features.shape[1]),
                dtype=np.float32,
            )
            note_features = np.concatenate([note_features, padding], axis=0)

        return note_features

    def _prepare_tempo_curve(self, tempo_curve: np.ndarray) -> np.ndarray:
        """Prepare tempo curve: truncate or pad to max length."""
        num_segments = len(tempo_curve)

        if num_segments > self.max_tempo_segments:
            # Downsample to fit
            indices = np.linspace(0, num_segments - 1, self.max_tempo_segments, dtype=int)
            tempo_curve = tempo_curve[indices]
        elif num_segments < self.max_tempo_segments:
            padding = np.ones(self.max_tempo_segments - num_segments, dtype=np.float32)
            tempo_curve = np.concatenate([tempo_curve, padding], axis=0)

        return tempo_curve

    def _prepare_note_locations(
        self, note_locations: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Prepare note_locations: truncate or pad to max length."""
        num_notes = len(note_locations["beat"])
        prepared = {}

        for key in ["beat", "measure", "voice"]:
            arr = note_locations[key]

            if num_notes > self.max_score_notes:
                # Truncate (same logic as note_features)
                if self.augment:
                    start = np.random.randint(0, num_notes - self.max_score_notes)
                else:
                    start = (num_notes - self.max_score_notes) // 2
                arr = arr[start : start + self.max_score_notes]
            elif num_notes < self.max_score_notes:
                # Pad with zeros
                padding = np.zeros(self.max_score_notes - num_notes, dtype=arr.dtype)
                arr = np.concatenate([arr, padding], axis=0)

            prepared[key] = torch.tensor(arr, dtype=torch.long)

        return prepared

    def _get_empty_note_locations(self) -> Dict[str, torch.Tensor]:
        """Return empty note_locations when score is unavailable."""
        # Create linear indices - every 4 notes is a beat, every 16 is a measure
        indices = np.arange(self.max_score_notes)
        return {
            "beat": torch.tensor(indices // 4, dtype=torch.long),
            "measure": torch.tensor(indices // 16, dtype=torch.long),
            "voice": torch.ones(self.max_score_notes, dtype=torch.long),
        }

    def _get_empty_score_features(self) -> Dict[str, torch.Tensor]:
        """Return empty/neutral score features when score is unavailable."""
        return {
            "note_features": torch.zeros(self.max_score_notes, NUM_NOTE_FEATURES, dtype=torch.float32),
            "global_features": torch.zeros(12, dtype=torch.float32),
            "tempo_curve": torch.ones(self.max_tempo_segments, dtype=torch.float32),
            "note_locations": self._get_empty_note_locations(),
        }

    def _augment_midi(self, tokens: np.ndarray) -> np.ndarray:
        """Apply data augmentation to MIDI tokens."""
        tokens = tokens.copy()

        # Pitch shift
        if np.random.random() < 0.5:
            pitch_shift = np.random.randint(-2, 3)
            tokens[:, 3] = np.clip(tokens[:, 3] + pitch_shift, 0, 87)

        # Velocity scaling
        if np.random.random() < 0.5:
            vel_scale = np.random.uniform(0.8, 1.2)
            tokens[:, 5] = np.clip(tokens[:, 5] * vel_scale, 0, 127).astype(np.int32)

        # Time jitter
        if np.random.random() < 0.3:
            time_jitter = np.random.randint(-1, 2, size=len(tokens))
            tokens[:, 2] = np.clip(tokens[:, 2] + time_jitter, 0, 15)

        return tokens

    def get_dimension_names(self) -> List[str]:
        """Return ordered list of dimension names."""
        return self.dimensions.copy()


def create_score_dataloaders(
    data_dir: Path,
    score_dir: Optional[Path] = None,
    batch_size: int = 16,
    max_midi_seq_length: int = 1024,
    max_score_notes: int = 1024,
    num_workers: int = 4,
    use_percepiano_dims: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders with score alignment.

    Args:
        data_dir: Directory containing processed PercePiano JSON files
        score_dir: Directory containing MusicXML score files
        batch_size: Batch size for training
        max_midi_seq_length: Maximum MIDI sequence length
        max_score_notes: Maximum number of score notes
        num_workers: Number of data loading workers
        use_percepiano_dims: If True, use 19 PercePiano dimensions.
            If False, use custom dimensions from scores dict.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = PercePianoScoreDataset(
        data_file=data_dir / "percepiano_train.json",
        score_dir=score_dir,
        max_midi_seq_length=max_midi_seq_length,
        max_score_notes=max_score_notes,
        augment=True,
        cache_midi=True,
        cache_scores=True,
        use_percepiano_dims=use_percepiano_dims,
    )

    val_dataset = PercePianoScoreDataset(
        data_file=data_dir / "percepiano_val.json",
        score_dir=score_dir,
        max_midi_seq_length=max_midi_seq_length,
        max_score_notes=max_score_notes,
        augment=False,
        cache_midi=True,
        cache_scores=True,
        use_percepiano_dims=use_percepiano_dims,
    )

    test_dataset = PercePianoScoreDataset(
        data_file=data_dir / "percepiano_test.json",
        score_dir=score_dir,
        max_midi_seq_length=max_midi_seq_length,
        max_score_notes=max_score_notes,
        augment=False,
        cache_midi=True,
        cache_scores=True,
        use_percepiano_dims=use_percepiano_dims,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("PercePiano Score Dataset module loaded successfully")
    print("Features:")
    print("- Performance MIDI tokenization (OctupleMIDI)")
    print("- Score alignment feature extraction")
    print("- Caching for MIDI and score features")
    print("- Data augmentation support")
