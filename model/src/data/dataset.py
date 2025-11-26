import torch
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset, DataLoader

from .audio_processing import load_audio, segment_audio, normalize_audio
from .midi_processing import load_midi, encode_octuple_midi, segment_midi, align_midi_to_audio
from .augmentation import AudioAugmentation
from .degradation import (
    degrade_audio_quality,
    degrade_midi_timing,
    inject_wrong_notes,
    compress_midi_dynamics,
)

# Import pretty_midi for type hints in degradation
try:
    import pretty_midi
except ImportError:
    pretty_midi = None


class PerformanceDataset(Dataset):
    """
    Dataset for piano performance evaluation.

    Loads audio, MIDI, and labels from annotation files.
    Supports both pseudo-labels (MAESTRO) and expert labels.

    Annotation format (JSONL):
    {
        "audio_path": "path/to/audio.wav",
        "midi_path": "path/to/score.mid",
        "start_time": 0.0,
        "end_time": 10.0,
        "labels": {
            "note_accuracy": 85.2,
            "rhythmic_precision": 78.5,
            ...
        }
    }
    """

    def __init__(
        self,
        annotation_path: str,
        dimension_names: List[str],
        audio_sample_rate: int = 24000,
        max_audio_length: int = 240000,  # 10 seconds at 24kHz
        max_midi_events: int = 512,
        augmentation_config: Optional[Dict[str, Any]] = None,
        apply_augmentation: bool = False,
        modality_dropout_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Performance Dataset.

        Args:
            annotation_path: Path to JSONL annotation file
            dimension_names: List of evaluation dimension names (e.g., ['note_accuracy', ...])
            audio_sample_rate: Audio sampling rate (24kHz for MERT)
            max_audio_length: Maximum audio length in samples (for padding/truncation)
            max_midi_events: Maximum MIDI events (for padding/truncation)
            augmentation_config: Augmentation configuration dict (see augmentation.py)
            apply_augmentation: Whether to apply augmentation (train only)
            modality_dropout_config: Config for modality dropout (train only)
                {
                    'enabled': True,
                    'audio_prob': 0.15,  # Probability of dropping audio
                    'midi_prob': 0.15,   # Probability of dropping MIDI
                }
        """
        super().__init__()

        self.annotation_path = Path(annotation_path)
        self.dimension_names = dimension_names
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_length = max_audio_length
        self.max_midi_events = max_midi_events
        self.augmentation_config = augmentation_config
        self.apply_augmentation = apply_augmentation

        # Modality dropout configuration
        self.modality_dropout_config = modality_dropout_config or {}
        self.modality_dropout_enabled = self.modality_dropout_config.get('enabled', False)
        self.audio_dropout_prob = self.modality_dropout_config.get('audio_prob', 0.15)
        self.midi_dropout_prob = self.modality_dropout_config.get('midi_prob', 0.15)

        # Initialize augmentation pipeline if enabled
        self.augmentor = None
        if self.apply_augmentation:
            self.augmentor = AudioAugmentation(sr=self.audio_sample_rate)

        # Load annotations
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load annotations from JSONL file."""
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")

        annotations = []
        with open(self.annotation_path, 'r') as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line))

        if len(annotations) == 0:
            raise ValueError(f"No annotations found in {self.annotation_path}")

        return annotations

    def _apply_audio_degradation(
        self,
        audio: np.ndarray,
        degradation_params: Dict[str, Any],
        seed: int
    ) -> np.ndarray:
        """
        Apply audio degradation based on annotation parameters.

        Args:
            audio: Audio waveform (normalized)
            degradation_params: Dict with 'audio_noise_snr_db' and 'audio_filter_enabled'
            seed: Random seed for reproducibility

        Returns:
            Degraded audio waveform
        """
        noise_snr_db = degradation_params.get('audio_noise_snr_db')
        apply_filter = degradation_params.get('audio_filter_enabled', False)

        # Skip if no degradation needed
        if noise_snr_db is None and not apply_filter:
            return audio

        return degrade_audio_quality(
            audio,
            sr=self.audio_sample_rate,
            noise_snr_db=noise_snr_db,
            apply_filtering=apply_filter,
            seed=seed
        )

    def _apply_midi_degradation(
        self,
        midi_obj: 'pretty_midi.PrettyMIDI',
        degradation_params: Dict[str, Any],
        seed: int
    ) -> 'pretty_midi.PrettyMIDI':
        """
        Apply MIDI degradation based on annotation parameters.

        Args:
            midi_obj: PrettyMIDI object
            degradation_params: Dict with midi_jitter_ms, wrong_note_rate, dynamics_compression
            seed: Random seed for reproducibility

        Returns:
            Degraded PrettyMIDI object
        """
        degraded = midi_obj

        # 1. Timing jitter
        jitter_ms = degradation_params.get('midi_jitter_ms', 0)
        if jitter_ms > 0:
            degraded = degrade_midi_timing(degraded, jitter_ms, seed=seed)

        # 2. Wrong notes
        wrong_note_rate = degradation_params.get('wrong_note_rate', 0.0)
        if wrong_note_rate > 0:
            degraded = inject_wrong_notes(
                degraded,
                wrong_note_rate,
                seed=seed + 1 if seed is not None else None
            )

        # 3. Dynamics compression
        compression = degradation_params.get('dynamics_compression', 0.0)
        if compression > 0:
            degraded = compress_midi_dynamics(degraded, compression)

        return degraded

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - audio_waveform: Raw audio [num_samples]
                - midi_tokens: OctupleMIDI tokens [num_events, 8] (if MIDI available)
                - labels: Ground truth scores [num_dimensions]
                - metadata: Additional info (paths, times, etc.)
        """
        return self._load_sample_with_retry(idx, max_retries=5)

    def _load_sample_with_retry(self, idx: int, max_retries: int = 5) -> Dict[str, torch.Tensor]:
        """Load a sample with retry logic for corrupted/unreadable files."""
        tried_indices = set()

        for attempt in range(max_retries):
            current_idx = idx if attempt == 0 else random.randint(0, len(self) - 1)

            # Avoid retrying same index
            if current_idx in tried_indices:
                continue
            tried_indices.add(current_idx)

            try:
                return self._load_sample(current_idx)
            except (OSError, IOError) as e:
                print(f"\nWarning: Failed to load sample {current_idx} (attempt {attempt + 1}/{max_retries})")
                print(f"  Error: {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to load any valid sample after {max_retries} attempts. "
                        f"Last error: {e}"
                    )

        raise RuntimeError(f"Exhausted retries loading sample {idx}")

    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample (internal method)."""
        annotation = self.annotations[idx]

        # Load audio
        audio_path = Path(annotation['audio_path'])
        audio, sr = load_audio(str(audio_path), sr=self.audio_sample_rate)

        # Extract segment if start/end times specified
        if 'start_time' in annotation and 'end_time' in annotation:
            start_sample = int(annotation['start_time'] * sr)
            end_sample = int(annotation['end_time'] * sr)
            audio = audio[start_sample:end_sample]

        # Normalize audio
        audio = normalize_audio(audio)

        # Apply runtime degradation if specified in annotation
        # This creates quality variance (pristine/good/moderate/poor tiers)
        if 'degradation_params' in annotation:
            audio = self._apply_audio_degradation(
                audio,
                annotation['degradation_params'],
                seed=idx
            )

        # Apply augmentation if enabled
        if self.apply_augmentation and self.augmentor is not None:
            audio = self.augmentor.augment_pipeline(audio, config=self.augmentation_config)

        # Pad or truncate audio to max length
        if len(audio) > self.max_audio_length:
            audio = audio[:self.max_audio_length]
        elif len(audio) < self.max_audio_length:
            audio = np.pad(audio, (0, self.max_audio_length - len(audio)))

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Load MIDI if available
        midi_tokens = None
        if 'midi_path' in annotation and annotation['midi_path']:
            midi_path = Path(annotation['midi_path'])
            try:
                midi = load_midi(str(midi_path))

                # Align MIDI to audio duration
                audio_duration = len(audio) / sr
                midi = align_midi_to_audio(midi, audio_duration)

                # Extract segment if needed
                if 'start_time' in annotation and 'end_time' in annotation:
                    start_time = annotation['start_time']
                    end_time = annotation['end_time']
                    segment_times = [(start_time, end_time)]
                    midi_segments = segment_midi(midi, segment_times)
                    midi = midi_segments[0] if midi_segments else midi

                # Apply runtime degradation to MIDI if specified
                # Degrades timing precision, adds wrong notes, compresses dynamics
                if 'degradation_params' in annotation:
                    midi = self._apply_midi_degradation(
                        midi,
                        annotation['degradation_params'],
                        seed=idx
                    )

                # Encode MIDI to OctupleMIDI tokens
                midi_tokens = encode_octuple_midi(midi)

                # Validate tokens before padding
                if len(midi_tokens) == 0:
                    # Empty MIDI file - log warning and fall back to audio-only
                    print(f"Warning: Empty MIDI file at {midi_path}, falling back to audio-only")
                    midi_tokens = None
                else:
                    # Pad or truncate MIDI tokens
                    if len(midi_tokens) > self.max_midi_events:
                        midi_tokens = midi_tokens[:self.max_midi_events]
                    elif len(midi_tokens) < self.max_midi_events:
                        # Pad with zeros
                        padding = np.zeros((self.max_midi_events - len(midi_tokens), 8), dtype=np.int64)
                        midi_tokens = np.concatenate([midi_tokens, padding], axis=0)

                    midi_tokens = torch.from_numpy(midi_tokens).long()

            except (ValueError, FileNotFoundError, OSError, KeyError, IndexError) as e:
                # Specific exceptions for MIDI loading failures:
                # - ValueError: divide-by-zero from align_midi_to_audio, invalid MIDI data
                # - FileNotFoundError: MIDI file doesn't exist
                # - OSError: file I/O errors (permissions, disk issues)
                # - KeyError: missing required fields in MIDI
                # - IndexError: malformed MIDI data structures
                # Note: Let other exceptions propagate to catch unexpected errors
                print(f"\nWarning: Failed to load MIDI for sample {idx}")
                print(f"  Path: {midi_path}")
                print(f"  Error: {type(e).__name__}: {e}")
                midi_tokens = None

        # Extract labels - all dimensions must be present
        labels_dict = annotation['labels']
        labels = []
        for dim_name in self.dimension_names:
            if dim_name not in labels_dict:
                raise KeyError(
                    f"Missing required dimension '{dim_name}' in sample {idx} "
                    f"(audio: {annotation.get('audio_path', 'unknown')}). "
                    f"Available dimensions: {list(labels_dict.keys())}"
                )
            labels.append(labels_dict[dim_name])

        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        # Metadata
        metadata = {
            'audio_path': str(audio_path),
            'midi_path': str(annotation.get('midi_path', '')),
            'start_time': annotation.get('start_time', 0.0),
            'end_time': annotation.get('end_time', len(audio) / sr),
        }

        # Apply modality dropout (only during training with augmentation enabled)
        # This prevents modality collapse and ensures both encoders learn useful representations
        drop_audio = False
        drop_midi = False

        if self.apply_augmentation and self.modality_dropout_enabled:
            drop_audio = random.random() < self.audio_dropout_prob
            drop_midi = random.random() < self.midi_dropout_prob

            # Ensure we don't drop both modalities (need at least one)
            if drop_audio and drop_midi:
                # Randomly keep one
                if random.random() < 0.5:
                    drop_audio = False
                else:
                    drop_midi = False

        result = {
            'labels': labels_tensor,
            'metadata': metadata,
        }

        # Add audio (or None if dropped)
        if drop_audio:
            # Return zeros for audio - model should handle gracefully
            result['audio_waveform'] = torch.zeros_like(audio_tensor)
            result['audio_dropped'] = True
        else:
            result['audio_waveform'] = audio_tensor
            result['audio_dropped'] = False

        # Add MIDI (or None if dropped)
        if drop_midi or midi_tokens is None:
            result['midi_dropped'] = True
            # Don't include midi_tokens key - model handles None
        else:
            result['midi_tokens'] = midi_tokens
            result['midi_dropped'] = False

        return result


def mixup_batch(
    audio: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
    probability: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mixup augmentation to a batch.

    Mixup creates virtual training examples by mixing pairs of examples and their labels.
    Research shows this improves generalization and calibration for regression tasks.

    Args:
        audio: Batch of audio waveforms [batch, num_samples]
        labels: Batch of labels [batch, num_dimensions]
        alpha: Beta distribution parameter (0.2 is standard)
        probability: Probability of applying mixup

    Returns:
        Tuple of (mixed_audio, mixed_labels)
    """
    if np.random.random() >= probability:
        # Don't apply mixup
        return audio, labels

    batch_size = audio.shape[0]
    if batch_size < 2:
        # Need at least 2 samples to mix
        return audio, labels

    # Sample mixing coefficient from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Create random permutation for pairing samples
    index = torch.randperm(batch_size, device=audio.device)

    # Mix audio and labels
    mixed_audio = lam * audio + (1 - lam) * audio[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]

    return mixed_audio, mixed_labels


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Handles variable-length sequences and optional MIDI.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary
    """
    # Stack audio waveforms (all same length after padding in __getitem__)
    audio_waveforms = torch.stack([item['audio_waveform'] for item in batch])

    # Stack labels
    labels = torch.stack([item['labels'] for item in batch])

    # Handle MIDI tokens (may not be present for all samples)
    midi_tokens = None
    if 'midi_tokens' in batch[0]:
        # Check if all samples have MIDI
        if all('midi_tokens' in item for item in batch):
            midi_tokens = torch.stack([item['midi_tokens'] for item in batch])
        else:
            # Some samples are missing MIDI - filter them out
            valid_indices = [i for i, item in enumerate(batch) if 'midi_tokens' in item]
            missing_indices = [i for i, item in enumerate(batch) if 'midi_tokens' not in item]

            if len(valid_indices) > 0:
                # Create partial batch with only valid samples
                batch = [batch[i] for i in valid_indices]
                audio_waveforms = torch.stack([item['audio_waveform'] for item in batch])
                labels = torch.stack([item['labels'] for item in batch])
                midi_tokens = torch.stack([item['midi_tokens'] for item in batch])
                metadata = [item['metadata'] for item in batch]

                # Log the filtering (but don't spam)
                if missing_indices[0] % 100 == 0:  # Log occasionally
                    missing_paths = [batch[i]['metadata']['midi_path'] for i in missing_indices]
                    print(f"\nInfo: Filtered out {len(missing_indices)} samples without MIDI from batch")

                return {
                    'audio_waveform': audio_waveforms,
                    'midi_tokens': midi_tokens,
                    'labels': labels,
                    'metadata': metadata,
                }
            else:
                # All samples failed - return None to skip this batch
                print(f"\nWarning: Entire batch failed MIDI loading - skipping")
                midi_tokens = None

    # Collect metadata
    metadata = [item['metadata'] for item in batch]

    result = {
        'audio_waveform': audio_waveforms,
        'labels': labels,
        'metadata': metadata,
    }

    if midi_tokens is not None:
        result['midi_tokens'] = midi_tokens

    return result


def create_collate_fn_with_mixup(
    apply_mixup: bool = True,
    mixup_alpha: float = 0.2,
    mixup_probability: float = 0.5
):
    """
    Create a collate function with optional mixup augmentation.

    Args:
        apply_mixup: Whether to apply mixup
        mixup_alpha: Beta distribution parameter
        mixup_probability: Probability of applying mixup

    Returns:
        Collate function
    """
    def collate_with_mixup(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # First, use standard collate
        result = collate_fn(batch)

        # Apply mixup to training data only (if enabled)
        if apply_mixup and 'audio_waveform' in result and 'labels' in result:
            mixed_audio, mixed_labels = mixup_batch(
                result['audio_waveform'],
                result['labels'],
                alpha=mixup_alpha,
                probability=mixup_probability
            )
            result['audio_waveform'] = mixed_audio
            result['labels'] = mixed_labels

        return result

    return collate_with_mixup


def create_dataloaders(
    train_annotation_path: str,
    val_annotation_path: str,
    test_annotation_path: Optional[str],
    dimension_names: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    augmentation_config: Optional[Dict[str, Any]] = None,
    modality_dropout_config: Optional[Dict[str, Any]] = None,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
    mixup_probability: float = 0.5,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test dataloaders.

    Args:
        train_annotation_path: Path to training annotations
        val_annotation_path: Path to validation annotations
        test_annotation_path: Path to test annotations (optional)
        dimension_names: List of evaluation dimensions
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        augmentation_config: Augmentation config (applied to train only)
        modality_dropout_config: Modality dropout config (applied to train only)
            {'enabled': True, 'audio_prob': 0.15, 'midi_prob': 0.15}
        use_mixup: Whether to apply mixup augmentation (train only)
        mixup_alpha: Beta distribution parameter for mixup
        mixup_probability: Probability of applying mixup per batch
        **dataset_kwargs: Additional arguments for PerformanceDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Training dataset (with augmentation and modality dropout)
    train_dataset = PerformanceDataset(
        annotation_path=train_annotation_path,
        dimension_names=dimension_names,
        augmentation_config=augmentation_config,
        apply_augmentation=True,
        modality_dropout_config=modality_dropout_config,
        **dataset_kwargs
    )

    # Validation dataset (no augmentation)
    val_dataset = PerformanceDataset(
        annotation_path=val_annotation_path,
        dimension_names=dimension_names,
        augmentation_config=None,
        apply_augmentation=False,
        **dataset_kwargs
    )

    # Test dataset (optional, no augmentation)
    test_dataset = None
    if test_annotation_path is not None:
        test_dataset = PerformanceDataset(
            annotation_path=test_annotation_path,
            dimension_names=dimension_names,
            augmentation_config=None,
            apply_augmentation=False,
            **dataset_kwargs
        )

    # Create collate functions
    # Training uses mixup, validation/test don't
    train_collate_fn = create_collate_fn_with_mixup(
        apply_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
        mixup_probability=mixup_probability
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,  # Use mixup for training
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,  # No mixup for validation
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,  # No mixup for testing
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return train_loader, val_loader, test_loader


class ContrastiveBatchSampler:
    """
    Batch sampler for contrastive learning with hard negative mining.

    Creates batches where:
    - Positive pairs: Same audio-MIDI segment (matched)
    - Hard negatives: Same piece at different degradation levels
    - Easy negatives: Different pieces (standard in-batch negatives)

    This forces the model to distinguish subtle quality differences rather than
    just learning to match pieces by identity.
    """

    def __init__(
        self,
        annotations: List[Dict[str, Any]],
        batch_size: int = 64,
        hard_neg_ratio: float = 0.25,
        drop_last: bool = True,
    ):
        """
        Initialize ContrastiveBatchSampler.

        Args:
            annotations: List of annotation dicts with 'piece_id' field
            batch_size: Number of samples per batch
            hard_neg_ratio: Fraction of batch that should be hard negatives (same piece)
            drop_last: Whether to drop the last incomplete batch
        """
        self.annotations = annotations
        self.batch_size = batch_size
        self.hard_neg_ratio = hard_neg_ratio
        self.drop_last = drop_last

        # Group samples by piece_id
        self.piece_groups: Dict[str, List[int]] = {}
        for idx, ann in enumerate(annotations):
            piece_id = ann.get('piece_id', ann.get('audio_path', str(idx)))
            if piece_id not in self.piece_groups:
                self.piece_groups[piece_id] = []
            self.piece_groups[piece_id].append(idx)

        # Filter pieces with multiple samples (for hard negatives)
        self.multi_sample_pieces = [
            piece_id for piece_id, indices in self.piece_groups.items()
            if len(indices) >= 2
        ]

        self.num_samples = len(annotations)
        self.num_batches = self.num_samples // batch_size
        if not drop_last and self.num_samples % batch_size != 0:
            self.num_batches += 1

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        """Generate batches with hard negatives."""
        # Shuffle all indices
        all_indices = list(range(self.num_samples))
        random.shuffle(all_indices)

        # Number of hard negatives per batch
        num_hard = int(self.batch_size * self.hard_neg_ratio)
        num_random = self.batch_size - num_hard

        # Shuffle pieces with multiple samples
        multi_sample_pieces = self.multi_sample_pieces.copy()
        random.shuffle(multi_sample_pieces)
        piece_iter = iter(multi_sample_pieces * 10)  # Repeat for enough pieces

        batch = []
        used_in_batch = set()
        random_idx = 0

        while len(all_indices) > 0 or len(batch) > 0:
            # Try to add hard negatives first
            if len(batch) < num_hard:
                try:
                    piece_id = next(piece_iter)
                    piece_indices = self.piece_groups[piece_id]

                    # Sample 2 indices from this piece (anchor + hard negative)
                    available = [i for i in piece_indices if i not in used_in_batch]
                    if len(available) >= 2:
                        sampled = random.sample(available, 2)
                        batch.extend(sampled)
                        used_in_batch.update(sampled)
                except StopIteration:
                    # Exhausted all pieces with multiple samples for hard negatives.
                    # This is expected behavior - continue filling batch with random samples.
                    # The while loop below handles this case by adding random indices.
                    break

            # Fill remaining slots with random samples
            while len(batch) < self.batch_size and random_idx < len(all_indices):
                idx = all_indices[random_idx]
                random_idx += 1
                if idx not in used_in_batch:
                    batch.append(idx)
                    used_in_batch.add(idx)

            # Yield batch if full
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
                used_in_batch = set(batch)
            elif not self.drop_last and len(batch) > 0 and random_idx >= len(all_indices):
                # Last incomplete batch
                yield batch
                break
            elif random_idx >= len(all_indices):
                break


def create_contrastive_dataloader(
    annotation_path: str,
    dimension_names: List[str],
    batch_size: int = 64,
    num_workers: int = 4,
    use_hard_negatives: bool = True,
    hard_neg_ratio: float = 0.25,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a dataloader for contrastive pre-training.

    Args:
        annotation_path: Path to annotation file
        dimension_names: List of evaluation dimensions
        batch_size: Batch size (larger is better for contrastive learning)
        num_workers: Number of DataLoader workers
        use_hard_negatives: Whether to use hard negative mining
        hard_neg_ratio: Fraction of batch that should be hard negatives
        **dataset_kwargs: Additional arguments for PerformanceDataset

    Returns:
        DataLoader configured for contrastive learning
    """
    dataset = PerformanceDataset(
        annotation_path=annotation_path,
        dimension_names=dimension_names,
        augmentation_config=None,
        apply_augmentation=False,
        **dataset_kwargs
    )

    if use_hard_negatives:
        # Use custom batch sampler for hard negative mining
        batch_sampler = ContrastiveBatchSampler(
            annotations=dataset.annotations,
            batch_size=batch_size,
            hard_neg_ratio=hard_neg_ratio,
            drop_last=True,
        )

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        # Standard random sampling
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )


if __name__ == "__main__":
    print("Performance dataset module loaded successfully")
    print("- Loads audio (24kHz raw waveforms)")
    print("- Loads MIDI (OctupleMIDI tokenization)")
    print("- Supports augmentation (train only)")
    print("- Mixup augmentation for better generalization")
    print("- Handles variable-length sequences")
    print("- ContrastiveBatchSampler for hard negative mining")
    print("- Annotation format: JSONL")
