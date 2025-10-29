import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset, DataLoader

from .audio_processing import load_audio, segment_audio, normalize_audio
from .midi_processing import load_midi, encode_octuple_midi, segment_midi, align_midi_to_audio
from .augmentation import AudioAugmentation


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
        """
        super().__init__()

        self.annotation_path = Path(annotation_path)
        self.dimension_names = dimension_names
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_length = max_audio_length
        self.max_midi_events = max_midi_events
        self.augmentation_config = augmentation_config
        self.apply_augmentation = apply_augmentation

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

                # Encode MIDI to OctupleMIDI tokens
                midi_tokens = encode_octuple_midi(midi)

                # Validate tokens before padding
                if len(midi_tokens) == 0:
                    # Empty MIDI file, skip
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

            except Exception as e:
                # Report MIDI loading errors loudly for debugging
                print(f"Warning: Failed to load MIDI for {midi_path}: {e}")
                midi_tokens = None

        # Extract labels
        labels_dict = annotation['labels']
        labels = []
        for dim_name in self.dimension_names:
            if dim_name in labels_dict:
                labels.append(labels_dict[dim_name])
            else:
                # Handle missing dimension (use 50.0 as neutral default)
                print(f"Warning: Missing dimension '{dim_name}' for sample {idx}, using default 50.0")
                labels.append(50.0)

        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        # Metadata
        metadata = {
            'audio_path': str(audio_path),
            'midi_path': str(annotation.get('midi_path', '')),
            'start_time': annotation.get('start_time', 0.0),
            'end_time': annotation.get('end_time', len(audio) / sr),
        }

        result = {
            'audio_waveform': audio_tensor,
            'labels': labels_tensor,
            'metadata': metadata,
        }

        if midi_tokens is not None:
            result['midi_tokens'] = midi_tokens

        return result


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


def create_dataloaders(
    train_annotation_path: str,
    val_annotation_path: str,
    test_annotation_path: Optional[str],
    dimension_names: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    augmentation_config: Optional[Dict[str, Any]] = None,
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
        **dataset_kwargs: Additional arguments for PerformanceDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Training dataset (with augmentation)
    train_dataset = PerformanceDataset(
        annotation_path=train_annotation_path,
        dimension_names=dimension_names,
        augmentation_config=augmentation_config,
        apply_augmentation=True,
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

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
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
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Performance dataset module loaded successfully")
    print("- Loads audio (24kHz raw waveforms)")
    print("- Loads MIDI (OctupleMIDI tokenization)")
    print("- Supports augmentation (train only)")
    print("- Handles variable-length sequences")
    print("- Annotation format: JSONL")
