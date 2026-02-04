"""PyTorch datasets for score-performance alignment.

These datasets load pre-extracted MuQ embeddings for both rendered scores
and student performances, along with ground truth alignments.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..config import MUQ_FRAME_RATE
from .asap import (
    ASAPPerformance,
    NoteAlignment,
    get_measure_boundaries,
    get_performance_key,
    load_note_alignments,
)


def path_to_cache_key(path_str: str) -> str:
    """Convert a path string to a safe cache key filename.

    Replaces path separators with underscores and removes .mid extension.
    """
    return path_str.replace("/", "_").replace("\\", "_").replace(".mid", "")


class FrameAlignmentDataset(Dataset):
    """Dataset for frame-level alignment between score and performance embeddings.

    Each sample contains:
        - score_embeddings: MuQ embeddings for rendered score MIDI
        - perf_embeddings: MuQ embeddings for student performance audio
        - ground_truth: Note-level alignment (score_onsets, perf_onsets)
    """

    def __init__(
        self,
        performances: List[ASAPPerformance],
        score_cache_dir: Path,
        perf_cache_dir: Path,
        asap_root: Path,
        max_frames: int = 3000,
        frame_rate: float = MUQ_FRAME_RATE,
    ):
        """Initialize frame alignment dataset.

        Args:
            performances: List of ASAPPerformance objects with alignment data.
            score_cache_dir: Directory containing cached score MuQ embeddings.
            perf_cache_dir: Directory containing cached performance MuQ embeddings.
            asap_root: Root directory of ASAP dataset for loading alignments.
            max_frames: Maximum number of frames to keep per sequence.
            frame_rate: Frame rate of MuQ embeddings (default: 75 fps).
        """
        self.score_cache_dir = Path(score_cache_dir)
        self.perf_cache_dir = Path(perf_cache_dir)
        self.asap_root = Path(asap_root)
        self.max_frames = max_frames
        self.frame_rate = frame_rate

        # Filter to performances with available embeddings
        score_available = {p.stem for p in self.score_cache_dir.glob("*.pt")}
        perf_available = {p.stem for p in self.perf_cache_dir.glob("*.pt")}

        self.samples = []
        for perf in performances:
            perf_key = self._get_perf_key(perf)
            score_key = self._get_score_key(perf)

            if (
                score_key in score_available
                and perf_key in perf_available
                and perf.has_alignment()
            ):
                self.samples.append(
                    {
                        "key": get_performance_key(perf),  # Original key for display
                        "perf_key": perf_key,  # Cache key for loading
                        "score_key": score_key,
                        "performance": perf,
                    }
                )

    def _get_score_key(self, perf: ASAPPerformance) -> str:
        """Get cache key for the score embeddings."""
        if perf.midi_score_path:
            # Use full path to ensure uniqueness (e.g., Bach_Fugue_bwv_846_midi_score)
            return path_to_cache_key(str(perf.midi_score_path))
        return f"{perf.composer}_{perf.title}_score"

    def _get_perf_key(self, perf: ASAPPerformance) -> str:
        """Get cache key for performance embeddings."""
        key = get_performance_key(perf)
        return path_to_cache_key(key)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        perf = sample["performance"]

        # Load embeddings
        score_emb = torch.load(
            self.score_cache_dir / f"{sample['score_key']}.pt",
            weights_only=True,
        )
        perf_emb = torch.load(
            self.perf_cache_dir / f"{sample['perf_key']}.pt",
            weights_only=True,
        )

        # Truncate if needed
        if score_emb.shape[0] > self.max_frames:
            score_emb = score_emb[: self.max_frames]
        if perf_emb.shape[0] > self.max_frames:
            perf_emb = perf_emb[: self.max_frames]

        # Load ground truth alignments
        alignments = load_note_alignments(perf.alignment_path, self.asap_root)
        score_onsets = torch.tensor(
            [a.score_onset for a in alignments], dtype=torch.float32
        )
        perf_onsets = torch.tensor(
            [a.performance_onset for a in alignments], dtype=torch.float32
        )

        return {
            "score_embeddings": score_emb,
            "perf_embeddings": perf_emb,
            "score_onsets": score_onsets,
            "perf_onsets": perf_onsets,
            "key": sample["key"],
            "score_length": score_emb.shape[0],
            "perf_length": perf_emb.shape[0],
            "num_notes": len(alignments),
        }


class MeasureAlignmentDataset(Dataset):
    """Dataset for measure-level alignment using pooled frame embeddings.

    Pools MuQ frame embeddings by measure boundaries to create coarser
    representations for alignment. Useful for faster DTW and reduced memory.
    """

    def __init__(
        self,
        performances: List[ASAPPerformance],
        score_cache_dir: Path,
        perf_cache_dir: Path,
        asap_root: Path,
        max_measures: int = 500,
        frame_rate: float = MUQ_FRAME_RATE,
        pooling: str = "mean",
    ):
        """Initialize measure alignment dataset.

        Args:
            performances: List of ASAPPerformance objects.
            score_cache_dir: Directory containing cached score MuQ embeddings.
            perf_cache_dir: Directory containing cached performance MuQ embeddings.
            asap_root: Root directory of ASAP dataset.
            max_measures: Maximum number of measures to keep.
            frame_rate: Frame rate of MuQ embeddings.
            pooling: Pooling method ("mean", "max").
        """
        self.score_cache_dir = Path(score_cache_dir)
        self.perf_cache_dir = Path(perf_cache_dir)
        self.asap_root = Path(asap_root)
        self.max_measures = max_measures
        self.frame_rate = frame_rate
        self.pooling = pooling

        # Filter to performances with available embeddings and annotations
        score_available = {p.stem for p in self.score_cache_dir.glob("*.pt")}
        perf_available = {p.stem for p in self.perf_cache_dir.glob("*.pt")}

        self.samples = []
        for perf in performances:
            perf_key = self._get_perf_key(perf)
            score_key = self._get_score_key(perf)

            if (
                score_key in score_available
                and perf_key in perf_available
                and perf.annotations_path
            ):
                self.samples.append(
                    {
                        "key": get_performance_key(perf),  # Original key for display
                        "perf_key": perf_key,  # Cache key for loading
                        "score_key": score_key,
                        "performance": perf,
                    }
                )

    def _get_score_key(self, perf: ASAPPerformance) -> str:
        """Get cache key for the score embeddings."""
        if perf.midi_score_path:
            # Use full path to ensure uniqueness (e.g., Bach_Fugue_bwv_846_midi_score)
            return path_to_cache_key(str(perf.midi_score_path))
        return f"{perf.composer}_{perf.title}_score"

    def _get_perf_key(self, perf: ASAPPerformance) -> str:
        """Get cache key for performance embeddings."""
        key = get_performance_key(perf)
        return path_to_cache_key(key)

    def _pool_by_measures(
        self,
        embeddings: torch.Tensor,
        measure_boundaries: List[Tuple[float, float]],
    ) -> torch.Tensor:
        """Pool frame embeddings by measure boundaries.

        Args:
            embeddings: [T, D] tensor of frame embeddings.
            measure_boundaries: List of (start_time, end_time) for each measure.

        Returns:
            [M, D] tensor of measure-pooled embeddings.
        """
        pooled = []

        for start_time, end_time in measure_boundaries:
            start_frame = int(start_time * self.frame_rate)
            end_frame = int(end_time * self.frame_rate)

            # Clamp to valid range
            start_frame = max(0, min(start_frame, embeddings.shape[0] - 1))
            end_frame = max(start_frame + 1, min(end_frame, embeddings.shape[0]))

            measure_frames = embeddings[start_frame:end_frame]

            if self.pooling == "max":
                pooled_emb = measure_frames.max(dim=0)[0]
            else:  # mean
                pooled_emb = measure_frames.mean(dim=0)

            pooled.append(pooled_emb)

        if not pooled:
            # Return single mean if no measures found
            return embeddings.mean(dim=0, keepdim=True)

        return torch.stack(pooled)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        perf = sample["performance"]

        # Load embeddings
        score_emb = torch.load(
            self.score_cache_dir / f"{sample['score_key']}.pt",
            weights_only=True,
        )
        perf_emb = torch.load(
            self.perf_cache_dir / f"{sample['perf_key']}.pt",
            weights_only=True,
        )

        # Load measure boundaries
        measure_bounds = get_measure_boundaries(perf.annotations_path, self.asap_root)

        # Truncate measures if needed
        if len(measure_bounds) > self.max_measures:
            measure_bounds = measure_bounds[: self.max_measures]

        # Pool by measures
        score_pooled = self._pool_by_measures(score_emb, measure_bounds)
        perf_pooled = self._pool_by_measures(perf_emb, measure_bounds)

        # Ground truth: measure start times
        score_times = torch.tensor(
            [m[0] for m in measure_bounds], dtype=torch.float32
        )

        return {
            "score_embeddings": score_pooled,
            "perf_embeddings": perf_pooled,
            "measure_times": score_times,
            "key": sample["key"],
            "score_length": score_pooled.shape[0],
            "perf_length": perf_pooled.shape[0],
            "num_measures": len(measure_bounds),
        }


def frame_alignment_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for FrameAlignmentDataset with padding.

    Pads both score and performance sequences to batch max length.
    Creates attention masks for valid positions.
    """
    # Separate score and performance embeddings
    score_embs = [b["score_embeddings"] for b in batch]
    perf_embs = [b["perf_embeddings"] for b in batch]

    score_lengths = torch.tensor([b["score_length"] for b in batch])
    perf_lengths = torch.tensor([b["perf_length"] for b in batch])

    # Pad sequences
    score_padded = pad_sequence(score_embs, batch_first=True)
    perf_padded = pad_sequence(perf_embs, batch_first=True)

    # Create masks
    score_mask = (
        torch.arange(score_padded.shape[1]).unsqueeze(0) < score_lengths.unsqueeze(1)
    )
    perf_mask = (
        torch.arange(perf_padded.shape[1]).unsqueeze(0) < perf_lengths.unsqueeze(1)
    )

    # Collect ground truth onsets (these are variable length per sample)
    score_onsets = [b["score_onsets"] for b in batch]
    perf_onsets = [b["perf_onsets"] for b in batch]

    return {
        "score_embeddings": score_padded,
        "perf_embeddings": perf_padded,
        "score_mask": score_mask,
        "perf_mask": perf_mask,
        "score_lengths": score_lengths,
        "perf_lengths": perf_lengths,
        "score_onsets": score_onsets,  # List of tensors (variable length)
        "perf_onsets": perf_onsets,  # List of tensors (variable length)
        "keys": [b["key"] for b in batch],
        "num_notes": torch.tensor([b["num_notes"] for b in batch]),
    }


def measure_alignment_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for MeasureAlignmentDataset with padding."""
    score_embs = [b["score_embeddings"] for b in batch]
    perf_embs = [b["perf_embeddings"] for b in batch]

    score_lengths = torch.tensor([b["score_length"] for b in batch])
    perf_lengths = torch.tensor([b["perf_length"] for b in batch])

    # Pad sequences
    score_padded = pad_sequence(score_embs, batch_first=True)
    perf_padded = pad_sequence(perf_embs, batch_first=True)

    # Create masks
    score_mask = (
        torch.arange(score_padded.shape[1]).unsqueeze(0) < score_lengths.unsqueeze(1)
    )
    perf_mask = (
        torch.arange(perf_padded.shape[1]).unsqueeze(0) < perf_lengths.unsqueeze(1)
    )

    # Measure times (pad with zeros)
    measure_times = [b["measure_times"] for b in batch]
    measure_times_padded = pad_sequence(measure_times, batch_first=True)

    return {
        "score_embeddings": score_padded,
        "perf_embeddings": perf_padded,
        "score_mask": score_mask,
        "perf_mask": perf_mask,
        "score_lengths": score_lengths,
        "perf_lengths": perf_lengths,
        "measure_times": measure_times_padded,
        "keys": [b["key"] for b in batch],
        "num_measures": torch.tensor([b["num_measures"] for b in batch]),
    }
