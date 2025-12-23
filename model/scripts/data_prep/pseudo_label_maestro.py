#!/usr/bin/env python3
"""
Pseudo-label MAESTRO dataset using trained PercePiano teacher model.

This script uses a trained PercePiano replica model to generate pseudo-labels
for the MAESTRO dataset, enabling semi-supervised learning with expanded data.

The pipeline:
1. Load trained teacher model (PercePiano replica with R^2 >= 0.25)
2. Load MAESTRO MIDI files
3. Extract score alignment features
4. Generate predictions with uncertainty estimates
5. Filter by confidence threshold
6. Save pseudo-labels in PercePiano format

Usage:
    python scripts/pseudo_label_maestro.py \
        --teacher /tmp/checkpoints/percepiano_replica/percepiano_teacher.pt \
        --maestro_dir /tmp/maestro_segments \
        --output_dir /tmp/pseudo_labels \
        --confidence_threshold 0.7

Requirements:
    - Trained PercePiano teacher model (R^2 >= 0.25)
    - MAESTRO dataset (segmented MIDI files)
    - Score files (optional, but recommended)

Attribution:
    Teacher model architecture from PercePiano (Park et al., ISMIR/Nature 2024)
    https://github.com/JonghoKimSNU/PercePiano
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class PseudoLabel:
    """Container for a pseudo-labeled sample."""

    midi_path: str
    score_path: Optional[str]
    predictions: Dict[str, float]  # Dimension -> predicted score
    confidence: float  # Overall confidence (0-1)
    per_dim_confidence: Dict[str, float]  # Per-dimension confidence
    teacher_r2: float  # Teacher model's R^2
    is_reliable: bool  # Whether confidence >= threshold


def load_teacher_model(
    checkpoint_path: Path, device: str = "cuda"
) -> Tuple[torch.nn.Module, dict]:
    """
    Load trained PercePiano teacher model.

    Args:
        checkpoint_path: Path to teacher checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, metadata)
    """
    print(f"Loading teacher model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model metadata
    metadata = {
        "dimensions": checkpoint.get("dimensions", []),
        "r2": checkpoint.get("metrics", {}).get("r2", 0),
        "per_dimension_r2": checkpoint.get("metrics", {}).get("per_dimension_r2", {}),
    }

    print(f"  Teacher R^2: {metadata['r2']:.4f}")
    print(f"  Dimensions: {len(metadata['dimensions'])}")

    # Load model
    from models.percepiano_replica import PercePianoReplicaModule

    # Reconstruct model from hyperparameters
    hparams = checkpoint.get("hparams", {})
    model = PercePianoReplicaModule(**hparams)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, metadata


def extract_features_from_midi(
    midi_path: Path,
    score_path: Optional[Path] = None,
    max_notes: int = 1024,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Extract score alignment features from MIDI file.

    Args:
        midi_path: Path to MIDI file
        score_path: Optional path to MusicXML score
        max_notes: Maximum number of notes to process

    Returns:
        Dictionary of feature tensors, or None if extraction fails
    """
    try:
        import pretty_midi
        from data.score_alignment import (
            NUM_NOTE_FEATURES,
            ScoreAlignmentFeatureExtractor,
        )

        # Load MIDI
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        # Initialize feature extractor
        extractor = ScoreAlignmentFeatureExtractor()

        if score_path and score_path.exists():
            # Extract full score alignment features
            features = extractor.extract_features(midi, score_path)
        else:
            # Extract MIDI-only features (no score alignment)
            features = extract_midi_only_features(midi, max_notes)

        # Convert to tensors
        note_features = torch.tensor(features["note_features"], dtype=torch.float32)
        global_features = torch.tensor(features["global_features"], dtype=torch.float32)
        tempo_curve = torch.tensor(features["tempo_curve"], dtype=torch.float32)

        # Pad/truncate note features
        num_notes = note_features.shape[0]
        if num_notes > max_notes:
            note_features = note_features[:max_notes]
            num_notes = max_notes
        elif num_notes < max_notes:
            padding = torch.zeros(max_notes - num_notes, note_features.shape[1])
            note_features = torch.cat([note_features, padding], dim=0)

        # Create note_locations (simplified - assume sequential beats/measures)
        note_locations = create_note_locations(
            features.get("note_locations", {}), num_notes, max_notes
        )

        # Pad tempo curve
        max_tempo = 256
        if tempo_curve.shape[0] > max_tempo:
            tempo_curve = tempo_curve[:max_tempo]
        elif tempo_curve.shape[0] < max_tempo:
            padding = torch.ones(max_tempo - tempo_curve.shape[0])
            tempo_curve = torch.cat([tempo_curve, padding])

        return {
            "score_note_features": note_features.unsqueeze(0),  # Add batch dim
            "score_global_features": global_features.unsqueeze(0),
            "score_tempo_curve": tempo_curve.unsqueeze(0),
            "note_locations_beat": note_locations["beat"].unsqueeze(0),
            "note_locations_measure": note_locations["measure"].unsqueeze(0),
            "note_locations_voice": note_locations["voice"].unsqueeze(0),
        }

    except Exception as e:
        print(f"  Warning: Failed to extract features from {midi_path}: {e}")
        return None


def extract_midi_only_features(midi: "pretty_midi.PrettyMIDI", max_notes: int) -> Dict:
    """
    Extract features from MIDI without score alignment.

    This is a fallback when no score is available.
    """
    from data.score_alignment import NUM_NOTE_FEATURES

    # Collect all notes
    all_notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            all_notes.append(
                {
                    "pitch": note.pitch,
                    "onset": note.start,
                    "duration": note.end - note.start,
                    "velocity": note.velocity,
                }
            )

    # Sort by onset
    all_notes.sort(key=lambda x: x["onset"])

    if len(all_notes) == 0:
        return {
            "note_features": np.zeros((1, NUM_NOTE_FEATURES), dtype=np.float32),
            "global_features": np.zeros(12, dtype=np.float32),
            "tempo_curve": np.ones(1, dtype=np.float32),
            "note_locations": {},
        }

    # Extract per-note features (simplified version)
    note_features = []
    for i, note in enumerate(all_notes[:max_notes]):
        # Simplified features when no score available
        feat = np.zeros(NUM_NOTE_FEATURES, dtype=np.float32)
        feat[0] = 0.0  # onset_deviation (unknown without score)
        feat[1] = 1.0  # duration_ratio (assume correct)
        feat[2] = 0.0  # articulation_log
        feat[3] = note["velocity"] / 127.0  # velocity
        feat[4] = 0.0  # velocity_deviation (unknown)
        feat[5] = 1.0  # local_tempo_ratio (assume correct)
        feat[6] = (note["pitch"] - 21) / 87.0  # midi_pitch normalized
        feat[7] = (note["pitch"] % 12) / 11.0  # pitch_class
        feat[8] = (note["pitch"] // 12 - 1) / 8.0  # octave
        feat[9] = note["onset"] % 1.0  # beat_position (approximate)
        feat[10] = min(i / max_notes, 1.0)  # beat_index (normalized)
        feat[11] = min(i / max_notes, 1.0)  # measure (approximate)
        feat[12] = 0.0  # voice
        feat[13] = 1.0  # matched (assumed)
        feat[14] = 0.0  # is_chord_member
        feat[15] = 0.0  # following_rest
        feat[16] = 0.0  # is_staccato
        feat[17] = 0.0  # is_legato
        feat[18] = 0.5  # dynamic_level (unknown)
        feat[19] = 0.5  # tempo_val (unknown)
        note_features.append(feat)

    note_features = np.array(note_features, dtype=np.float32)

    # Global features (simplified)
    velocities = [n["velocity"] for n in all_notes]
    durations = [n["duration"] for n in all_notes]

    global_features = np.array(
        [
            0.0,  # mean onset deviation
            0.0,  # std onset deviation
            np.mean(durations),  # mean duration
            np.std(durations),  # std duration
            np.mean(velocities) / 127.0,  # mean velocity
            np.std(velocities) / 127.0,  # std velocity
            1.0,  # match rate
            0.0,  # Q1 onset deviation
            0.0,  # Q3 onset deviation
            0.0,  # max abs onset deviation
            np.median(durations),  # median duration
            midi.estimate_tempo()
            if hasattr(midi, "estimate_tempo")
            else 120.0,  # tempo
        ],
        dtype=np.float32,
    )

    # Tempo curve (uniform)
    tempo_curve = np.ones(max(1, len(all_notes) // 10), dtype=np.float32)

    # Note locations (sequential)
    note_locations = {
        "beat": np.arange(1, len(note_features) + 1, dtype=np.int64),
        "measure": np.arange(1, len(note_features) + 1, dtype=np.int64) // 4 + 1,
        "voice": np.ones(len(note_features), dtype=np.int64),
    }

    return {
        "note_features": note_features,
        "global_features": global_features,
        "tempo_curve": tempo_curve,
        "note_locations": note_locations,
    }


def create_note_locations(
    locations: Dict, num_notes: int, max_notes: int
) -> Dict[str, torch.Tensor]:
    """Create padded note location tensors."""
    if locations and "beat" in locations:
        beat = torch.tensor(locations["beat"][:max_notes], dtype=torch.long)
        measure = torch.tensor(locations["measure"][:max_notes], dtype=torch.long)
        voice = torch.tensor(locations["voice"][:max_notes], dtype=torch.long)
    else:
        # Default sequential locations
        beat = torch.arange(1, num_notes + 1, dtype=torch.long)
        measure = torch.arange(1, num_notes + 1, dtype=torch.long) // 4 + 1
        voice = torch.ones(num_notes, dtype=torch.long)

    # Pad to max_notes
    if len(beat) < max_notes:
        pad_size = max_notes - len(beat)
        beat = torch.cat([beat, torch.zeros(pad_size, dtype=torch.long)])
        measure = torch.cat([measure, torch.zeros(pad_size, dtype=torch.long)])
        voice = torch.cat([voice, torch.zeros(pad_size, dtype=torch.long)])

    return {"beat": beat, "measure": measure, "voice": voice}


def compute_prediction_confidence(
    predictions: torch.Tensor,
    model: torch.nn.Module,
    features: Dict[str, torch.Tensor],
    n_samples: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate prediction confidence using MC Dropout.

    Args:
        predictions: Model predictions [batch, dims]
        model: The teacher model (with dropout)
        features: Input features
        n_samples: Number of MC samples

    Returns:
        Tuple of (mean_predictions, confidence_scores)
    """
    # Enable dropout for MC sampling
    model.train()

    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            note_locations = {
                "beat": features["note_locations_beat"],
                "measure": features["note_locations_measure"],
                "voice": features["note_locations_voice"],
            }
            outputs = model(
                features["score_note_features"],
                features["score_global_features"],
                features["score_tempo_curve"],
                note_locations,
            )
            all_preds.append(outputs["predictions"])

    model.eval()

    # Stack predictions
    all_preds = torch.stack(all_preds, dim=0)  # [n_samples, batch, dims]

    # Compute mean and std
    mean_preds = all_preds.mean(dim=0)
    std_preds = all_preds.std(dim=0)

    # Confidence = 1 - normalized std (higher std = lower confidence)
    # Normalize std to 0-1 range (assume max std of 0.3)
    confidence = 1.0 - torch.clamp(std_preds / 0.3, 0, 1)

    return mean_preds, confidence


def pseudo_label_maestro(
    teacher_path: Path,
    maestro_dir: Path,
    output_dir: Path,
    score_dir: Optional[Path] = None,
    confidence_threshold: float = 0.7,
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> List[PseudoLabel]:
    """
    Generate pseudo-labels for MAESTRO dataset.

    Args:
        teacher_path: Path to trained teacher model
        maestro_dir: Directory containing MAESTRO MIDI files
        output_dir: Output directory for pseudo-labels
        score_dir: Optional directory containing MusicXML scores
        confidence_threshold: Minimum confidence to include sample
        max_samples: Maximum samples to process (for testing)
        device: Device to use

    Returns:
        List of PseudoLabel objects
    """
    print("=" * 70)
    print("PSEUDO-LABELING MAESTRO")
    print("=" * 70)

    # Load teacher model
    model, metadata = load_teacher_model(teacher_path, device)
    dimensions = metadata["dimensions"]
    teacher_r2 = metadata["r2"]

    if teacher_r2 < 0.15:
        print(f"\nWARNING: Teacher R^2 ({teacher_r2:.4f}) is very low!")
        print(
            "Pseudo-labels may not be reliable. Consider training a better teacher first."
        )
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != "y":
            return []

    # Find MIDI files
    midi_files = list(maestro_dir.rglob("*.mid")) + list(maestro_dir.rglob("*.midi"))
    print(f"\nFound {len(midi_files)} MIDI files")

    if max_samples:
        midi_files = midi_files[:max_samples]
        print(f"Processing first {max_samples} files (testing mode)")

    # Process files
    pseudo_labels = []
    reliable_count = 0

    for midi_path in tqdm(midi_files, desc="Generating pseudo-labels"):
        # Find corresponding score if available
        score_path = None
        if score_dir:
            # Try to find matching score
            score_name = midi_path.stem + ".musicxml"
            potential_score = score_dir / score_name
            if potential_score.exists():
                score_path = potential_score

        # Extract features
        features = extract_features_from_midi(midi_path, score_path)
        if features is None:
            continue

        # Move to device
        features = {k: v.to(device) for k, v in features.items()}

        # Get predictions with confidence
        with torch.no_grad():
            note_locations = {
                "beat": features["note_locations_beat"],
                "measure": features["note_locations_measure"],
                "voice": features["note_locations_voice"],
            }
            outputs = model(
                features["score_note_features"],
                features["score_global_features"],
                features["score_tempo_curve"],
                note_locations,
            )
            predictions = outputs["predictions"]

        # Compute confidence using MC Dropout
        mean_preds, confidence = compute_prediction_confidence(
            predictions, model, features, n_samples=10
        )

        # Convert to numpy
        mean_preds = mean_preds.cpu().numpy()[0]
        confidence = confidence.cpu().numpy()[0]
        overall_confidence = float(confidence.mean())

        # Create pseudo-label
        pred_dict = {dim: float(mean_preds[i]) for i, dim in enumerate(dimensions)}
        conf_dict = {dim: float(confidence[i]) for i, dim in enumerate(dimensions)}

        is_reliable = overall_confidence >= confidence_threshold
        if is_reliable:
            reliable_count += 1

        pl = PseudoLabel(
            midi_path=str(midi_path),
            score_path=str(score_path) if score_path else None,
            predictions=pred_dict,
            confidence=overall_confidence,
            per_dim_confidence=conf_dict,
            teacher_r2=teacher_r2,
            is_reliable=is_reliable,
        )
        pseudo_labels.append(pl)

    # Summary
    print(f"\n{'=' * 70}")
    print("PSEUDO-LABELING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total samples processed: {len(pseudo_labels)}")
    print(
        f"Reliable samples (conf >= {confidence_threshold}): {reliable_count} ({100 * reliable_count / len(pseudo_labels):.1f}%)"
    )
    print(f"Teacher R^2: {teacher_r2:.4f}")

    # Save pseudo-labels
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all labels
    all_labels_path = output_dir / "pseudo_labels_all.json"
    with open(all_labels_path, "w") as f:
        json.dump([asdict(pl) for pl in pseudo_labels], f, indent=2)
    print(f"\nSaved all labels to: {all_labels_path}")

    # Save only reliable labels
    reliable_labels = [pl for pl in pseudo_labels if pl.is_reliable]
    reliable_path = output_dir / "pseudo_labels_reliable.json"
    with open(reliable_path, "w") as f:
        json.dump([asdict(pl) for pl in reliable_labels], f, indent=2)
    print(f"Saved reliable labels to: {reliable_path}")

    # Save in PercePiano format (for compatibility with existing data loader)
    percepiano_format = []
    for pl in reliable_labels:
        sample = {
            "name": Path(pl.midi_path).stem,
            "midi_path": pl.midi_path,
            "score_path": pl.score_path,
            "scores": pl.predictions,
            "percepiano_scores": [pl.predictions[dim] for dim in dimensions],
            "is_pseudo_label": True,
            "confidence": pl.confidence,
            "teacher_r2": pl.teacher_r2,
        }
        percepiano_format.append(sample)

    percepiano_path = output_dir / "maestro_pseudo_train.json"
    with open(percepiano_path, "w") as f:
        json.dump(percepiano_format, f, indent=2)
    print(f"Saved PercePiano format to: {percepiano_path}")

    return pseudo_labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels for MAESTRO using PercePiano teacher"
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        required=True,
        help="Path to trained teacher model checkpoint",
    )
    parser.add_argument(
        "--maestro_dir",
        type=Path,
        required=True,
        help="Directory containing MAESTRO MIDI files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/tmp/pseudo_labels"),
        help="Output directory for pseudo-labels",
    )
    parser.add_argument(
        "--score_dir",
        type=Path,
        default=None,
        help="Optional directory containing MusicXML scores",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (0-1)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    if not args.teacher.exists():
        print(f"ERROR: Teacher model not found: {args.teacher}")
        print("\nTrain a teacher model first:")
        print("  1. Run train_percepiano_replica.ipynb")
        print("  2. Achieve R^2 >= 0.25")
        print("  3. Model saved as percepiano_teacher.pt")
        sys.exit(1)

    if not args.maestro_dir.exists():
        print(f"ERROR: MAESTRO directory not found: {args.maestro_dir}")
        print("\nDownload MAESTRO first:")
        print("  rclone copy gdrive:maestro_data /tmp/maestro")
        sys.exit(1)

    pseudo_label_maestro(
        teacher_path=args.teacher,
        maestro_dir=args.maestro_dir,
        output_dir=args.output_dir,
        score_dir=args.score_dir,
        confidence_threshold=args.confidence_threshold,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
