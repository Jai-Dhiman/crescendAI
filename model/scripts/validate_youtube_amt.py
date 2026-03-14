"""Validate YouTube AMT pipeline: A1 audio scoring vs S2 symbolic scoring.

Five phases, each resumable (skips if output exists):
  1. A1 scoring (MuQ embeddings -> 6-dim scores)
  2. AMT transcription (WAV -> MIDI via ByteDance piano_transcription)
  3. Graph building (MIDI -> PyG graphs)
  4. S2 scoring (graphs -> 6-dim scores)
  5. Pairwise agreement report (A1 vs S2)

Usage:
    cd model
    uv run python scripts/validate_youtube_amt.py --phase all
    uv run python scripts/validate_youtube_amt.py --phase a1-score
    uv run python scripts/validate_youtube_amt.py --phase transcribe
    uv run python scripts/validate_youtube_amt.py --phase graph-and-score
    uv run python scripts/validate_youtube_amt.py --phase report
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pretty_midi
import torch

MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_ROOT / "src"))

from model_improvement.audio_encoders import MuQLoRAModel
from model_improvement.graph import count_midi_notes, parsed_midi_to_graph
from model_improvement.layer1_validation import score_competition_segments
from model_improvement.symbolic_encoders import GNNSymbolicEncoder
from model_improvement.taxonomy import DIMENSIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Defaults ---
DEFAULT_CACHE_DIR = MODEL_ROOT / "data" / "intermediate_cache"

A1_CKPT = MODEL_ROOT / "data/checkpoints/model_improvement/A1/fold_0/epoch=8-val_loss=0.4879.ckpt"
S2_CKPT = MODEL_ROOT / "data/checkpoints/model_improvement/S2/fold_3/epoch=6-val_loss=0.4949.ckpt"

MAX_NOTES = 10_000


# ---------------------------------------------------------------------------
# Phase 1: A1 scoring
# ---------------------------------------------------------------------------

def phase1_a1_score(cache_dir: Path) -> None:
    """Score competition segments with A1 (MuQ LoRA) model."""
    output_path = cache_dir / "a1_scores.json"
    if output_path.exists():
        logger.info("Phase 1: a1_scores.json already exists, skipping.")
        return

    embeddings_dir = cache_dir / "muq_embeddings"
    if not embeddings_dir.exists():
        raise FileNotFoundError(
            f"MuQ embeddings directory not found: {embeddings_dir}. "
            "Run extract_muq_embeddings.py first."
        )

    if not A1_CKPT.exists():
        raise FileNotFoundError(f"A1 checkpoint not found: {A1_CKPT}")

    logger.info("Loading A1 checkpoint: %s", A1_CKPT)
    model = MuQLoRAModel.load_from_checkpoint(str(A1_CKPT), map_location="cpu")

    logger.info("Scoring segments from: %s", embeddings_dir)
    scores = score_competition_segments(model, embeddings_dir)

    # Convert numpy arrays to lists for JSON serialization
    scores_json = {k: v.tolist() for k, v in scores.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores_json, f, indent=2)

    logger.info("Phase 1 complete: %d segments scored -> %s", len(scores_json), output_path)


# ---------------------------------------------------------------------------
# Phase 2: AMT transcription
# ---------------------------------------------------------------------------

def phase2_transcribe(cache_dir: Path) -> None:
    """Transcribe WAV files to MIDI using ByteDance piano_transcription."""
    try:
        from piano_transcription_inference import PianoTranscription  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "piano_transcription_inference is not installed. "
            "Install: uv pip install piano-transcription-inference"
        ) from e

    import soundfile as sf

    audio_dir = cache_dir / "audio"
    midi_dir = cache_dir / "transcribed_midi"

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")

    midi_dir.mkdir(parents=True, exist_ok=True)

    # Device selection: prefer MPS on Apple Silicon, fall back to CPU
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Using device: %s", device)

    transcriber = PianoTranscription(device=device)

    transcribed = 0
    skipped = 0
    failed = 0

    for wav_path in wav_files:
        video_id = wav_path.stem
        midi_path = midi_dir / f"{video_id}.mid"

        if midi_path.exists() and midi_path.stat().st_size > 0:
            logger.debug("Skipping %s (already transcribed)", video_id)
            skipped += 1
            continue

        logger.info("Transcribing %s ...", video_id)
        try:
            audio_data, _ = sf.read(str(wav_path), dtype="float32")
            # PianoTranscription expects mono float32 at 16kHz (resamples internally)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            transcriber.transcribe(audio_data, str(midi_path))
        except Exception as e:
            logger.error("Transcription failed for %s: %s", video_id, e)
            failed += 1
            continue

        if not midi_path.exists() or midi_path.stat().st_size == 0:
            logger.error("Transcription produced empty MIDI for %s", video_id)
            failed += 1
            continue

        # Log note count for sanity checking
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            n_notes = count_midi_notes(midi)
            logger.info("  %s: %d notes", video_id, n_notes)
        except Exception as e:
            logger.warning("Could not count notes for %s: %s", video_id, e)

        transcribed += 1

    logger.info(
        "Phase 2 complete: %d transcribed, %d skipped, %d failed",
        transcribed, skipped, failed,
    )


# ---------------------------------------------------------------------------
# Phase 3: Graph building
# ---------------------------------------------------------------------------

def phase3_build_graphs(cache_dir: Path) -> None:
    """Convert transcribed MIDI files to PyG graphs."""
    midi_dir = cache_dir / "transcribed_midi"
    output_path = cache_dir / "amt_graphs.pt"

    if output_path.exists():
        logger.info("Phase 3: amt_graphs.pt already exists, skipping.")
        return

    if not midi_dir.exists():
        raise FileNotFoundError(
            f"Transcribed MIDI directory not found: {midi_dir}. "
            "Run phase 2 (transcribe) first."
        )

    midi_files = sorted(midi_dir.glob("*.mid"))
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {midi_dir}")

    logger.info("Building graphs from %d MIDI files", len(midi_files))

    graphs: dict[str, object] = {}
    skipped = 0
    failed = 0

    for midi_path in midi_files:
        video_id = midi_path.stem

        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            logger.error("Failed to parse %s: %s", video_id, e)
            failed += 1
            continue

        note_count = count_midi_notes(midi)

        if note_count == 0:
            logger.info("Skipping %s: 0 notes", video_id)
            skipped += 1
            continue

        if note_count > MAX_NOTES:
            logger.info("Skipping %s: %d notes (> %d)", video_id, note_count, MAX_NOTES)
            skipped += 1
            continue

        try:
            graph = parsed_midi_to_graph(midi)
        except Exception as e:
            logger.error("Graph build failed for %s: %s", video_id, e)
            failed += 1
            continue

        graphs[video_id] = graph
        logger.debug("  %s: %d notes -> graph", video_id, note_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, output_path)

    logger.info(
        "Phase 3 complete: %d graphs saved, %d skipped, %d failed -> %s",
        len(graphs), skipped, failed, output_path,
    )


# ---------------------------------------------------------------------------
# Phase 4: S2 scoring
# ---------------------------------------------------------------------------

def phase4_s2_score(cache_dir: Path) -> None:
    """Score graphs with S2 (GNN) model."""
    output_path = cache_dir / "s2_scores.json"
    graphs_path = cache_dir / "amt_graphs.pt"

    if output_path.exists():
        logger.info("Phase 4: s2_scores.json already exists, skipping.")
        return

    if not graphs_path.exists():
        raise FileNotFoundError(
            f"Graphs file not found: {graphs_path}. "
            "Run phase 3 (graph building) first."
        )

    if not S2_CKPT.exists():
        raise FileNotFoundError(f"S2 checkpoint not found: {S2_CKPT}")

    logger.info("Loading S2 checkpoint: %s", S2_CKPT)
    model = GNNSymbolicEncoder.load_from_checkpoint(str(S2_CKPT), map_location="cpu")
    model.cpu()
    model.freeze()

    logger.info("Loading graphs from: %s", graphs_path)
    graphs = torch.load(graphs_path, map_location="cpu", weights_only=False)

    scores_json: dict[str, list[float]] = {}
    failed = 0

    for video_id, graph in graphs.items():
        try:
            with torch.no_grad():
                batch_tensor = torch.zeros(graph.x.size(0), dtype=torch.long)
                result = model(graph.x, graph.edge_index, batch_tensor)
                scores = result["scores"].squeeze(0).tolist()
            scores_json[video_id] = scores
        except Exception as e:
            logger.error("S2 scoring failed for %s: %s", video_id, e)
            failed += 1
            continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores_json, f, indent=2)

    logger.info(
        "Phase 4 complete: %d scored, %d failed -> %s",
        len(scores_json), failed, output_path,
    )


# ---------------------------------------------------------------------------
# Phase 5: Pairwise agreement report
# ---------------------------------------------------------------------------

def _load_metadata(cache_dir: Path) -> dict[str, str]:
    """Load metadata.jsonl and return {segment_id: video_id} mapping."""
    metadata_path = cache_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    segment_to_video: dict[str, str] = {}
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            segment_id = record.get("segment_id", "")
            # Extract video_id from segment_id pattern: yt_{video_id}_segNNN
            match = re.match(r"yt_(.+)_seg\d+", segment_id)
            if match:
                segment_to_video[segment_id] = match.group(1)

    return segment_to_video


def _aggregate_a1_scores(
    a1_scores: dict[str, list[float]],
    segment_to_video: dict[str, str],
) -> dict[str, np.ndarray]:
    """Aggregate A1 segment-level scores to recording-level means."""
    video_segments: dict[str, list[np.ndarray]] = defaultdict(list)

    for segment_id, scores in a1_scores.items():
        video_id = segment_to_video.get(segment_id)
        if video_id is None:
            # Try extracting directly from segment_id
            match = re.match(r"yt_(.+)_seg\d+", segment_id)
            if match:
                video_id = match.group(1)
            else:
                continue
        video_segments[video_id].append(np.array(scores, dtype=np.float32))

    aggregated: dict[str, np.ndarray] = {}
    for video_id, segment_scores in video_segments.items():
        aggregated[video_id] = np.mean(segment_scores, axis=0)

    return aggregated


def phase5_report(cache_dir: Path) -> None:
    """Generate pairwise agreement report between A1 and S2 scores."""
    a1_path = cache_dir / "a1_scores.json"
    s2_path = cache_dir / "s2_scores.json"
    output_path = cache_dir / "amt_validation_report.json"

    if not a1_path.exists():
        raise FileNotFoundError(f"A1 scores not found: {a1_path}. Run phase 1 first.")
    if not s2_path.exists():
        raise FileNotFoundError(f"S2 scores not found: {s2_path}. Run phase 4 first.")

    with open(a1_path) as f:
        a1_raw = json.load(f)
    with open(s2_path) as f:
        s2_scores_raw = json.load(f)

    # Aggregate A1 segment scores to recording level
    segment_to_video = _load_metadata(cache_dir)
    a1_by_video = _aggregate_a1_scores(a1_raw, segment_to_video)

    s2_by_video = {k: np.array(v, dtype=np.float32) for k, v in s2_scores_raw.items()}

    # Find recordings with both A1 and S2 scores
    common_ids = sorted(set(a1_by_video.keys()) & set(s2_by_video.keys()))
    logger.info("Recordings with both A1 and S2 scores: %d", len(common_ids))

    if len(common_ids) < 2:
        raise RuntimeError(
            f"Need at least 2 recordings with both scores for pairwise comparison, "
            f"got {len(common_ids)}."
        )

    # Pairwise comparison
    n_dims = len(DIMENSIONS)
    agree_per_dim = [0] * n_dims
    total_per_dim = [0] * n_dims
    agree_overall = 0
    total_overall = 0

    tie_threshold = 0.01

    for i in range(len(common_ids)):
        for j in range(i + 1, len(common_ids)):
            id_i = common_ids[i]
            id_j = common_ids[j]
            a1_i = a1_by_video[id_i]
            a1_j = a1_by_video[id_j]
            s2_i = s2_by_video[id_i]
            s2_j = s2_by_video[id_j]

            for d in range(n_dims):
                a1_diff = float(a1_i[d] - a1_j[d])
                s2_diff = float(s2_i[d] - s2_j[d])

                # Skip ties
                if abs(a1_diff) < tie_threshold or abs(s2_diff) < tie_threshold:
                    continue

                total_per_dim[d] += 1
                total_overall += 1

                if (a1_diff > 0) == (s2_diff > 0):
                    agree_per_dim[d] += 1
                    agree_overall += 1

    # Build report
    per_dim_report = {}
    for d, dim_name in enumerate(DIMENSIONS):
        if total_per_dim[d] > 0:
            pct = 100.0 * agree_per_dim[d] / total_per_dim[d]
        else:
            pct = 0.0
        per_dim_report[dim_name] = {
            "agree": agree_per_dim[d],
            "total": total_per_dim[d],
            "agreement_pct": round(pct, 2),
        }

    overall_pct = 100.0 * agree_overall / total_overall if total_overall > 0 else 0.0

    gate_threshold = 60.0
    report = {
        "n_recordings": len(common_ids),
        "n_pairs": len(common_ids) * (len(common_ids) - 1) // 2,
        "tie_threshold": tie_threshold,
        "overall": {
            "agree": agree_overall,
            "total": total_overall,
            "agreement_pct": round(overall_pct, 2),
        },
        "gate_threshold": gate_threshold,
        "gate_pass": overall_pct > gate_threshold,
        "per_dimension": per_dim_report,
        "recording_ids": common_ids,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Report saved to %s", output_path)

    # Print terminal summary
    print("\n=== AMT Validation: A1 vs S2 Pairwise Agreement ===")
    print(f"Recordings: {report['n_recordings']}")
    print(f"Pairs: {report['n_pairs']}")
    print(f"Overall agreement: {overall_pct:.1f}% ({agree_overall}/{total_overall})")
    print()
    print(f"{'Dimension':<16} {'Agreement':>10} {'Pairs':>8}")
    print("-" * 36)
    for dim_name in DIMENSIONS:
        d = per_dim_report[dim_name]
        print(f"{dim_name:<16} {d['agreement_pct']:>9.1f}% {d['total']:>8}")
    print()
    print(f"Gate (>60% overall): {'PASS' if report['gate_pass'] else 'FAIL'}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate YouTube AMT pipeline: A1 vs S2 pairwise agreement."
    )
    parser.add_argument(
        "--phase",
        choices=["a1-score", "transcribe", "graph-and-score", "report", "all"],
        required=True,
        help="Which phase(s) to run.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    args = parser.parse_args()

    cache_dir: Path = args.cache_dir
    phase: str = args.phase

    logger.info("=== YouTube AMT Validation ===")
    logger.info("Cache dir: %s", cache_dir)
    logger.info("Phase: %s", phase)

    if phase in ("a1-score", "all"):
        logger.info("--- Phase 1: A1 scoring ---")
        phase1_a1_score(cache_dir)

    if phase in ("transcribe", "all"):
        logger.info("--- Phase 2: AMT transcription ---")
        phase2_transcribe(cache_dir)

    if phase in ("graph-and-score", "all"):
        logger.info("--- Phase 3: Graph building ---")
        phase3_build_graphs(cache_dir)

        logger.info("--- Phase 4: S2 scoring ---")
        phase4_s2_score(cache_dir)

    if phase in ("report", "all"):
        logger.info("--- Phase 5: Pairwise agreement report ---")
        phase5_report(cache_dir)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
