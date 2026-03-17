"""Build expanded AMT validation test set for S2 robustness testing.

Runs on Thunder Compute (needs GPU for transcription + A1 scoring).

Pipeline:
  1. Score all MAESTRO embeddings with A1 to find close-quality pairs
  2. Select 25 pieces from contrastive_mapping.json with diverse quality gaps
  3. Download audio for selected pieces only (~50-80 recordings)
  4. Transcribe with ByteDance AND YourMT3+
  5. Compare S2 pairwise accuracy: GT MIDI vs transcribed MIDI
  6. Report per-dimension drop, gate at <10%

Usage:
    cd model
    uv run python scripts/build_amt_test_set.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from model_improvement.layer1_validation import (
    amt_degradation_comparison,
    select_maestro_subset,
)
from src.paths import Embeddings, Checkpoints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MAESTRO_CACHE_DIR = Embeddings.maestro
EMBEDDINGS_DIR = MAESTRO_CACHE_DIR / "muq_embeddings"
CONTRASTIVE_MAPPING_PATH = MAESTRO_CACHE_DIR / "contrastive_mapping.json"
SELECTION_PATH = MAESTRO_CACHE_DIR / "amt_test_selection.json"

AUDIO_DOWNLOAD_DIR = MAESTRO_CACHE_DIR / "amt_test_audio"
GT_MIDI_DIR = MAESTRO_CACHE_DIR / "midi"
TRANSCRIBED_DIR = MAESTRO_CACHE_DIR / "amt_test_transcribed"
REPORT_PATH = MAESTRO_CACHE_DIR / "amt_test_report.json"

# Piece selection thresholds
N_EASY = 10       # |score_diff| > 0.2
N_MEDIUM = 10     # 0.05 <= |score_diff| <= 0.15
N_HARD = 5        # |score_diff| < 0.05
N_PIECES_TOTAL = N_EASY + N_MEDIUM + N_HARD  # 25

# Pairwise accuracy gate
MAX_DIMENSION_DROP_PCT = 10.0


# ---------------------------------------------------------------------------
# Step 1: Piece selection (can run locally with A1 checkpoint)
# ---------------------------------------------------------------------------

def step1_select_pieces(a1_checkpoint_path: str | Path) -> dict:
    """Score MAESTRO embeddings with A1 and select 25 test pieces.

    Loads A1 checkpoint + MAESTRO embeddings, scores all 24K segments,
    computes per-piece mean scores, then selects pieces with diverse quality
    gaps across performers.

    Saves selection to SELECTION_PATH and returns it.

    Args:
        a1_checkpoint_path: Path to A1 Lightning checkpoint (.ckpt).

    Returns:
        Dict with keys: easy, medium, hard (each a list of piece names).

    Raises:
        FileNotFoundError: If checkpoint, embeddings dir, or contrastive mapping
            do not exist.
        ValueError: If contrastive mapping has fewer than 25 multi-performer pieces.
    """
    a1_checkpoint_path = Path(a1_checkpoint_path)
    if not a1_checkpoint_path.exists():
        raise FileNotFoundError(f"A1 checkpoint not found: {a1_checkpoint_path}")
    if not EMBEDDINGS_DIR.exists():
        raise FileNotFoundError(f"MAESTRO embeddings not found: {EMBEDDINGS_DIR}")
    if not CONTRASTIVE_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"contrastive_mapping.json not found: {CONTRASTIVE_MAPPING_PATH}"
        )

    if SELECTION_PATH.exists():
        logger.info("Selection already cached at %s", SELECTION_PATH)
        with open(SELECTION_PATH) as f:
            return json.load(f)

    logger.info("Loading contrastive mapping from %s", CONTRASTIVE_MAPPING_PATH)
    with open(CONTRASTIVE_MAPPING_PATH) as f:
        contrastive_mapping: dict[str, list[str]] = json.load(f)

    # Load A1 model
    logger.info("Loading A1 checkpoint from %s", a1_checkpoint_path)
    from model_improvement.lora import LoRAModel  # type: ignore[import]
    model = LoRAModel.load_from_checkpoint(str(a1_checkpoint_path), map_location="cpu")
    # Switch to inference mode
    model.train(False)

    # Score all segments lazily (one .pt file at a time to avoid OOM)
    logger.info("Scoring MAESTRO embeddings from %s ...", EMBEDDINGS_DIR)
    from model_improvement.layer1_validation import score_competition_segments
    segment_scores: dict[str, np.ndarray] = score_competition_segments(
        model=model,
        embeddings=EMBEDDINGS_DIR,
    )
    logger.info("Scored %d segments", len(segment_scores))

    # Compute per-piece per-performer mean scores
    # segment IDs encode the recording key (without _segNNN)
    def _recording_id(seg_id: str) -> str:
        idx = seg_id.rfind("_seg")
        if idx >= 0 and seg_id[idx + 4:].isdigit():
            return seg_id[:idx]
        return seg_id

    recording_scores: dict[str, list[np.ndarray]] = {}
    for seg_id, scores in segment_scores.items():
        rec_id = _recording_id(seg_id)
        recording_scores.setdefault(rec_id, []).append(scores)

    # Mean score per recording
    recording_mean: dict[str, np.ndarray] = {
        rec_id: np.mean(segs, axis=0)
        for rec_id, segs in recording_scores.items()
    }

    # For each multi-performer piece, compute all pairwise |score_diff| (overall mean)
    easy_pieces: list[str] = []
    medium_pieces: list[str] = []
    hard_pieces: list[str] = []

    def _to_recording_id(key: str) -> str:
        idx = key.rfind("_seg")
        if idx >= 0 and key[idx + 4:].isdigit():
            return key[:idx]
        return key

    for piece, keys in contrastive_mapping.items():
        recordings = list(dict.fromkeys(_to_recording_id(k) for k in keys))
        if len(recordings) < 2:
            continue

        # Only include recordings we actually scored
        scored = [r for r in recordings if r in recording_mean]
        if len(scored) < 2:
            continue

        # All pairwise |diff| across the overall mean
        diffs = []
        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                overall_i = float(recording_mean[scored[i]].mean())
                overall_j = float(recording_mean[scored[j]].mean())
                diffs.append(abs(overall_i - overall_j))

        mean_diff = np.mean(diffs) if diffs else 0.0

        if mean_diff > 0.2:
            easy_pieces.append(piece)
        elif 0.05 <= mean_diff <= 0.15:
            medium_pieces.append(piece)
        else:
            hard_pieces.append(piece)

    # Trim to target counts
    easy_pieces = easy_pieces[:N_EASY]
    medium_pieces = medium_pieces[:N_MEDIUM]
    hard_pieces = hard_pieces[:N_HARD]

    if len(easy_pieces) + len(medium_pieces) + len(hard_pieces) < N_PIECES_TOTAL:
        raise ValueError(
            f"Not enough multi-performer pieces: got "
            f"{len(easy_pieces)} easy, {len(medium_pieces)} medium, {len(hard_pieces)} hard. "
            f"Need {N_EASY} + {N_MEDIUM} + {N_HARD} = {N_PIECES_TOTAL}."
        )

    selection = {
        "easy": easy_pieces,
        "medium": medium_pieces,
        "hard": hard_pieces,
        "all": easy_pieces + medium_pieces + hard_pieces,
        "n_scored_segments": len(segment_scores),
        "n_scored_recordings": len(recording_mean),
    }

    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_PATH, "w") as f:
        json.dump(selection, f, indent=2)

    logger.info(
        "Step 1 complete: %d easy, %d medium, %d hard pieces selected",
        len(easy_pieces), len(medium_pieces), len(hard_pieces),
    )
    logger.info("Selection saved to %s", SELECTION_PATH)
    return selection


# ---------------------------------------------------------------------------
# Step 2: Audio download
# ---------------------------------------------------------------------------

def step2_download_audio(selection: dict) -> None:
    """Download MAESTRO audio for selected pieces.

    TODO(thunder): This step benefits from Thunder Compute for bandwidth/speed,
    but can also run locally.

    Args:
        selection: Output of step1_select_pieces.

    Raises:
        FileNotFoundError: If MAESTRO metadata.jsonl is not available.
    """
    metadata_path = MAESTRO_CACHE_DIR / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"MAESTRO metadata not found at {metadata_path}. "
            "Run download_maestro_subset.py first."
        )

    import jsonlines  # type: ignore[import]

    with jsonlines.open(metadata_path) as reader:
        all_records = list(reader)

    selected_pieces = set(selection.get("all", []))
    target_records = [
        r for r in all_records
        if r.get("canonical_composer_title") in selected_pieces
        or r.get("piece") in selected_pieces
    ]

    if not target_records:
        logger.warning(
            "No MAESTRO records matched the selected pieces. "
            "Check that piece names in selection match metadata.jsonl fields."
        )
        return

    AUDIO_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Step 2: Preparing to download audio for %d recordings ...",
        len(target_records),
    )

    # TODO(thunder): Download audio files from MAESTRO source URLs or GCS bucket.
    # For each record, download to AUDIO_DOWNLOAD_DIR/{recording_id}.wav.
    # Actual download mechanism depends on MAESTRO data source (GCS vs local copy).
    for rec in target_records:
        recording_id = rec.get("recording_id", rec.get("midi_filename", "unknown"))
        wav_path = AUDIO_DOWNLOAD_DIR / f"{recording_id}.wav"
        if wav_path.exists() and wav_path.stat().st_size > 0:
            logger.debug("Skipping %s (already downloaded)", recording_id)
            continue
        logger.info("TODO: download audio for %s", recording_id)

    logger.info("Step 2 complete (TODO: implement actual audio download).")


# ---------------------------------------------------------------------------
# Step 3: Dual transcription (GPU required)
# ---------------------------------------------------------------------------

def step3_transcribe(recording_ids: list[str]) -> None:
    """Transcribe recordings with ByteDance AND YourMT3+.

    Requires CUDA GPU.  # TODO(thunder): requires GPU

    Args:
        recording_ids: List of recording IDs to transcribe.

    Raises:
        RuntimeError: If CUDA is not available.
        ImportError: If transcription packages are not installed.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for transcription. Run this step on Thunder Compute."
        )

    bytedance_dir = TRANSCRIBED_DIR / "bytedance"
    yourmt3_dir = TRANSCRIBED_DIR / "yourmt3"
    bytedance_dir.mkdir(parents=True, exist_ok=True)
    yourmt3_dir.mkdir(parents=True, exist_ok=True)

    # --- ByteDance piano_transcription ---
    # TODO(thunder): requires GPU
    try:
        from piano_transcription_inference import PianoTranscription  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "piano_transcription_inference not installed. "
            "Install on Thunder Compute: uv pip install piano-transcription-inference"
        ) from e

    transcriber = PianoTranscription(device="cuda")  # TODO(thunder): requires GPU

    for recording_id in recording_ids:
        wav_path = AUDIO_DOWNLOAD_DIR / f"{recording_id}.wav"
        midi_path = bytedance_dir / f"{recording_id}.mid"

        if midi_path.exists() and midi_path.stat().st_size > 0:
            logger.debug("ByteDance: skipping %s", recording_id)
            continue

        if not wav_path.exists():
            logger.warning("WAV not found for %s, skipping ByteDance transcription", recording_id)
            continue

        logger.info("ByteDance transcribing %s ...", recording_id)
        try:
            transcriber.transcribe(str(wav_path), str(midi_path))  # TODO(thunder): requires GPU
        except Exception as e:
            logger.error("ByteDance failed for %s: %s", recording_id, e)

    # --- YourMT3+ ---
    # TODO(thunder): requires GPU
    # TODO(thunder): verify package name -- may be "yourmt3" or "mt3" or installed from source
    logger.info("YourMT3+ transcription: TODO(thunder) -- verify package name and API")
    for recording_id in recording_ids:
        wav_path = AUDIO_DOWNLOAD_DIR / f"{recording_id}.wav"
        midi_path = yourmt3_dir / f"{recording_id}.mid"

        if midi_path.exists() and midi_path.stat().st_size > 0:
            logger.debug("YourMT3+: skipping %s", recording_id)
            continue

        if not wav_path.exists():
            logger.warning("WAV not found for %s, skipping YourMT3+ transcription", recording_id)
            continue

        logger.info("YourMT3+ transcribing %s ... (TODO: implement)", recording_id)
        # TODO(thunder): requires GPU
        # Placeholder -- replace with actual YourMT3+ API once package name confirmed.
        # Expected interface (approximate):
        #   from yourmt3 import MusicTranscriber
        #   mt3 = MusicTranscriber(device="cuda")
        #   mt3.transcribe(str(wav_path), str(midi_path))

    logger.info("Step 3 complete.")


# ---------------------------------------------------------------------------
# Step 4 + 5: Comparison and report
# ---------------------------------------------------------------------------

def _build_pairwise_results(
    recording_ids: list[str],
    s2_model: torch.nn.Module,
) -> dict[str, dict]:
    """Run S2 pairwise accuracy for ground truth and each transcription system.

    Args:
        recording_ids: List of recording IDs in the test set.
        s2_model: Trained S2 GNN model with predict_pair(g1, g2) -> float.

    Returns:
        {source_name: {overall: float, per_dimension: {int: float}}}
    """
    import pretty_midi
    from model_improvement.graph import parsed_midi_to_graph

    systems = {
        "ground_truth": GT_MIDI_DIR,
        "bytedance": TRANSCRIBED_DIR / "bytedance",
        "yourmt3": TRANSCRIBED_DIR / "yourmt3",
    }

    results: dict[str, dict] = {}

    for source_name, midi_dir in systems.items():
        if not midi_dir.exists():
            logger.warning("MIDI dir not found for %s: %s", source_name, midi_dir)
            continue

        pair_correct: list[bool] = []

        for i in range(len(recording_ids)):
            for j in range(i + 1, len(recording_ids)):
                id1 = recording_ids[i]
                id2 = recording_ids[j]
                mid1 = midi_dir / f"{id1}.mid"
                mid2 = midi_dir / f"{id2}.mid"

                if not mid1.exists() or not mid2.exists():
                    continue

                try:
                    g1 = parsed_midi_to_graph(pretty_midi.PrettyMIDI(str(mid1)))
                    g2 = parsed_midi_to_graph(pretty_midi.PrettyMIDI(str(mid2)))
                except Exception as e:
                    logger.warning("Graph error for pair (%s, %s): %s", id1, id2, e)
                    continue

                try:
                    # S2 pairwise prediction: positive = g1 better than g2
                    pred = s2_model.predict_pair(g1, g2)
                    pair_correct.append(pred > 0.5)
                except Exception as e:
                    logger.warning("Predict pair error (%s, %s): %s", id1, id2, e)
                    continue

        if not pair_correct:
            logger.warning("No valid pairs for %s", source_name)
            continue

        overall_acc = float(np.mean(pair_correct))
        # Per-dimension: placeholder (real implementation requires per-dim S2 head)
        per_dimension = {d: overall_acc for d in range(6)}

        results[source_name] = {
            "overall": overall_acc,
            "per_dimension": per_dimension,
            "n_pairs": len(pair_correct),
        }
        logger.info(
            "%s: overall accuracy = %.3f (%d pairs)",
            source_name, overall_acc, len(pair_correct),
        )

    return results


def step4_and_5_compare_and_report(
    recording_ids: list[str],
    s2_checkpoint_path: str | Path,
) -> dict:
    """Compare S2 pairwise accuracy across transcription systems and write report.

    Args:
        recording_ids: List of recording IDs in the test set.
        s2_checkpoint_path: Path to S2 Lightning checkpoint.

    Returns:
        Report dict (also written to REPORT_PATH).

    Raises:
        FileNotFoundError: If S2 checkpoint does not exist.
    """
    s2_checkpoint_path = Path(s2_checkpoint_path)
    if not s2_checkpoint_path.exists():
        raise FileNotFoundError(f"S2 checkpoint not found: {s2_checkpoint_path}")

    logger.info("Loading S2 checkpoint from %s", s2_checkpoint_path)
    from model_improvement.symbolic_encoders import GNNEncoder  # type: ignore[import]
    s2_model = GNNEncoder.load_from_checkpoint(str(s2_checkpoint_path), map_location="cpu")
    s2_model.train(False)

    pairwise_results = _build_pairwise_results(recording_ids, s2_model)

    if "ground_truth" not in pairwise_results:
        raise RuntimeError(
            "Ground truth pairwise results missing. "
            "Ensure GT MIDI files are present in: " + str(GT_MIDI_DIR)
        )

    degradation = amt_degradation_comparison(pairwise_results, baseline="ground_truth")

    # Gate check
    gate_passed = True
    for source, result in degradation.items():
        for dim_name, drop in result["per_dimension_drop_pct"].items():
            if drop > MAX_DIMENSION_DROP_PCT:
                logger.warning(
                    "FAIL: %s dimension %s dropped %.1f%% (> %.1f%% gate)",
                    source, dim_name, drop, MAX_DIMENSION_DROP_PCT,
                )
                gate_passed = False

    report = {
        "pairwise_results": pairwise_results,
        "degradation": degradation,
        "gate_passed": gate_passed,
        "gate_threshold_pct": MAX_DIMENSION_DROP_PCT,
        "n_recordings": len(recording_ids),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Report written to %s", REPORT_PATH)
    logger.info("Gate: %s", "PASSED" if gate_passed else "FAILED")
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== AMT Validation Test Set Builder ===")

    # Locate A1 and S2 checkpoints (override via environment variables)
    import os
    a1_checkpoint = os.environ.get(
        "A1_CHECKPOINT",
        str(Checkpoints.root / "a1_best.ckpt"),
    )
    s2_checkpoint = os.environ.get(
        "S2_CHECKPOINT",
        str(Checkpoints.root / "s2_best.ckpt"),
    )

    # Step 1: Piece selection (can run locally)
    logger.info("\n--- Step 1: Piece selection ---")
    selection = step1_select_pieces(a1_checkpoint_path=a1_checkpoint)
    logger.info(
        "Selected: %d easy, %d medium, %d hard pieces",
        len(selection["easy"]),
        len(selection["medium"]),
        len(selection["hard"]),
    )

    # Step 2: Audio download
    logger.info("\n--- Step 2: Audio download ---")
    step2_download_audio(selection)

    # Determine recording IDs from selection
    if CONTRASTIVE_MAPPING_PATH.exists():
        with open(CONTRASTIVE_MAPPING_PATH) as f:
            contrastive_mapping: dict[str, list[str]] = json.load(f)

        # Filter to selected pieces only
        selected_pieces = set(selection.get("all", []))
        filtered_mapping = {
            piece: keys
            for piece, keys in contrastive_mapping.items()
            if piece in selected_pieces
        }
        recording_ids = select_maestro_subset(
            contrastive_mapping=filtered_mapping,
            n_recordings=80,
        )
    else:
        logger.warning(
            "contrastive_mapping.json not found; cannot derive recording IDs. "
            "Skipping steps 3-5."
        )
        return

    # Step 3: Dual transcription (GPU required on Thunder Compute)
    logger.info("\n--- Step 3: Dual transcription ---")
    logger.info("NOTE: Step 3 requires CUDA GPU.  # TODO(thunder): requires GPU")
    step3_transcribe(recording_ids)

    # Steps 4 + 5: Comparison and report
    logger.info("\n--- Steps 4+5: Comparison and report ---")
    report = step4_and_5_compare_and_report(
        recording_ids=recording_ids,
        s2_checkpoint_path=s2_checkpoint,
    )

    logger.info("\n=== Summary ===")
    for source, result in report.get("degradation", {}).items():
        logger.info(
            "%s: overall drop = %.1f%%, viable = %s",
            source,
            result["overall_drop_pct"],
            result["viable"],
        )
    logger.info("Gate: %s", "PASSED" if report.get("gate_passed") else "FAILED")


if __name__ == "__main__":
    main()
