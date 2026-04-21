"""Pre-render 1500 corrupted MAESTRO clips and extract MuQ embeddings.

For each of N_CLIPS MAESTRO segments (sampled from the existing metadata.jsonl):
  1. Locate the corresponding MIDI file in the MAESTRO raw directory.
  2. Apply a random combination of MIDI corruption primitives via practice_synthesis.
  3. Synthesise to audio at 24 kHz via FluidSynth through pretty_midi.
  4. Convolve with a synthetic practice-room IR (or real IR if room_irs/ exists).
  5. Extract MuQ embeddings.
  6. Save embeddings as data/embeddings/practice_corrupted/corrupt_{segment_id}.pt
  7. Also copy the corresponding clean embedding from data/embeddings/maestro/
     muq_embeddings/{segment_id}.pt into the same directory as {segment_id}.pt
     so that the OOD labels file can form clean-vs-corrupted pairs.

After all clips are processed, writes:
  data/evals/ood_practice/labels.json
    {
      "segment_id_1":         {"scores": [0.75, ...], "ordinal": null},
      "corrupt_segment_id_1": {"scores": [0.30, ...], "ordinal": null},
      ...
    }
  data/evals/ood_practice/metadata.jsonl (segment-level provenance)

Usage:
    cd model/
    uv run python scripts/render_corrupted_audio.py [--n-clips 1500] \\
        [--soundfont /path/to/soundfont.sf2] [--seed 42] [--split train]

Requirements:
    - FluidSynth must be installed: brew install fluidsynth (mac) / apt-get install fluidsynth
    - A soundfont (.sf2) is needed for synthesis. Download e.g. FluidR3_GM.sf2 or
      Salamander Grand Piano (recommended for piano-only output).
    - MAESTRO must be downloaded to data/raw/maestro/.
    - MuQ embeddings for clean MAESTRO must exist in data/embeddings/maestro/muq_embeddings/.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import jsonlines
import numpy as np
import torch

# Ensure project src is importable when running from model/scripts/
REPO_MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_MODEL_DIR / "src"))

from audio_experiments.extractors.muq import MuQExtractor
from model_improvement.augmentation import AudioAugmentor
from model_improvement.practice_synthesis import apply_practice_corruptions
from src.paths import Embeddings, Evals, Raw

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
CLEAN_SCORES = [0.75] * 6
CORRUPT_SCORES = [0.30] * 6


def _check_fluidsynth() -> None:
    """Raise a clear error if FluidSynth is not installed."""
    import shutil
    if shutil.which("fluidsynth") is None:
        raise RuntimeError(
            "FluidSynth not found. Install it first:\n"
            "  macOS: brew install fluidsynth\n"
            "  Linux: sudo apt-get install fluidsynth\n"
            "Then re-run this script."
        )


def _synthesise_midi(
    midi_obj,  # pretty_midi.PrettyMIDI — imported lazily inside render_corrupted_clips
    soundfont: str | None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Render a PrettyMIDI object to a mono float32 waveform at sample_rate.

    Args:
        midi_obj: PrettyMIDI object to render.
        soundfont: Path to .sf2 soundfont, or None to let pretty_midi find one.
        sample_rate: Target sample rate in Hz.

    Returns:
        1D float32 numpy array.
    """
    kwargs: dict = {"fs": sample_rate}
    if soundfont:
        kwargs["sf2_path"] = soundfont

    audio = midi_obj.fluidsynth(**kwargs)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    return audio.astype(np.float32)


def _apply_room_ir(
    audio: np.ndarray,
    augmentor: AudioAugmentor,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convolve audio with a practice-room IR via AudioAugmentor.

    Forces IR convolution by temporarily setting IR_CONV_PROB to 1.0
    on a throwaway augmentor with augment_prob=1.0.
    """
    waveform = torch.from_numpy(audio).unsqueeze(0).float()  # [1, T]
    # Build a one-shot augmentor that always applies IR and nothing else
    ir_augmentor = AudioAugmentor.__new__(AudioAugmentor)
    ir_augmentor.augment_prob = 1.0
    ir_augmentor.ir_dir = augmentor.ir_dir
    ir_augmentor._ir_paths = augmentor._ir_paths

    result_np = ir_augmentor._apply_room_ir(waveform.numpy(), sample_rate)
    return result_np.squeeze(0) if result_np.ndim == 2 else result_np


def _load_metadata(maestro_emb_dir: Path) -> list[dict]:
    """Load existing MAESTRO segment metadata from the embedding cache."""
    metadata_path = maestro_emb_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No MAESTRO metadata at {metadata_path}. "
            "Run segment_and_embed_maestro() first."
        )
    with jsonlines.open(metadata_path) as reader:
        return list(reader)


def _sample_segments(
    metadata: list[dict],
    n_clips: int,
    split: str | None,
    seed: int,
) -> list[dict]:
    """Sample up to n_clips segments, balanced across pieces."""
    if split:
        metadata = [m for m in metadata if m.get("split") == split]
    if not metadata:
        raise ValueError(
            f"No segments found (split filter: {split!r}). "
            "Check that MAESTRO is segmented and split values are correct."
        )

    rng = random.Random(seed)
    # Group by piece to ensure variety
    by_piece: dict[str, list[dict]] = {}
    for seg in metadata:
        title = seg.get("canonical_title", "unknown")
        by_piece.setdefault(title, []).append(seg)

    pieces = sorted(by_piece.keys())
    rng.shuffle(pieces)

    selected: list[dict] = []
    idx = 0
    while len(selected) < n_clips and idx < len(pieces) * 1000:
        piece = pieces[idx % len(pieces)]
        pool = by_piece[piece]
        seg = rng.choice(pool)
        # Avoid duplicates
        if seg["segment_id"] not in {s["segment_id"] for s in selected}:
            selected.append(seg)
        idx += 1

    return selected[:n_clips]


def render_corrupted_clips(
    n_clips: int = 1500,
    soundfont: str | None = None,
    seed: int = 42,
    split: str | None = "train",
    maestro_dir: Path = Raw.maestro,
    maestro_emb_dir: Path = Embeddings.maestro,
    output_emb_dir: Path = Embeddings.practice_corrupted,
    ood_dir: Path = Evals.ood_practice,
    ir_dir: Path = Raw.room_irs,
) -> None:
    """Main pipeline: corrupt MIDI, synthesise, embed, write labels."""
    import pretty_midi

    _check_fluidsynth()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_emb_dir.mkdir(parents=True, exist_ok=True)
    ood_dir.mkdir(parents=True, exist_ok=True)

    # AudioAugmentor only used for room IR; all other augments off
    ir_path = ir_dir if ir_dir.exists() else None
    augmentor = AudioAugmentor(augment_prob=1.0, ir_dir=ir_path)

    # Load MAESTRO segment metadata
    metadata = _load_metadata(maestro_emb_dir)
    logger.info("Loaded %d MAESTRO segments from metadata.jsonl", len(metadata))

    segments = _sample_segments(metadata, n_clips, split, seed)
    logger.info("Sampled %d segments for corruption", len(segments))

    extractor = MuQExtractor(cache_dir=output_emb_dir)

    # Check which are already processed
    existing_corrupt = {
        p.stem for p in output_emb_dir.glob("corrupt_*.pt")
    }

    labels: dict[str, dict] = {}
    provenance: list[dict] = []

    rng = random.Random(seed)
    n_rendered = 0
    n_skipped = 0
    n_clean_missing = 0

    for i, seg in enumerate(segments):
        seg_id = seg["segment_id"]
        corrupt_id = f"corrupt_{seg_id}"

        # --- Copy / verify clean embedding ---
        clean_emb_path = maestro_emb_dir / "muq_embeddings" / f"{seg_id}.pt"
        clean_dest = output_emb_dir / f"{seg_id}.pt"

        if not clean_emb_path.exists():
            logger.warning("[%d/%d] Clean embedding missing: %s", i + 1, len(segments), seg_id)
            n_clean_missing += 1
            continue

        if not clean_dest.exists():
            import shutil
            shutil.copy2(clean_emb_path, clean_dest)

        # --- Render corrupted version if not cached ---
        if corrupt_id not in existing_corrupt:
            midi_rel = seg.get("audio_filename", "").replace(".wav", ".midi")
            if not midi_rel:
                midi_rel = seg.get("midi_filename", "")

            midi_path = maestro_dir / midi_rel
            if not midi_path.exists():
                # Try .mid extension
                midi_path = midi_path.with_suffix(".mid")
            if not midi_path.exists():
                logger.warning("[%d/%d] MIDI not found for %s", i + 1, len(segments), seg_id)
                n_skipped += 1
                continue

            try:
                midi_obj = pretty_midi.PrettyMIDI(str(midi_path))
            except Exception as exc:
                logger.warning("[%d/%d] MIDI parse error %s: %s", i + 1, len(segments), midi_path, exc)
                n_skipped += 1
                continue

            # Crop MIDI to the segment's time window
            seg_start = seg.get("segment_start", 0.0)
            seg_end = seg.get("segment_end", seg_start + 30.0)
            cropped = pretty_midi.PrettyMIDI()
            for instrument in midi_obj.instruments:
                new_inst = pretty_midi.Instrument(
                    program=instrument.program,
                    is_drum=instrument.is_drum,
                    name=instrument.name,
                )
                for note in instrument.notes:
                    if note.end < seg_start or note.start > seg_end:
                        continue
                    clipped = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=max(0.0, note.start - seg_start),
                        end=min(seg_end - seg_start, note.end - seg_start),
                    )
                    new_inst.notes.append(clipped)
                if new_inst.notes:
                    cropped.instruments.append(new_inst)

            if not any(inst.notes for inst in cropped.instruments):
                logger.warning("[%d/%d] Empty MIDI crop for %s", i + 1, len(segments), seg_id)
                n_skipped += 1
                continue

            # Apply MIDI corruptions
            corrupt_midi = apply_practice_corruptions(
                cropped,
                rng=random.Random(rng.randint(0, 2**31 - 1)),
            )

            # Synthesise
            try:
                audio = _synthesise_midi(corrupt_midi, soundfont=soundfont)
            except Exception as exc:
                logger.warning("[%d/%d] Synthesis failed for %s: %s", i + 1, len(segments), seg_id, exc)
                n_skipped += 1
                continue

            if len(audio) < SAMPLE_RATE:
                logger.warning("[%d/%d] Synthesised audio too short: %s", i + 1, len(segments), seg_id)
                n_skipped += 1
                continue

            # Practice-room IR convolution
            audio = _apply_room_ir(audio, augmentor, sample_rate=SAMPLE_RATE)

            # Extract MuQ embedding
            audio_tensor = torch.from_numpy(audio).float()
            embedding = extractor.extract_from_audio(audio_tensor)

            torch.save(embedding, output_emb_dir / f"{corrupt_id}.pt")  # nosemgrep
            existing_corrupt.add(corrupt_id)
            n_rendered += 1

        # Record OOD labels for both clean and corrupted
        labels[seg_id] = {"scores": CLEAN_SCORES, "ordinal": None}
        labels[corrupt_id] = {"scores": CORRUPT_SCORES, "ordinal": None}

        provenance.append({
            "segment_id": seg_id,
            "corrupt_segment_id": corrupt_id,
            "canonical_title": seg.get("canonical_title", ""),
            "canonical_composer": seg.get("canonical_composer", ""),
            "split": seg.get("split", ""),
            "segment_start": seg.get("segment_start", 0.0),
            "segment_end": seg.get("segment_end", 0.0),
        })

        if (i + 1) % 100 == 0:
            logger.info(
                "[%d/%d] rendered=%d skipped=%d clean_missing=%d",
                i + 1, len(segments), n_rendered, n_skipped, n_clean_missing,
            )

    del extractor

    # Write OOD labels
    labels_path = ood_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    logger.info("Wrote %d OOD label entries to %s", len(labels), labels_path)

    # Write provenance metadata
    meta_path = ood_dir / "metadata.jsonl"
    with jsonlines.open(meta_path, mode="w") as writer:
        writer.write_all(provenance)
    logger.info("Wrote provenance to %s", meta_path)

    logger.info(
        "Done. rendered=%d  skipped=%d  clean_missing=%d  total_labels=%d",
        n_rendered, n_skipped, n_clean_missing, len(labels),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-clips", type=int, default=1500,
        help="Number of MAESTRO clips to corrupt (default: 1500)",
    )
    parser.add_argument(
        "--soundfont", type=str, default=None,
        help="Path to .sf2 soundfont for FluidSynth synthesis",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling and corruption (default: 42)",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "validation", "test", "all"],
        help="Which MAESTRO split to sample from (default: train)",
    )
    args = parser.parse_args()

    render_corrupted_clips(
        n_clips=args.n_clips,
        soundfont=args.soundfont,
        seed=args.seed,
        split=None if args.split == "all" else args.split,
    )


if __name__ == "__main__":
    main()
