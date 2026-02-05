"""Extract MuQ embeddings from rendered ASAP audio.

Standalone script version of notebook cell 14 from
notebooks/score_alignment/01_alignment_exploration.ipynb.

Run from model/ directory:
    python scripts/extract_muq_embeddings.py

Logs progress to stdout so you can watch it in a separate terminal tab.
"""

import gc
import sys
import time
from pathlib import Path

import torch

# Add src to path
MODEL_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = MODEL_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from audio_experiments.extractors.muq import MuQExtractor
from score_alignment.data.asap import parse_asap_metadata, get_performance_key
from score_alignment.data.midi_render import get_render_jobs_for_asap

# --- Paths (same as notebook) ---
DATA_ROOT = MODEL_ROOT / "data"
ASAP_ROOT = DATA_ROOT / "asap-dataset"
AUDIO_CACHE = DATA_ROOT / "audio_cache"
SCORE_CACHE_DIR = DATA_ROOT / "muq_cache" / "scores"
PERF_CACHE_DIR = DATA_ROOT / "muq_cache" / "performances"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def extract_dir(audio_dir: Path, cache_dir: Path, label: str) -> int:
    """Extract MuQ embeddings for all WAV files in audio_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(audio_dir.glob("*.wav"))
    cached = {p.stem for p in cache_dir.glob("*.pt")}
    to_extract = [f for f in wav_files if f.stem not in cached]

    if not to_extract:
        log(f"{label}: all {len(wav_files)} embeddings already cached")
        return 0

    log(f"{label}: {len(to_extract)} to extract ({len(cached)} cached)")
    log(f"{label}: loading MuQ model...")
    extractor = MuQExtractor(cache_dir=cache_dir)
    log(f"{label}: model loaded on {extractor.device}")

    failed = 0
    for i, wav_file in enumerate(to_extract, 1):
        t0 = time.time()
        try:
            extractor.extract_from_file(wav_file)
            dt = time.time() - t0
            log(f"{label}: [{i}/{len(to_extract)}] {wav_file.stem} ({dt:.1f}s)")
        except Exception as e:
            failed += 1
            log(f"{label}: [{i}/{len(to_extract)}] FAILED {wav_file.stem}: {e}")

    extracted = len(to_extract) - failed
    log(f"{label}: done -- {extracted} extracted, {failed} failed")

    # Free model memory
    del extractor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return extracted


def main() -> None:
    log("MuQ embedding extraction script")
    log(f"Data root: {DATA_ROOT}")
    log(f"Audio cache: {AUDIO_CACHE}")

    # Verify ASAP dataset
    if not ASAP_ROOT.exists():
        log(f"ERROR: ASAP dataset not found at {ASAP_ROOT}")
        sys.exit(1)

    # Parse metadata to get counts (for logging only)
    asap_index = parse_asap_metadata(ASAP_ROOT)
    aligned_perfs = asap_index.filter_with_alignments()
    log(f"ASAP: {len(aligned_perfs)} aligned performances")

    scores_dir = AUDIO_CACHE / "scores"
    perfs_dir = AUDIO_CACHE / "performances"

    if not scores_dir.exists() or not perfs_dir.exists():
        log(f"ERROR: rendered audio not found in {AUDIO_CACHE}")
        log("Run the MIDI rendering cells in the notebook first.")
        sys.exit(1)

    log(f"Score WAVs: {len(list(scores_dir.glob('*.wav')))}")
    log(f"Performance WAVs: {len(list(perfs_dir.glob('*.wav')))}")

    t_start = time.time()

    # Extract scores
    log("=" * 50)
    extract_dir(scores_dir, SCORE_CACHE_DIR, "SCORES")

    # Extract performances
    log("=" * 50)
    extract_dir(perfs_dir, PERF_CACHE_DIR, "PERFS")

    # Summary
    log("=" * 50)
    total_time = time.time() - t_start
    score_count = len(list(SCORE_CACHE_DIR.glob("*.pt")))
    perf_count = len(list(PERF_CACHE_DIR.glob("*.pt")))
    log(f"Total cached: {score_count} scores, {perf_count} performances")
    log(f"Total time: {total_time / 60:.1f} minutes")


if __name__ == "__main__":
    main()
