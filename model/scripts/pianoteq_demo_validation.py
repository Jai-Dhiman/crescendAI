#!/usr/bin/env python3
"""
Pianoteq Demo Mode Validation Script
=====================================

GOAL: Determine if Pianoteq's demo mode (with 9.5% silent notes) produces
acceptable audio for MuQ embedding extraction in piano performance evaluation.

CONTEXT:
- We want to augment training data with multiple piano soundfonts for timbre-invariance
- Pianoteq demo has 8 disabled notes: F#1, G#1, A#1, C#5, D#5, F#5, G#5, A#5
- These represent ~9.5% of notes in the PercePiano dataset
- Our model measures mood, pedaling, and technique - NOT note accuracy
- Question: Do silent notes create artifacts in MuQ embeddings?

PRESETS TO TEST (6 total):
1. NY Steinway Model D     - Concert grand (bright American)
2. HB Steinway Model D     - Concert grand (warm European)
3. NY Steinway D Honky Tonk - Out-of-tune bar piano
4. NY Steinway D Worn Out   - Degraded/old piano
5. U4 Small                 - Upright piano
6. K2 Basic                 - Kawai upright

VALIDATION APPROACH:
1. Render 50-100 test MIDIs with each preset using Pianoteq demo
2. Extract MuQ embeddings from rendered audio
3. Compare embeddings to original PercePiano rendered audio (same performances)
4. Analyze: Do silent notes cause significant embedding differences?

SUCCESS CRITERIA:
- Embedding cosine similarity > 0.85 between demo and reference audio
- No systematic artifacts in embedding space
- Per-dimension correlation > 0.8 for key perceptual dimensions

REQUIREMENTS:
- Pianoteq 9 installed at: /Applications/Pianoteq 9/Pianoteq 9.app/Contents/MacOS/Pianoteq 9
- MuQ model (will be downloaded automatically)
- PercePiano MIDI files at: model/data/raw/PercePiano/virtuoso/data/all_2rounds/
- Reference rendered audio (optional): gdrive:crescendai_data/audio_baseline/percepiano_rendered

Author: Claude Code
Date: 2025-01-19
"""

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project source to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model" / "src"))


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    "pianoteq_path": "/Applications/Pianoteq 9/Pianoteq 9.app/Contents/MacOS/Pianoteq 9",
    "midi_dir": PROJECT_ROOT / "model" / "data" / "raw" / "PercePiano" / "virtuoso" / "data" / "all_2rounds",
    "output_dir": PROJECT_ROOT / "model" / "data" / "pianoteq_validation",
    "reference_audio_dir": PROJECT_ROOT / "model" / "data" / "audio_baseline" / "percepiano_rendered",

    # Rendering settings
    "sample_rate": 24000,
    "test_sample_size": 50,  # Number of MIDIs to test
    "random_seed": 42,

    # Presets to test
    "presets": [
        "NY Steinway Model D",
        "HB Steinway Model D",
        "NY Steinway D Honky Tonk",
        "NY Steinway D Worn Out",
        "U4 Small",
        "K2 Basic",
    ],

    # MuQ settings
    "muq_layer_start": 9,
    "muq_layer_end": 13,  # Layers 9-12 (exclusive end)

    # Validation thresholds
    "min_cosine_similarity": 0.85,
    "min_dimension_correlation": 0.80,

    # Disabled notes in demo mode (MIDI note numbers)
    "disabled_notes": {30, 32, 34, 73, 75, 78, 80, 82},  # F#1, G#1, A#1, C#5, D#5, F#5, G#5, A#5
}


# =============================================================================
# STEP 1: MIDI SELECTION AND ANALYSIS
# =============================================================================

def analyze_midi_for_disabled_notes(midi_path: Path) -> Dict:
    """Analyze a MIDI file for disabled demo notes."""
    import mido

    disabled_notes = CONFIG["disabled_notes"]
    total_notes = 0
    disabled_count = 0
    disabled_breakdown = defaultdict(int)

    mid = mido.MidiFile(str(midi_path))
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                total_notes += 1
                if msg.note in disabled_notes:
                    disabled_count += 1
                    disabled_breakdown[msg.note] += 1

    return {
        "total_notes": total_notes,
        "disabled_count": disabled_count,
        "disabled_pct": 100 * disabled_count / total_notes if total_notes > 0 else 0,
        "disabled_breakdown": dict(disabled_breakdown),
    }


def select_test_midis(n_samples: int = None) -> List[Path]:
    """Select a stratified sample of MIDI files for testing."""
    import random

    n_samples = n_samples or CONFIG["test_sample_size"]
    midi_dir = CONFIG["midi_dir"]

    # Get all MIDI files
    midi_files = list(midi_dir.glob("*.mid"))
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {midi_dir}")

    # Group by composer
    composer_files = defaultdict(list)
    for f in midi_files:
        composer = f.stem.split("_")[0]
        composer_files[composer].append(f)

    # Stratified sampling
    random.seed(CONFIG["random_seed"])
    selected = []
    per_composer = max(1, n_samples // len(composer_files))

    for composer, files in composer_files.items():
        sample_size = min(per_composer, len(files))
        selected.extend(random.sample(files, sample_size))

    # Trim to exact size
    if len(selected) > n_samples:
        selected = random.sample(selected, n_samples)

    return sorted(selected)


# =============================================================================
# STEP 2: PIANOTEQ RENDERING
# =============================================================================

def render_midi_with_pianoteq(
    midi_path: Path,
    output_path: Path,
    preset: str,
    sample_rate: int = None,
) -> bool:
    """Render a MIDI file with Pianoteq using specified preset."""
    sample_rate = sample_rate or CONFIG["sample_rate"]
    pianoteq = CONFIG["pianoteq_path"]

    if not Path(pianoteq).exists():
        raise FileNotFoundError(f"Pianoteq not found at {pianoteq}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        pianoteq,
        "--headless",
        "--preset", preset,
        "--midi", str(midi_path),
        "--wav", str(output_path),
        "--rate", str(sample_rate),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return output_path.exists()
    except subprocess.TimeoutExpired:
        print(f"  Timeout rendering {midi_path.name}")
        return False
    except Exception as e:
        print(f"  Error rendering {midi_path.name}: {e}")
        return False


def render_test_batch(midi_files: List[Path], presets: List[str] = None) -> Dict[str, List[Path]]:
    """Render all test MIDIs with all presets."""
    from tqdm import tqdm

    presets = presets or CONFIG["presets"]
    output_dir = CONFIG["output_dir"]

    rendered_files = {preset: [] for preset in presets}

    total_renders = len(midi_files) * len(presets)
    print(f"\nRendering {len(midi_files)} MIDIs x {len(presets)} presets = {total_renders} files")

    with tqdm(total=total_renders, desc="Rendering") as pbar:
        for preset in presets:
            # Create safe directory name
            preset_dir = output_dir / preset.replace(" ", "_").replace(".", "")
            preset_dir.mkdir(parents=True, exist_ok=True)

            for midi_path in midi_files:
                output_path = preset_dir / f"{midi_path.stem}.wav"

                if output_path.exists():
                    rendered_files[preset].append(output_path)
                else:
                    success = render_midi_with_pianoteq(midi_path, output_path, preset)
                    if success:
                        rendered_files[preset].append(output_path)

                pbar.update(1)

    return rendered_files


# =============================================================================
# STEP 3: MUQ EMBEDDING EXTRACTION
# =============================================================================

def extract_muq_embeddings_batch(
    audio_files: List[Path],
    output_dir: Path,
    layer_start: int = None,
    layer_end: int = None,
) -> Dict[str, Path]:
    """Extract MuQ embeddings for a batch of audio files."""
    import torch
    from tqdm import tqdm

    layer_start = layer_start or CONFIG["muq_layer_start"]
    layer_end = layer_end or CONFIG["muq_layer_end"]

    # Import MuQ extractor
    try:
        from audio_experiments.extractors import MuQExtractor
    except ImportError:
        print("MuQExtractor not found. Installing/importing MuQ...")
        raise ImportError("Please ensure audio_experiments package is available")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor with layer range and cache directory
    extractor = MuQExtractor(
        layer_start=layer_start,
        layer_end=layer_end,
        cache_dir=output_dir,
    )

    embeddings = {}
    for audio_path in tqdm(audio_files, desc="Extracting MuQ embeddings"):
        key = audio_path.stem
        output_path = output_dir / f"{key}.pt"

        if output_path.exists():
            embeddings[key] = output_path
            continue

        try:
            # Extract embeddings using the file-based method
            emb = extractor.extract_from_file(audio_path, use_cache=True)
            embeddings[key] = output_path

        except Exception as e:
            print(f"  Error extracting {audio_path.name}: {e}")

    return embeddings


# =============================================================================
# STEP 4: EMBEDDING COMPARISON AND VALIDATION
# =============================================================================

def compute_embedding_similarity(emb1_path: Path, emb2_path: Path) -> Dict:
    """Compute similarity metrics between two embeddings."""
    import torch
    import numpy as np
    from scipy.stats import pearsonr

    emb1 = torch.load(emb1_path, weights_only=True)
    emb2 = torch.load(emb2_path, weights_only=True)

    # Flatten to 1D for comparison (mean pooling over time)
    emb1_flat = emb1.mean(dim=0).numpy() if emb1.dim() > 1 else emb1.numpy()
    emb2_flat = emb2.mean(dim=0).numpy() if emb2.dim() > 1 else emb2.numpy()

    # Cosine similarity
    cos_sim = np.dot(emb1_flat, emb2_flat) / (np.linalg.norm(emb1_flat) * np.linalg.norm(emb2_flat))

    # Pearson correlation
    corr, _ = pearsonr(emb1_flat, emb2_flat)

    # L2 distance
    l2_dist = np.linalg.norm(emb1_flat - emb2_flat)

    return {
        "cosine_similarity": float(cos_sim),
        "pearson_correlation": float(corr),
        "l2_distance": float(l2_dist),
    }


def compare_preset_embeddings(
    preset_embeddings: Dict[str, Dict[str, Path]],
    reference_embeddings: Optional[Dict[str, Path]] = None,
) -> Dict:
    """Compare embeddings across presets and against reference."""
    import numpy as np
    from itertools import combinations

    results = {
        "preset_vs_preset": {},
        "preset_vs_reference": {},
        "summary": {},
    }

    presets = list(preset_embeddings.keys())

    # Compare presets against each other
    print("\nComparing presets against each other...")
    for p1, p2 in combinations(presets, 2):
        key = f"{p1} vs {p2}"
        similarities = []

        common_keys = set(preset_embeddings[p1].keys()) & set(preset_embeddings[p2].keys())
        for midi_key in common_keys:
            sim = compute_embedding_similarity(
                preset_embeddings[p1][midi_key],
                preset_embeddings[p2][midi_key],
            )
            similarities.append(sim["cosine_similarity"])

        results["preset_vs_preset"][key] = {
            "mean_cosine_similarity": float(np.mean(similarities)),
            "std_cosine_similarity": float(np.std(similarities)),
            "min_cosine_similarity": float(np.min(similarities)),
            "n_samples": len(similarities),
        }

    # Compare against reference if available
    if reference_embeddings:
        print("\nComparing presets against reference audio...")
        for preset, emb_dict in preset_embeddings.items():
            similarities = []
            common_keys = set(emb_dict.keys()) & set(reference_embeddings.keys())

            for midi_key in common_keys:
                sim = compute_embedding_similarity(
                    emb_dict[midi_key],
                    reference_embeddings[midi_key],
                )
                similarities.append(sim["cosine_similarity"])

            if similarities:
                results["preset_vs_reference"][preset] = {
                    "mean_cosine_similarity": float(np.mean(similarities)),
                    "std_cosine_similarity": float(np.std(similarities)),
                    "min_cosine_similarity": float(np.min(similarities)),
                    "n_samples": len(similarities),
                }

    # Summary statistics
    all_preset_sims = [v["mean_cosine_similarity"] for v in results["preset_vs_preset"].values()]
    results["summary"]["preset_diversity"] = {
        "mean_inter_preset_similarity": float(np.mean(all_preset_sims)),
        "std_inter_preset_similarity": float(np.std(all_preset_sims)),
        "interpretation": "Lower similarity = more diverse timbres (good for augmentation)",
    }

    if results["preset_vs_reference"]:
        ref_sims = [v["mean_cosine_similarity"] for v in results["preset_vs_reference"].values()]
        results["summary"]["reference_comparison"] = {
            "mean_similarity_to_reference": float(np.mean(ref_sims)),
            "min_similarity_to_reference": float(np.min(ref_sims)),
            "passes_threshold": float(np.min(ref_sims)) >= CONFIG["min_cosine_similarity"],
        }

    return results


# =============================================================================
# STEP 5: VALIDATION REPORT
# =============================================================================

def generate_validation_report(
    midi_analysis: Dict,
    comparison_results: Dict,
    output_path: Path,
) -> None:
    """Generate a comprehensive validation report."""

    report = {
        "config": {
            "test_sample_size": CONFIG["test_sample_size"],
            "presets_tested": CONFIG["presets"],
            "disabled_notes": list(CONFIG["disabled_notes"]),
            "validation_thresholds": {
                "min_cosine_similarity": CONFIG["min_cosine_similarity"],
                "min_dimension_correlation": CONFIG["min_dimension_correlation"],
            },
        },
        "midi_analysis": midi_analysis,
        "embedding_comparison": comparison_results,
        "verdict": {},
    }

    # Determine verdict
    passes_diversity = comparison_results["summary"]["preset_diversity"]["mean_inter_preset_similarity"] < 0.95

    if "reference_comparison" in comparison_results["summary"]:
        passes_reference = comparison_results["summary"]["reference_comparison"]["passes_threshold"]
        report["verdict"]["demo_mode_acceptable"] = passes_reference
        report["verdict"]["reason"] = (
            "Demo mode embeddings are sufficiently similar to reference audio"
            if passes_reference
            else "Demo mode embeddings differ too much from reference - consider purchasing Pianoteq"
        )
    else:
        report["verdict"]["demo_mode_acceptable"] = "UNKNOWN - no reference audio available"
        report["verdict"]["reason"] = "Cannot determine without reference audio comparison"

    report["verdict"]["preset_diversity_good"] = passes_diversity

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION REPORT SUMMARY")
    print("=" * 60)
    print(f"\nMIDI Analysis:")
    print(f"  Total files tested: {midi_analysis['total_files']}")
    print(f"  Average disabled notes: {midi_analysis['avg_disabled_pct']:.1f}%")

    print(f"\nPreset Diversity:")
    print(f"  Mean inter-preset similarity: {comparison_results['summary']['preset_diversity']['mean_inter_preset_similarity']:.3f}")
    print(f"  Diversity is {'GOOD' if passes_diversity else 'LOW'}")

    if "reference_comparison" in comparison_results["summary"]:
        ref = comparison_results["summary"]["reference_comparison"]
        print(f"\nReference Comparison:")
        print(f"  Mean similarity to reference: {ref['mean_similarity_to_reference']:.3f}")
        print(f"  Min similarity to reference: {ref['min_similarity_to_reference']:.3f}")
        print(f"  Passes threshold ({CONFIG['min_cosine_similarity']}): {'YES' if ref['passes_threshold'] else 'NO'}")

    print(f"\nVERDICT: {report['verdict']['demo_mode_acceptable']}")
    print(f"Reason: {report['verdict']['reason']}")
    print(f"\nFull report saved to: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution flow."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Pianoteq demo mode for MuQ training")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of MIDIs to test")
    parser.add_argument("--skip-render", action="store_true", help="Skip rendering (use existing files)")
    parser.add_argument("--skip-extract", action="store_true", help="Skip embedding extraction")
    parser.add_argument("--reference-dir", type=str, help="Path to reference audio embeddings")
    args = parser.parse_args()

    print("=" * 60)
    print("PIANOTEQ DEMO MODE VALIDATION")
    print("=" * 60)

    # Update config
    CONFIG["test_sample_size"] = args.n_samples
    if args.reference_dir:
        CONFIG["reference_audio_dir"] = Path(args.reference_dir)

    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Select and analyze MIDIs
    print("\n[STEP 1] Selecting and analyzing MIDI files...")
    midi_files = select_test_midis(args.n_samples)
    print(f"Selected {len(midi_files)} MIDI files")

    midi_analysis = {
        "total_files": len(midi_files),
        "per_file": {},
    }

    total_disabled_pct = 0
    for midi_path in midi_files:
        analysis = analyze_midi_for_disabled_notes(midi_path)
        midi_analysis["per_file"][midi_path.stem] = analysis
        total_disabled_pct += analysis["disabled_pct"]

    midi_analysis["avg_disabled_pct"] = total_disabled_pct / len(midi_files)
    print(f"Average disabled notes: {midi_analysis['avg_disabled_pct']:.1f}%")

    # Step 2: Render with Pianoteq
    if not args.skip_render:
        print("\n[STEP 2] Rendering with Pianoteq...")
        rendered_files = render_test_batch(midi_files, CONFIG["presets"])

        for preset, files in rendered_files.items():
            print(f"  {preset}: {len(files)} files")
    else:
        print("\n[STEP 2] Skipping render (using existing files)...")
        rendered_files = {}
        for preset in CONFIG["presets"]:
            preset_dir = output_dir / preset.replace(" ", "_").replace(".", "")
            if preset_dir.exists():
                rendered_files[preset] = list(preset_dir.glob("*.wav"))

    # Step 3: Extract MuQ embeddings
    if not args.skip_extract:
        print("\n[STEP 3] Extracting MuQ embeddings...")
        preset_embeddings = {}

        for preset, audio_files in rendered_files.items():
            if not audio_files:
                continue

            emb_dir = output_dir / "embeddings" / preset.replace(" ", "_").replace(".", "")
            embeddings = extract_muq_embeddings_batch(audio_files, emb_dir)
            preset_embeddings[preset] = embeddings
            print(f"  {preset}: {len(embeddings)} embeddings")
    else:
        print("\n[STEP 3] Skipping embedding extraction (using existing files)...")
        preset_embeddings = {}
        for preset in CONFIG["presets"]:
            emb_dir = output_dir / "embeddings" / preset.replace(" ", "_").replace(".", "")
            if emb_dir.exists():
                preset_embeddings[preset] = {
                    p.stem: p for p in emb_dir.glob("*.pt")
                }

    # Step 4: Load reference embeddings if available
    reference_embeddings = None
    ref_dir = CONFIG["reference_audio_dir"]
    if ref_dir.exists():
        print("\n[STEP 4] Loading reference embeddings...")
        # Check for pre-extracted embeddings or extract from audio
        ref_emb_dir = output_dir / "embeddings" / "reference"
        ref_audio_files = list(ref_dir.glob("*.wav"))

        if ref_audio_files:
            # Filter to only files that match our test set
            test_keys = {m.stem for m in midi_files}
            ref_audio_files = [f for f in ref_audio_files if f.stem in test_keys]

            if ref_audio_files:
                reference_embeddings = extract_muq_embeddings_batch(ref_audio_files, ref_emb_dir)
                print(f"  Reference: {len(reference_embeddings)} embeddings")
    else:
        print("\n[STEP 4] No reference audio directory found, skipping reference comparison")

    # Step 5: Compare embeddings
    print("\n[STEP 5] Comparing embeddings...")
    comparison_results = compare_preset_embeddings(preset_embeddings, reference_embeddings)

    # Step 6: Generate report
    print("\n[STEP 6] Generating validation report...")
    report_path = output_dir / "validation_report.json"
    generate_validation_report(midi_analysis, comparison_results, report_path)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
