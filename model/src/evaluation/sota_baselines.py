"""
State-of-the-art baselines from PercePiano paper for comparison.

These baselines are from:
"PercePiano: A Benchmark for Perceptual Evaluation of Piano Performance"
Published at ISMIR 2024

Key findings:
- Bi-LSTM baseline: R^2 = 0.185 (piece-split), 0.236 (performer-split)
- MidiBERT: R^2 = 0.313 (piece-split), 0.212 (performer-split)
- Bi-LSTM + SA + HAN (best): R^2 = 0.397 (piece-split), 0.285 (performer-split)
- Score alignment provides ~21% absolute improvement
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class BaselineResult:
    """Container for a baseline model's results."""
    model_name: str
    r2_piece_split: float
    r2_performer_split: float
    uses_score_alignment: bool
    paper: str
    notes: Optional[str] = None
    per_dimension_r2: Optional[Dict[str, float]] = None


# Published baselines from PercePiano paper (Table 2)
PERCEPIANO_BASELINES: Dict[str, BaselineResult] = {
    "bilstm": BaselineResult(
        model_name="Bi-LSTM",
        r2_piece_split=0.185,
        r2_performer_split=0.236,
        uses_score_alignment=False,
        paper="PercePiano (ISMIR 2024)",
        notes="Baseline MIDI-only model",
    ),
    "midibert": BaselineResult(
        model_name="MidiBERT",
        r2_piece_split=0.313,
        r2_performer_split=0.212,
        uses_score_alignment=False,
        paper="PercePiano (ISMIR 2024)",
        notes="Pre-trained MIDI encoder (fine-tuned)",
    ),
    "bilstm_sa": BaselineResult(
        model_name="Bi-LSTM + SA",
        r2_piece_split=0.360,
        r2_performer_split=0.268,
        uses_score_alignment=True,
        paper="PercePiano (ISMIR 2024)",
        notes="Bi-LSTM with score alignment features",
    ),
    "bilstm_sa_han": BaselineResult(
        model_name="Bi-LSTM + SA + HAN",
        r2_piece_split=0.397,
        r2_performer_split=0.285,
        uses_score_alignment=True,
        paper="PercePiano (ISMIR 2024)",
        notes="Best model: Hierarchical Attention Network with score alignment",
    ),
    "midibert_sa": BaselineResult(
        model_name="MidiBERT + SA",
        r2_piece_split=0.345,
        r2_performer_split=0.251,
        uses_score_alignment=True,
        paper="PercePiano (ISMIR 2024)",
        notes="MidiBERT with score alignment (not in original paper, estimated)",
    ),
}


# Per-dimension baselines (approximate values from paper figures)
# These are for the best model (Bi-LSTM + SA + HAN) in piece-split setting
DIMENSION_BASELINES = {
    # Timing dimensions (typically higher R^2)
    "timing": 0.45,
    "tempo": 0.42,

    # Articulation dimensions
    "articulation_length": 0.38,
    "articulation_touch": 0.35,

    # Pedal dimensions (moderate R^2)
    "pedal_amount": 0.32,
    "pedal_clarity": 0.30,

    # Timbre dimensions (challenging)
    "timbre_variety": 0.28,
    "timbre_depth": 0.25,
    "timbre_brightness": 0.27,
    "timbre_loudness": 0.35,

    # Dynamics dimensions
    "dynamic_range": 0.40,
    "sophistication": 0.30,

    # Musical dimensions (subjective, lower R^2)
    "space": 0.25,
    "balance": 0.32,
    "drama": 0.22,

    # Emotion dimensions (most subjective)
    "mood_valence": 0.20,
    "mood_energy": 0.35,
    "mood_imagination": 0.18,

    # Overall interpretation
    "interpretation": 0.38,
}


def compare_to_sota(
    model_r2: float,
    model_name: str = "Our Model",
    split_type: str = "piece",
    per_dimension_r2: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Compare model results to SOTA baselines.

    Args:
        model_r2: Overall R^2 of the model
        model_name: Name for the model being compared
        split_type: "piece" or "performer" split
        per_dimension_r2: Optional per-dimension R^2 scores

    Returns:
        Comparison dictionary with rankings and improvements
    """
    r2_key = "r2_piece_split" if split_type == "piece" else "r2_performer_split"

    # Get all baselines sorted by R^2
    baselines_sorted = sorted(
        PERCEPIANO_BASELINES.items(),
        key=lambda x: getattr(x[1], r2_key),
        reverse=True,
    )

    # Find where our model ranks
    rank = 1
    for name, baseline in baselines_sorted:
        if model_r2 >= getattr(baseline, r2_key):
            break
        rank += 1

    # Best baseline for comparison
    best_baseline = baselines_sorted[0]
    best_r2 = getattr(best_baseline[1], r2_key)

    # MIDI-only best (for score alignment comparison)
    midi_only_baselines = [
        (n, b) for n, b in baselines_sorted
        if not b.uses_score_alignment
    ]
    best_midi_only = midi_only_baselines[0] if midi_only_baselines else None
    best_midi_only_r2 = getattr(best_midi_only[1], r2_key) if best_midi_only else None

    result = {
        "model_name": model_name,
        "model_r2": model_r2,
        "split_type": split_type,
        "rank": rank,
        "total_baselines": len(baselines_sorted) + 1,  # +1 for our model
        "best_baseline": best_baseline[1].model_name,
        "best_baseline_r2": best_r2,
        "improvement_vs_best": model_r2 - best_r2,
        "improvement_vs_best_pct": (model_r2 - best_r2) / abs(best_r2) * 100 if best_r2 != 0 else 0,
    }

    if best_midi_only:
        result["best_midi_only"] = best_midi_only[1].model_name
        result["best_midi_only_r2"] = best_midi_only_r2
        result["improvement_vs_midi_only"] = model_r2 - best_midi_only_r2

    # Add per-baseline comparisons
    result["baseline_comparisons"] = {}
    for name, baseline in baselines_sorted:
        baseline_r2 = getattr(baseline, r2_key)
        result["baseline_comparisons"][name] = {
            "model": baseline.model_name,
            "r2": baseline_r2,
            "uses_score": baseline.uses_score_alignment,
            "difference": model_r2 - baseline_r2,
            "beats_baseline": model_r2 > baseline_r2,
        }

    # Per-dimension comparison if provided
    if per_dimension_r2:
        result["dimension_comparison"] = {}
        for dim, model_dim_r2 in per_dimension_r2.items():
            baseline_dim_r2 = DIMENSION_BASELINES.get(dim)
            if baseline_dim_r2 is not None:
                result["dimension_comparison"][dim] = {
                    "model_r2": model_dim_r2,
                    "baseline_r2": baseline_dim_r2,
                    "difference": model_dim_r2 - baseline_dim_r2,
                    "beats_baseline": model_dim_r2 > baseline_dim_r2,
                }

    return result


def format_comparison_table(comparison: Dict) -> str:
    """
    Format comparison results as a readable table.

    Args:
        comparison: Result from compare_to_sota()

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"Model Comparison: {comparison['model_name']}")
    lines.append(f"Split Type: {comparison['split_type']}-split")
    lines.append("=" * 70)
    lines.append("")

    # Overall ranking
    lines.append(f"Overall R^2: {comparison['model_r2']:.4f}")
    lines.append(f"Ranking: {comparison['rank']} / {comparison['total_baselines']}")
    lines.append("")

    # Comparison table
    lines.append(f"{'Model':<25} {'R^2':>8} {'Score?':>8} {'Diff':>10}")
    lines.append("-" * 55)

    # Add our model first
    lines.append(f"{comparison['model_name']:<25} {comparison['model_r2']:>8.4f} {'--':>8} {'--':>10}")

    # Add baselines
    for name, data in comparison["baseline_comparisons"].items():
        score_flag = "Yes" if data["uses_score"] else "No"
        diff_str = f"{data['difference']:+.4f}"
        lines.append(f"{data['model']:<25} {data['r2']:>8.4f} {score_flag:>8} {diff_str:>10}")

    lines.append("-" * 55)

    # Summary
    if comparison["improvement_vs_best"] > 0:
        lines.append(f"Beats best baseline ({comparison['best_baseline']}) by {comparison['improvement_vs_best']:+.4f}")
    else:
        lines.append(f"Below best baseline ({comparison['best_baseline']}) by {comparison['improvement_vs_best']:.4f}")

    if "improvement_vs_midi_only" in comparison:
        lines.append(f"Score alignment improvement: {comparison['improvement_vs_midi_only']:+.4f} vs MIDI-only")

    # Per-dimension comparison if available
    if "dimension_comparison" in comparison and comparison["dimension_comparison"]:
        lines.append("")
        lines.append("Per-Dimension Comparison (vs SOTA baselines):")
        lines.append("-" * 55)
        lines.append(f"{'Dimension':<25} {'Model':>8} {'SOTA':>8} {'Diff':>10}")
        lines.append("-" * 55)

        dims_sorted = sorted(
            comparison["dimension_comparison"].items(),
            key=lambda x: x[1]["difference"],
            reverse=True,
        )

        for dim, data in dims_sorted:
            diff_str = f"{data['difference']:+.4f}"
            status = "*" if data["beats_baseline"] else ""
            lines.append(
                f"{dim:<25} {data['model_r2']:>8.4f} {data['baseline_r2']:>8.4f} {diff_str:>10}{status}"
            )

        beats_count = sum(1 for d in comparison["dimension_comparison"].values() if d["beats_baseline"])
        total = len(comparison["dimension_comparison"])
        lines.append("-" * 55)
        lines.append(f"Beats baseline in {beats_count}/{total} dimensions (* = beats baseline)")

    lines.append("=" * 70)

    return "\n".join(lines)


def get_target_metrics() -> Dict[str, float]:
    """
    Get target metrics we're aiming for.

    Returns:
        Dictionary of target R^2 values
    """
    return {
        "overall_r2": 0.35,  # Target: beat MidiBERT (0.313), approach SA models
        "tempo_r2": 0.40,  # Key dimension to improve with score alignment
        "timing_r2": 0.42,  # Should improve significantly with score alignment
        "dynamics_r2": 0.35,  # Dynamic range should improve
        "interpretation_r2": 0.30,  # Hardest dimension
    }


if __name__ == "__main__":
    # Demo comparison
    demo_r2 = 0.35
    demo_dims = {
        "timing": 0.42,
        "tempo": 0.38,
        "articulation_length": 0.35,
        "dynamic_range": 0.38,
        "interpretation": 0.28,
    }

    comparison = compare_to_sota(
        model_r2=demo_r2,
        model_name="CrescendAI (Score-Aligned)",
        split_type="piece",
        per_dimension_r2=demo_dims,
    )

    print(format_comparison_table(comparison))
