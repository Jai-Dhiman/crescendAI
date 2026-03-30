"""Audio gate approach comparison harness.

Compares 4 classification approaches against a labeled dataset:
1. Spectral flatness (single feature)
2. Spectral flatness + centroid (two features)
3. Multi-feature set (flatness + centroid + kurtosis + ZCR + chroma)
4. YAMNet (521-class ML model)

Usage:
    cd apps/evals/audio_gate
    uv run python run_comparison.py
    uv run python run_comparison.py --skip-yamnet  # faster, skip YAMNet
    uv run python run_comparison.py --sample-rate 48000  # test at 48kHz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from approaches import spectral_flatness, spectral_two_feature, multi_feature

VALID_CLASSES = {"piano", "not_piano"}
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def validate_labels(labels: dict) -> dict[str, dict]:
    """Validate label schema. Fail fast on errors."""
    entries = {k: v for k, v in labels.items() if not k.startswith("_")}
    errors = []

    for filepath, meta in entries.items():
        if meta["class"] not in VALID_CLASSES:
            errors.append(f"Invalid class '{meta['class']}' for {filepath}")
        full_path = DATA_DIR / filepath
        if not full_path.exists():
            # Check if it's a symlink pointing to a valid file
            if full_path.is_symlink():
                target = full_path.resolve()
                if not target.exists():
                    errors.append(f"Broken symlink: {filepath} -> {target}")
            else:
                errors.append(f"Missing file: {filepath}")

    if errors:
        print("Label validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"Labels validated: {len(entries)} entries")
    piano_count = sum(1 for v in entries.values() if v["class"] == "piano")
    not_piano_count = len(entries) - piano_count
    print(f"  Piano: {piano_count}, Not piano: {not_piano_count}")
    return entries


def load_clip(filepath: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Load a single audio clip, taking first 15 seconds."""
    full_path = DATA_DIR / filepath
    audio, sr = librosa.load(str(full_path), sr=target_sr, mono=True, duration=15.0)
    return audio, sr


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    approach_name: str,
) -> dict:
    """Compute classification metrics with recall >= 0.95 constraint."""
    recall = recall_score(y_true, y_pred, pos_label="piano", zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label="piano", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="piano", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["piano", "not_piano"])

    meets_recall = recall >= 0.95

    return {
        "approach": approach_name,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "meets_recall_constraint": meets_recall,
        "confusion_matrix": {
            "tp": int(cm[0][0]),
            "fn": int(cm[0][1]),
            "fp": int(cm[1][0]),
            "tn": int(cm[1][1]),
        },
    }


def find_best_threshold(
    sweep_results: list[dict],
    y_true: list[str],
    approach_name: str,
) -> dict:
    """Find threshold that maximizes F1 while maintaining recall >= 0.95."""
    best = None
    best_f1 = -1.0

    for result in sweep_results:
        preds = result["predictions"]
        recall = recall_score(y_true, preds, pos_label="piano", zero_division=0)
        if recall < 0.95:
            continue
        f1 = f1_score(y_true, preds, pos_label="piano", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            params = {k: v for k, v in result.items() if k != "predictions"}
            best = {
                "params": params,
                "metrics": compute_metrics(y_true, preds, approach_name),
            }

    if best is None:
        # No threshold meets recall constraint -- find closest
        best_recall = -1.0
        for result in sweep_results:
            preds = result["predictions"]
            recall = recall_score(y_true, preds, pos_label="piano", zero_division=0)
            if recall > best_recall:
                best_recall = recall
                params = {k: v for k, v in result.items() if k != "predictions"}
                best = {
                    "params": params,
                    "metrics": compute_metrics(y_true, preds, approach_name),
                    "warning": f"No threshold meets recall >= 0.95. Best recall: {best_recall:.4f}",
                }

    return best


def run_comparison(target_sr: int = 24000, skip_yamnet: bool = False) -> dict:
    """Run the full comparison across all approaches."""
    # Load and validate labels
    labels_path = Path(__file__).parent / "labels.json"
    with open(labels_path) as f:
        raw_labels = json.load(f)
    entries = validate_labels(raw_labels)

    filepaths = list(entries.keys())
    y_true = [entries[fp]["class"] for fp in filepaths]
    subtypes = [entries[fp]["subtype"] for fp in filepaths]

    print(f"\nSample rate: {target_sr} Hz")
    print(f"Loading {len(filepaths)} clips...")

    # Load all audio clips
    clips = []
    for fp in filepaths:
        try:
            audio, sr = load_clip(fp, target_sr)
            clips.append((audio, sr))
        except Exception as e:
            print(f"  ERROR loading {fp}: {e}")
            sys.exit(1)
    print(f"All clips loaded.\n")

    results = {"sample_rate": target_sr, "clip_count": len(clips), "approaches": {}}

    # --- Approach 1: Spectral Flatness ---
    print("=" * 60)
    print("APPROACH 1: Spectral Flatness (single feature)")
    print("=" * 60)
    t0 = time.monotonic()
    features_1 = [spectral_flatness.compute_features(a, sr) for a, sr in clips]
    t1 = time.monotonic()
    latency_1 = (t1 - t0) / len(clips) * 1000

    sweep_1 = spectral_flatness.sweep_thresholds(features_1, y_true)
    best_1 = find_best_threshold(sweep_1, y_true, "spectral_flatness")
    best_1["latency_ms_per_clip"] = round(latency_1, 2)
    best_1["features_sample"] = features_1[:3]
    results["approaches"]["spectral_flatness"] = best_1
    _print_result(best_1)

    # --- Approach 2: Two-Feature Gate ---
    print("\n" + "=" * 60)
    print("APPROACH 2: Spectral Flatness + Centroid (two features)")
    print("=" * 60)
    t0 = time.monotonic()
    features_2 = [spectral_two_feature.compute_features(a, sr) for a, sr in clips]
    t1 = time.monotonic()
    latency_2 = (t1 - t0) / len(clips) * 1000

    sweep_2 = spectral_two_feature.sweep_thresholds(features_2)
    best_2 = find_best_threshold(sweep_2, y_true, "two_feature")
    best_2["latency_ms_per_clip"] = round(latency_2, 2)
    best_2["features_sample"] = features_2[:3]
    results["approaches"]["two_feature"] = best_2
    _print_result(best_2)

    # --- Approach 3: Multi-Feature ---
    print("\n" + "=" * 60)
    print("APPROACH 3: Multi-Feature (flatness + centroid + ZCR + chroma + more)")
    print("=" * 60)
    t0 = time.monotonic()
    features_3 = [multi_feature.compute_features(a, sr) for a, sr in clips]
    t1 = time.monotonic()
    latency_3 = (t1 - t0) / len(clips) * 1000

    # Use default thresholds for multi-feature
    preds_3 = [multi_feature.classify(f)[0] for f in features_3]
    metrics_3 = compute_metrics(y_true, preds_3, "multi_feature")
    results["approaches"]["multi_feature"] = {
        "metrics": metrics_3,
        "latency_ms_per_clip": round(latency_3, 2),
        "features_sample": features_3[:3],
    }
    _print_result(results["approaches"]["multi_feature"])

    # --- Approach 4: YAMNet ---
    if not skip_yamnet:
        print("\n" + "=" * 60)
        print("APPROACH 4: YAMNet (521-class ML model)")
        print("NOTE: Python latency is for relative comparison only.")
        print("      Browser TF.js performance will differ.")
        print("=" * 60)
        try:
            from approaches import yamnet_classifier

            t0 = time.monotonic()
            features_4 = [yamnet_classifier.compute_features(a, sr) for a, sr in clips]
            t1 = time.monotonic()
            latency_4 = (t1 - t0) / len(clips) * 1000

            sweep_4 = yamnet_classifier.sweep_thresholds(features_4)
            best_4 = find_best_threshold(sweep_4, y_true, "yamnet")
            best_4["latency_ms_per_clip"] = round(latency_4, 2)
            best_4["features_sample"] = features_4[:3]
            results["approaches"]["yamnet"] = best_4
            _print_result(best_4)
        except ImportError as e:
            print(f"  SKIPPED: {e}")
            results["approaches"]["yamnet"] = {"skipped": str(e)}
    else:
        print("\n  YAMNet SKIPPED (--skip-yamnet)")
        results["approaches"]["yamnet"] = {"skipped": "flag"}

    # --- Per-subtype breakdown ---
    print("\n" + "=" * 60)
    print("PER-SUBTYPE BREAKDOWN (best two-feature threshold)")
    print("=" * 60)
    if best_2 and "params" in best_2:
        params = best_2["params"].get("params", best_2["params"])
        ft = params.get("flatness", 0.12)
        cx = params.get("centroid_max", 2000.0)
        preds_2 = [
            spectral_two_feature.classify(f, ft, cx)[0] for f in features_2
        ]
        subtype_results = {}
        for fp, true, pred, st in zip(filepaths, y_true, preds_2, subtypes):
            if st not in subtype_results:
                subtype_results[st] = {"correct": 0, "total": 0, "files": []}
            subtype_results[st]["total"] += 1
            if true == pred:
                subtype_results[st]["correct"] += 1
            else:
                subtype_results[st]["files"].append(
                    {"file": fp, "true": true, "pred": pred}
                )

        for st, sr_data in sorted(subtype_results.items()):
            acc = sr_data["correct"] / sr_data["total"] if sr_data["total"] > 0 else 0
            status = "PASS" if acc == 1.0 else f"MISS ({sr_data['total'] - sr_data['correct']})"
            print(f"  {st:30s}  {sr_data['correct']}/{sr_data['total']}  {status}")
            for miss in sr_data["files"]:
                print(f"    -> {miss['file']} (true={miss['true']}, pred={miss['pred']})")

        results["subtype_breakdown"] = subtype_results

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Approach':<25} {'F1':>6} {'Recall':>8} {'Precision':>10} {'Recall>=0.95':>14} {'Latency':>10}")
    print("-" * 75)
    for name, data in results["approaches"].items():
        if "skipped" in data:
            print(f"  {name:<23} {'SKIPPED':>6}")
            continue
        m = data.get("metrics", {})
        lat = data.get("latency_ms_per_clip", "?")
        meets = "YES" if m.get("meets_recall_constraint") else "NO"
        print(
            f"  {name:<23} {m.get('f1', '?'):>6} {m.get('recall', '?'):>8} "
            f"{m.get('precision', '?'):>10} {meets:>14} {lat:>8} ms"
        )

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "comparison_summary.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


def _print_result(result: dict) -> None:
    """Print a single approach result."""
    m = result.get("metrics", {})
    lat = result.get("latency_ms_per_clip", "?")
    print(f"  F1:        {m.get('f1', '?')}")
    print(f"  Recall:    {m.get('recall', '?')}")
    print(f"  Precision: {m.get('precision', '?')}")
    print(f"  Meets recall >= 0.95: {m.get('meets_recall_constraint', '?')}")
    cm = m.get("confusion_matrix", {})
    if cm:
        print(f"  Confusion: TP={cm.get('tp')} FN={cm.get('fn')} FP={cm.get('fp')} TN={cm.get('tn')}")
    print(f"  Latency:   {lat} ms/clip")
    if "warning" in result:
        print(f"  WARNING: {result['warning']}")
    if "params" in result:
        print(f"  Best params: {result['params']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio gate approach comparison")
    parser.add_argument("--skip-yamnet", action="store_true", help="Skip YAMNet (faster)")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Target sample rate (default: 24000)")
    args = parser.parse_args()

    run_comparison(target_sr=args.sample_rate, skip_yamnet=args.skip_yamnet)
