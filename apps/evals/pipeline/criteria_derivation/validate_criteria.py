"""Validate clusters against effectiveness signals and select criteria.

Step 3 of criteria derivation: for each cluster, compute frequency,
repetition correlation (using embedding similarity), time investment
signal, and severity signal. Select criteria that meet frequency > 5%
AND at least one validity signal.

Usage:
    cd apps/evals/
    uv run python -m pipeline.criteria_derivation.validate_criteria
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"


def load_data() -> tuple[list[dict], dict]:
    """Load qualities (moments) and cluster assignments."""
    qualities_path = DATA_DIR / "qualities_raw.jsonl"
    moments = []
    with open(qualities_path) as f:
        for line in f:
            moments.append(json.loads(line))

    clusters_path = DATA_DIR / "clusters.json"
    with open(clusters_path) as f:
        clusters = json.load(f)

    return moments, clusters


def compute_repetition_signal(moments: list[dict]) -> dict[str, bool]:
    """Compute whether each moment's issue was repeated in the same video.

    Uses sentence-transformer cosine similarity (>= 0.75) on
    (musical_dimension, passage_description) pairs within the same video.

    Returns: {moment_id: is_repeated}
    """
    by_video: dict[str, list[dict]] = defaultdict(list)
    for m in moments:
        by_video[m["video_id"]].append(m)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    is_repeated: dict[str, bool] = {}

    for video_id, video_moments in by_video.items():
        if len(video_moments) < 2:
            for m in video_moments:
                is_repeated[m["moment_id"]] = False
            continue

        descriptors = [
            f"{m['musical_dimension']}: {m.get('passage_description', '')}"
            for m in video_moments
        ]
        embeddings = model.encode(descriptors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        similarity = (embeddings / norms) @ (embeddings / norms).T

        for i, m in enumerate(video_moments):
            repeated = any(
                similarity[i, j] >= 0.75
                for j in range(len(video_moments)) if i != j
            )
            is_repeated[m["moment_id"]] = repeated

    return is_repeated


def validate_clusters(
    moments: list[dict],
    clusters_data: dict,
    is_repeated: dict[str, bool],
) -> list[dict]:
    """Validate each cluster against effectiveness signals."""
    moment_by_id = {m["moment_id"]: m for m in moments}
    total_moments = len(moments)

    all_times = [m["time_spent_seconds"] for m in moments if m["time_spent_seconds"] > 0]
    overall_mean_time = float(np.mean(all_times)) if all_times else 30.0

    all_sev = [m["severity"] for m in moments]
    overall_significant_ratio = sum(1 for s in all_sev if s == "significant") / len(all_sev) if all_sev else 0

    overall_non_repeated = (
        sum(1 for v in is_repeated.values() if not v) / len(is_repeated)
        if is_repeated else 0.5
    )

    results = []
    for cluster in clusters_data["clusters"]:
        if cluster["cluster_id"] == -1:
            continue

        cluster_moment_ids = set(cluster["moment_ids"])
        cluster_moments = [moment_by_id[mid] for mid in cluster_moment_ids if mid in moment_by_id]

        if not cluster_moments:
            continue

        freq = len(cluster_moments) / total_moments

        # Time investment signal (primary)
        times = [m["time_spent_seconds"] for m in cluster_moments if m["time_spent_seconds"] > 0]
        mean_time = float(np.mean(times)) if times else 0.0
        time_ratio = mean_time / overall_mean_time if overall_mean_time > 0 else 1.0

        # Severity signal
        significant_count = sum(1 for m in cluster_moments if m["severity"] == "significant")
        cluster_significant_ratio = significant_count / len(cluster_moments)
        severity_lift = cluster_significant_ratio - overall_significant_ratio

        # Repetition signal (weak)
        cluster_repeated = [is_repeated.get(mid, False) for mid in cluster_moment_ids if mid in is_repeated]
        cluster_non_repeated_ratio = (
            sum(1 for r in cluster_repeated if not r) / len(cluster_repeated)
            if cluster_repeated else 0
        )
        repetition_lift = cluster_non_repeated_ratio - overall_non_repeated

        # Feedback type distribution
        type_counts: dict[str, int] = {}
        for m in cluster_moments:
            ft = m["feedback_type"]
            type_counts[ft] = type_counts.get(ft, 0) + 1

        # Selection
        passes_frequency = freq > 0.05
        has_time_signal = time_ratio > 1.2
        has_severity_signal = severity_lift > 0.05
        has_repetition_signal = repetition_lift > 0.05
        selected = passes_frequency and (has_time_signal or has_severity_signal or has_repetition_signal)

        results.append({
            "cluster_id": cluster["cluster_id"],
            "name": cluster.get("name", ""),
            "size": cluster["size"],
            "unique_moments": len(cluster_moments),
            "frequency": round(freq, 4),
            "mean_time_seconds": round(mean_time, 1),
            "time_ratio": round(time_ratio, 2),
            "has_time_signal": has_time_signal,
            "significant_ratio": round(cluster_significant_ratio, 3),
            "severity_lift": round(severity_lift, 3),
            "has_severity_signal": has_severity_signal,
            "non_repeated_ratio": round(cluster_non_repeated_ratio, 3),
            "repetition_lift": round(repetition_lift, 3),
            "has_repetition_signal": has_repetition_signal,
            "feedback_type_distribution": type_counts,
            "passes_frequency": passes_frequency,
            "selected": selected,
            "example_descriptors": cluster["example_descriptors"][:5],
        })

    return results


def main():
    print("Loading data...")
    moments, clusters_data = load_data()
    print(f"  {len(moments)} moments, {clusters_data['n_clusters']} clusters")

    print("Computing repetition signal (embedding similarity >= 0.75)...")
    is_repeated = compute_repetition_signal(moments)
    repeated_count = sum(1 for v in is_repeated.values() if v)
    print(f"  {repeated_count}/{len(is_repeated)} moments have repeated issues "
          f"({repeated_count / len(is_repeated) * 100:.1f}%)")

    print("Validating clusters...")
    results = validate_clusters(moments, clusters_data, is_repeated)

    selected = [r for r in results if r["selected"]]
    print(f"\n  {len(selected)} criteria selected (of {len(results)} clusters):")
    for r in sorted(selected, key=lambda x: x["frequency"], reverse=True):
        signals = []
        if r["has_time_signal"]:
            signals.append(f"time={r['time_ratio']:.1f}x")
        if r["has_severity_signal"]:
            signals.append(f"sev_lift=+{r['severity_lift']:.0%}")
        if r["has_repetition_signal"]:
            signals.append(f"rep_lift=+{r['repetition_lift']:.0%}")
        examples = ", ".join(r["example_descriptors"][:3])
        print(f"    Cluster {r['cluster_id']}: freq={r['frequency']:.1%}, "
              f"{', '.join(signals)} -- {examples}")

    report = {
        "total_moments": len(moments),
        "total_clusters": len(results),
        "selected_count": len(selected),
        "repeated_moment_rate": round(repeated_count / len(is_repeated), 3) if is_repeated else 0,
        "overall_mean_time_seconds": round(float(np.mean([
            m["time_spent_seconds"] for m in moments if m["time_spent_seconds"] > 0
        ])), 1) if moments else 0,
        "criteria": results,
    }

    output_path = DATA_DIR / "validation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved to {output_path}")
    print("\n  NEXT: Review selected clusters, name them, then write judge rubrics "
          "in apps/evals/shared/prompts/observation_quality_judge_v2.txt")


if __name__ == "__main__":
    main()
