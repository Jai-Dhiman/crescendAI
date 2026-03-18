"""Cluster quality descriptors to discover feedback quality criteria.

Step 2 of criteria derivation: embed all quality descriptors with
sentence-transformer, cluster with HDBSCAN, output cluster assignments
with per-descriptor labels and moment IDs for downstream validation.

Usage:
    cd apps/evals/
    uv run python -m pipeline.criteria_derivation.cluster_qualities
    uv run python -m pipeline.criteria_derivation.cluster_qualities --min-cluster-size 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hdbscan
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"


def load_qualities() -> tuple[list[str], list[dict]]:
    """Load quality descriptors and their source metadata."""
    qualities_path = DATA_DIR / "qualities_raw.jsonl"
    if not qualities_path.exists():
        raise FileNotFoundError(
            f"No qualities file at {qualities_path}. Run extract_qualities.py first."
        )

    all_descriptors: list[str] = []
    metadata: list[dict] = []

    with open(qualities_path) as f:
        for line in f:
            moment = json.loads(line)
            for quality in moment["qualities"]:
                all_descriptors.append(quality)
                metadata.append({
                    "moment_id": moment["moment_id"],
                    "video_id": moment["video_id"],
                    "feedback_type": moment["feedback_type"],
                    "severity": moment["severity"],
                    "time_spent_seconds": moment["time_spent_seconds"],
                    "stop_group": moment["stop_group"],
                    "musical_dimension": moment["musical_dimension"],
                    "passage_description": moment.get("passage_description", ""),
                })

    return all_descriptors, metadata


def embed_descriptors(
    descriptors: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Embed quality descriptors with sentence-transformer."""
    print(f"  Embedding {len(descriptors)} descriptors with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(descriptors, show_progress_bar=True, batch_size=256)
    return np.array(embeddings)


def run_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int = 20,
    min_samples: int = 5,
) -> np.ndarray:
    """Cluster embeddings with HDBSCAN. Returns cluster labels (-1 = noise)."""
    print(f"  Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points "
          f"({n_noise / len(labels) * 100:.1f}%)")
    return labels


def summarize_clusters(
    descriptors: list[str],
    metadata: list[dict],
    labels: np.ndarray,
) -> list[dict]:
    """Summarize each cluster with examples, size, and metadata distributions."""
    clusters = []

    for label in sorted(set(labels)):
        mask = labels == label
        desc_list = [d for d, m in zip(descriptors, mask) if m]
        cluster_meta = [m for m, msk in zip(metadata, mask) if msk]

        type_counts: dict[str, int] = {}
        for m in cluster_meta:
            type_counts[m["feedback_type"]] = type_counts.get(m["feedback_type"], 0) + 1

        severity_counts: dict[str, int] = {}
        for m in cluster_meta:
            severity_counts[m["severity"]] = severity_counts.get(m["severity"], 0) + 1

        times = [m["time_spent_seconds"] for m in cluster_meta if m["time_spent_seconds"] > 0]
        mean_time = float(np.mean(times)) if times else 0.0

        unique_moment_ids = list(set(m["moment_id"] for m in cluster_meta))

        clusters.append({
            "cluster_id": int(label),
            "size": int(mask.sum()),
            "unique_moments": len(unique_moment_ids),
            "moment_ids": unique_moment_ids,
            "example_descriptors": desc_list[:10],
            "feedback_type_distribution": type_counts,
            "severity_distribution": severity_counts,
            "mean_time_spent_seconds": round(mean_time, 1),
            "name": "",
        })

    return clusters


def main():
    parser = argparse.ArgumentParser(description="Cluster quality descriptors")
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    descriptors, metadata = load_qualities()
    print(f"Loaded {len(descriptors)} quality descriptors")

    embeddings = embed_descriptors(descriptors, model_name=args.model)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "embeddings.npy", embeddings)

    labels = run_clustering(embeddings, args.min_cluster_size, args.min_samples)
    clusters = summarize_clusters(descriptors, metadata, labels)

    output = {
        "n_descriptors": len(descriptors),
        "n_clusters": len([c for c in clusters if c["cluster_id"] != -1]),
        "n_noise": int((labels == -1).sum()),
        "params": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "embedding_model": args.model,
        },
        "clusters": clusters,
        "descriptor_labels": labels.tolist(),
        "descriptor_moment_ids": [m["moment_id"] for m in metadata],
    }

    output_path = DATA_DIR / "clusters.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    print("\nCluster summary:")
    for c in sorted(clusters, key=lambda x: x["size"], reverse=True):
        label_str = "NOISE" if c["cluster_id"] == -1 else f"Cluster {c['cluster_id']}"
        examples = ", ".join(c["example_descriptors"][:3])
        print(f"  {label_str:>12} ({c['size']:>4} desc, {c['unique_moments']:>4} moments): {examples}")


if __name__ == "__main__":
    main()
