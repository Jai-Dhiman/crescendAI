# Eval System Redesign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive empirical feedback quality criteria from masterclass data, then build two evals -- a skill-level eval (model quality) and a practice recording eval (pipeline quality) -- to determine whether the bottleneck is the model or the pipeline.

**Architecture:** Three sequential phases. Phase 1 uses LLM extraction + HDBSCAN clustering on 2,136 masterclass moments to derive 5-8 feedback quality criteria. Phase 2 builds two independent eval frameworks: Eval 1 tests model score correlation with skill level (63 YouTube recordings, local MPS inference), Eval 2 tests full pipeline feedback quality on practice recordings (15-25 YouTube practice sessions, local inference + wrangler dev + LLM judge with derived criteria). Phase 3 runs both and analyzes.

**Tech Stack:** Python (uv), sentence-transformers, hdbscan, anthropic SDK, yt-dlp, yaml, numpy, scipy. Local inference via MuQ + AMT on M4 MPS. Pipeline eval via wrangler dev (Rust Worker) + Groq + Anthropic APIs.

**Spec:** `docs/superpowers/specs/2026-03-17-eval-system-redesign.md`

**Parallelism note:** Tasks 5-6 (skill eval label curation + download + inference) can run in parallel with Tasks 1-3 (criteria derivation) and Task 7 Steps 1-4 (practice video collection). However, Task 7 Step 5 (practice inference) and Task 6 Steps 3-4 (skill eval inference) share the local MPS GPU and must be sequenced. Task 9 (practice eval runner) depends on Phase 1 output (judge v2 prompt).

---

## File Structure

### New Files

```
apps/evals/
  pipeline/
    criteria_derivation/
      __init__.py
      extract_qualities.py           # Step 1: LLM extraction from masterclass moments
      cluster_qualities.py           # Step 2: embed + HDBSCAN clustering
      validate_criteria.py           # Step 3: effectiveness validation + criteria selection
      data/
        .gitignore                   # track clusters.json + validation_report.json only
        qualities_raw.jsonl          # (gitignored) extracted quality descriptors
        embeddings.npy               # (gitignored) sentence-transformer embeddings
        clusters.json                # cluster assignments + labels + moment_ids
        validation_report.json       # frequency, repetition corr, severity corr

    practice_eval/
      __init__.py
      collect_practice.py            # YouTube search for practice videos
      eval_practice.py               # runs practice recordings through full pipeline
      scenarios/
        fur_elise.yaml               # scenario cards (human-annotated)
        nocturne_op9no2.yaml         # scenario cards (human-annotated)

  shared/
    prompts/
      observation_quality_judge_v2.txt  # derived criteria judge prompt
```

### Modified Files

```
apps/evals/shared/judge.py              # forward all context keys to prompt template
apps/evals/shared/pipeline_client.py    # (no changes needed -- piece_query already supported)
model/data/evals/skill_eval/*/manifest.yaml  # curated skill bucket labels
```

---

## Chunk 1: Phase 1 -- Criteria Derivation

### Task 1: Extract quality descriptors from masterclass moments

**Files:**
- Create: `apps/evals/pipeline/criteria_derivation/__init__.py`
- Create: `apps/evals/pipeline/criteria_derivation/extract_qualities.py`

- [ ] **Step 1: Create the extraction script**

Create `apps/evals/pipeline/criteria_derivation/__init__.py` (empty) and `extract_qualities.py`:

```python
"""Extract feedback quality descriptors from masterclass teaching moments.

Step 1 of criteria derivation: for each of the 2,136 masterclass moments,
ask an LLM to identify 2-5 qualities that make the intervention effective
or ineffective.

Usage:
    cd apps/evals/
    uv run python -m pipeline.criteria_derivation.extract_qualities
    uv run python -m pipeline.criteria_derivation.extract_qualities --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from paths import MODEL_DATA

import anthropic

MOMENTS_DIR = MODEL_DATA / "raw" / "masterclass" / "teaching_moments"
OUTPUT_DIR = Path(__file__).parent / "data"

EXTRACTION_PROMPT = """Given this piano masterclass teaching moment:

Teacher: {teacher}
Piece: {composer} - {piece}
What the teacher said: {feedback_summary}
Transcript excerpt: {transcript_excerpt}
Feedback type: {feedback_type}
Teacher demonstrated: {demonstrated}
Severity: {severity}

What qualities make this teaching intervention effective or ineffective?
List 2-5 specific qualities, each in 2-5 words.
Focus on qualities expressible in text (not physical demonstration).

Format: one quality per line, no numbering."""


def load_all_moments() -> list[dict]:
    """Load all teaching moments from JSONL files."""
    moments = []
    for jsonl_path in sorted(MOMENTS_DIR.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    moments.append(json.loads(line))
    return moments


def build_prompt(moment: dict) -> str:
    """Build the extraction prompt for a single moment."""
    transcript = moment.get("transcript_text", "")
    if len(transcript) > 500:
        transcript = transcript[:500] + "..."

    return EXTRACTION_PROMPT.format(
        teacher=moment.get("teacher", "Unknown"),
        composer=moment.get("composer", "Unknown"),
        piece=moment.get("piece", "Unknown"),
        feedback_summary=moment.get("feedback_summary", ""),
        transcript_excerpt=transcript,
        feedback_type=moment.get("feedback_type", "Unknown"),
        demonstrated=moment.get("demonstrated", False),
        severity=moment.get("severity", "Unknown"),
    )


def load_existing_results() -> dict[str, dict]:
    """Load already-processed moment IDs for resume support."""
    output_path = OUTPUT_DIR / "qualities_raw.jsonl"
    if not output_path.exists():
        return {}
    existing = {}
    with open(output_path) as f:
        for line in f:
            r = json.loads(line)
            existing[r["moment_id"]] = r
    return existing


def extract_qualities(
    moments: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    limit: int | None = None,
) -> list[dict]:
    """Extract quality descriptors for each moment via LLM."""
    client = anthropic.Anthropic()
    existing = load_existing_results()
    results = list(existing.values())
    processed_ids = set(existing.keys())

    if limit:
        moments = moments[:limit]

    for i, moment in enumerate(moments):
        moment_id = moment.get("moment_id", f"moment_{i}")
        if moment_id in processed_ids:
            continue

        prompt = build_prompt(moment)
        qualities: list[str] = []

        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                qualities = [
                    q.strip() for q in text.split("\n")
                    if q.strip() and len(q.strip()) > 3
                ]
                break
            except anthropic.RateLimitError:
                time.sleep(2 ** (attempt + 1))
            except anthropic.APIStatusError as e:
                if e.status_code == 529 and attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    raise

        result = {
            "moment_id": moment_id,
            "video_id": moment.get("video_id", ""),
            "teacher": moment.get("teacher", ""),
            "feedback_type": moment.get("feedback_type", ""),
            "severity": moment.get("severity", ""),
            "time_spent_seconds": moment.get("time_spent_seconds", 0),
            "stop_group": moment.get("stop_group", 0),
            "musical_dimension": moment.get("musical_dimension", ""),
            "passage_description": moment.get("passage_description", ""),
            "demonstrated": moment.get("demonstrated", False),
            "qualities": qualities,
        }
        results.append(result)

        # Save incrementally every 50 moments
        if (len(results) - len(existing)) % 50 == 0 or i == len(moments) - 1:
            save_results(results)
            print(f"  [{i+1}/{len(moments)}] {sum(len(r['qualities']) for r in results)} total qualities")

    return results


def save_results(results: list[dict]) -> Path:
    """Save extraction results to JSONL."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "qualities_raw.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract feedback quality descriptors")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N moments")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    print("Loading masterclass moments...")
    moments = load_all_moments()
    print(f"  Loaded {len(moments)} moments from {len(list(MOMENTS_DIR.glob('*.jsonl')))} videos")

    print("Extracting quality descriptors...")
    results = extract_qualities(moments, model=args.model, limit=args.limit)

    total_qualities = sum(len(r["qualities"]) for r in results)
    print(f"\n  Total: {total_qualities} qualities from {len(results)} moments "
          f"({total_qualities / len(results):.1f} avg per moment)")

    save_results(results)
    print(f"  Saved to {OUTPUT_DIR / 'qualities_raw.jsonl'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test with --limit 5**

Run: `cd apps/evals && uv run python -m pipeline.criteria_derivation.extract_qualities --limit 5`
Expected: 5 moments processed, ~15-25 quality descriptors, saved to `pipeline/criteria_derivation/data/qualities_raw.jsonl`

- [ ] **Step 3: Verify output format**

Read `apps/evals/pipeline/criteria_derivation/data/qualities_raw.jsonl`. Verify each line has `moment_id`, `qualities` (list of strings), and metadata fields (`severity`, `time_spent_seconds`, `stop_group`, etc.).

- [ ] **Step 4: HUMAN STEP -- Run full extraction (~15-30 min with Haiku)**

Run: `cd apps/evals && uv run python -m pipeline.criteria_derivation.extract_qualities`
Expected: ~2,136 moments processed, ~6,000-10,000 quality descriptors. Saves incrementally (resume-safe).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/pipeline/criteria_derivation/__init__.py
git add apps/evals/pipeline/criteria_derivation/extract_qualities.py
git commit -m "feat(evals): add quality descriptor extraction from masterclass moments"
```

---

### Task 2: Embed, cluster, and validate quality descriptors

**Files:**
- Create: `apps/evals/pipeline/criteria_derivation/data/.gitignore`
- Create: `apps/evals/pipeline/criteria_derivation/cluster_qualities.py`
- Create: `apps/evals/pipeline/criteria_derivation/validate_criteria.py`

- [ ] **Step 1: Add dependencies**

Run: `cd apps/evals && uv add sentence-transformers hdbscan`

- [ ] **Step 2: Create .gitignore for data directory**

Create `apps/evals/pipeline/criteria_derivation/data/.gitignore`:
```
# Large intermediate files
qualities_raw.jsonl
embeddings.npy

# Track final outputs
!clusters.json
!validation_report.json
```

- [ ] **Step 3: Create the clustering script**

Create `apps/evals/pipeline/criteria_derivation/cluster_qualities.py`:

```python
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
            "name": "",  # filled in during manual review
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
```

- [ ] **Step 4: Run clustering**

Run: `cd apps/evals && uv run python -m pipeline.criteria_derivation.cluster_qualities`
Expected: 10-20 clusters found. Review output for cluster coherence.

- [ ] **Step 5: Create the validation script**

Create `apps/evals/pipeline/criteria_derivation/validate_criteria.py`:

```python
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

    # Compute overall averages for comparison
    all_times = [m["time_spent_seconds"] for m in moments if m["time_spent_seconds"] > 0]
    overall_mean_time = float(np.mean(all_times)) if all_times else 30.0

    all_sev = [m["severity"] for m in moments]
    overall_significant_ratio = sum(1 for s in all_sev if s == "significant") / len(all_sev)

    results = []
    for cluster in clusters_data["clusters"]:
        if cluster["cluster_id"] == -1:
            continue

        # Get moments in this cluster
        cluster_moment_ids = set(cluster["moment_ids"])
        cluster_moments = [moment_by_id[mid] for mid in cluster_moment_ids if mid in moment_by_id]

        freq = len(cluster_moments) / total_moments

        # Time investment signal (primary): mean time vs overall mean
        times = [m["time_spent_seconds"] for m in cluster_moments if m["time_spent_seconds"] > 0]
        mean_time = float(np.mean(times)) if times else 0.0
        time_ratio = mean_time / overall_mean_time if overall_mean_time > 0 else 1.0

        # Severity signal: ratio of significant in this cluster vs overall
        significant_count = sum(1 for m in cluster_moments if m["severity"] == "significant")
        cluster_significant_ratio = significant_count / len(cluster_moments) if cluster_moments else 0
        severity_lift = cluster_significant_ratio - overall_significant_ratio

        # Repetition signal (weak): ratio of non-repeated in cluster vs overall
        cluster_repeated = [is_repeated.get(mid, False) for mid in cluster_moment_ids if mid in is_repeated]
        cluster_non_repeated_ratio = (
            sum(1 for r in cluster_repeated if not r) / len(cluster_repeated)
            if cluster_repeated else 0
        )
        overall_non_repeated = sum(1 for v in is_repeated.values() if not v) / len(is_repeated)
        repetition_lift = cluster_non_repeated_ratio - overall_non_repeated

        # Feedback type distribution
        type_counts: dict[str, int] = {}
        for m in cluster_moments:
            ft = m["feedback_type"]
            type_counts[ft] = type_counts.get(ft, 0) + 1

        # Selection: frequency > 5% AND at least one signal
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
        "repeated_moment_rate": round(repeated_count / len(is_repeated), 3),
        "overall_mean_time_seconds": round(float(np.mean([
            m["time_spent_seconds"] for m in moments if m["time_spent_seconds"] > 0
        ])), 1),
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
```

- [ ] **Step 6: Run validation**

Run: `cd apps/evals && uv run python -m pipeline.criteria_derivation.validate_criteria`
Expected: 5-8 clusters selected with frequency, time, severity, and repetition signals printed.

- [ ] **Step 7: Commit**

```bash
git add apps/evals/pipeline/criteria_derivation/
git commit -m "feat(evals): add criteria derivation pipeline (extract, cluster, validate)"
```

---

### Task 3: HUMAN STEP -- Name clusters and write judge prompt v2

**Files:**
- Create: `apps/evals/shared/prompts/observation_quality_judge_v2.txt`

- [ ] **Step 1: Review validation report**

Read `apps/evals/pipeline/criteria_derivation/data/validation_report.json`. For each selected cluster:
1. Read the example descriptors
2. Assign a criterion name (e.g., "Passage Specificity", "Mechanistic Clarity", "Musical Grounding")
3. Update the `name` field in `clusters.json`

- [ ] **Step 2: Write judge prompt v2**

Create `apps/evals/shared/prompts/observation_quality_judge_v2.txt` following the same format as v1. For each derived criterion:
- Definition (1-2 sentences)
- Pass/fail boundary
- Use `{piece_name}`, `{bar_range}`, `{predictions}`, `{baselines}`, `{recent_observations}`, `{analysis_facts}`, `{observation_text}` as template variables (these match the existing `judge_observation` format kwargs)

Additional context fields (`teaching_moment`, `subagent_output`, `scenario_notes`) are available but must be added as format kwargs. See Task 8.

- [ ] **Step 3: Commit**

```bash
git add apps/evals/shared/prompts/observation_quality_judge_v2.txt
git add apps/evals/pipeline/criteria_derivation/data/clusters.json
git commit -m "feat(evals): add judge prompt v2 with empirically derived criteria"
```

---

## Chunk 2: Phase 2A -- Skill Level Eval

### Task 4: HUMAN STEP -- Curate skill level labels

**Files:**
- Modify: `model/data/evals/skill_eval/fur_elise/manifest.yaml`
- Modify: `model/data/evals/skill_eval/nocturne_op9no2/manifest.yaml`

- [ ] **Step 1: Review and fix Fur Elise manifest**

Open `model/data/evals/skill_eval/fur_elise/manifest.yaml`. For each recording:
- Listen to a 30s sample (`https://youtube.com/watch?v={video_id}`)
- Verify or update `skill_bucket` (1-5)
- Add `label_source: manual` field
- Add `exclude: true` + `exclude_reason: "..."` for non-piece, tutorial, or unusable audio

Known misclassifications to fix:
- Rousseau (`wfF0zHeU3Zs`): bucket 3 -> 5
- Paul Barton (`wKvKiN1wYHw`): bucket 3 -> 5

- [ ] **Step 2: Review and fix Nocturne manifest**

Same process. Known misclassifications:
- Rousseau (`p29JUpsOSTE`): 3 -> 5
- Valentina Lisitsa (`tV5U8kVYS88`): 3 -> 5
- Elisabeth Leonskaja (`-7-iLKAWC0s`): 3 -> 5
- Tiffany Poon (`yDSxPiFOrEY`): 3 -> 4/5
- Yundi Li (`XMv53orNKnc`): 3 -> 5
- Dmitry Shishkin (`JVBzE0mUlSs`): 3 -> 5
- Vadim Chaimovich (`-PaYq5Pt228`): 3 -> 4/5

- [ ] **Step 3: Commit**

```bash
git add model/data/evals/skill_eval/
git commit -m "data: curate skill eval labels with manual verification"
```

---

### Task 5: HUMAN STEP -- Download audio and run skill eval inference

**Files:** No code changes -- uses existing `collect.py` and `run_inference.py`

- [ ] **Step 1: Download Fur Elise audio (~30-60 min)**

Run: `cd apps/evals && uv run python -m model.skill_eval.collect --piece fur_elise`

- [ ] **Step 2: Download Nocturne audio (~30-60 min)**

Run: `cd apps/evals && uv run python -m model.skill_eval.collect --piece nocturne_op9no2`

- [ ] **Step 3: Run inference on Fur Elise (~3-5 hours on M4 MPS)**

Run: `cd apps/evals && CRESCEND_DEVICE=mps uv run python -m model.skill_eval.run_inference --config ensemble_4fold --piece fur_elise`

Results cached in `model/data/evals/skill_eval/ensemble_4fold/fur_elise/results.json`. Run overnight.

- [ ] **Step 4: Run inference on Nocturne (~3-5 hours on M4 MPS)**

Run: `cd apps/evals && CRESCEND_DEVICE=mps uv run python -m model.skill_eval.run_inference --config ensemble_4fold --piece nocturne_op9no2`

- [ ] **Step 5: Run analysis**

Run: `cd apps/evals && uv run python -m model.skill_eval.analyze --piece all`
Expected: Spearman rho, Cohen's d, confusion rate printed. Plots saved to `model/data/evals/skill_eval/figures/`.

- [ ] **Step 6: Commit results**

```bash
git add model/data/evals/skill_eval/ensemble_4fold/
git add model/data/evals/skill_eval/figures/
git commit -m "data: skill eval results (ensemble_4fold, both pieces)"
```

---

## Chunk 3: Phase 2B -- Practice Recording Eval

### Task 6: Collect practice recordings

**Files:**
- Create: `apps/evals/pipeline/practice_eval/__init__.py`
- Create: `apps/evals/pipeline/practice_eval/collect_practice.py`

- [ ] **Step 1: Create the practice video collection script**

Create `apps/evals/pipeline/practice_eval/__init__.py` (empty) and `collect_practice.py`:

```python
"""Collect YouTube practice recordings for pipeline eval.

Searches for students practicing Fur Elise and Nocturne Op. 9 No. 2,
filters for actual practice sessions, outputs YAML scenario cards
for human review.

Usage:
    cd apps/evals/
    uv run python -m pipeline.practice_eval.collect_practice --piece fur_elise --search-only
    uv run python -m pipeline.practice_eval.collect_practice --piece nocturne_op9no2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
AUDIO_DIR = Path(__file__).parent / "audio"

PIECES = {
    "fur_elise": {
        "title": "Fur Elise",
        "composer": "Beethoven",
        "piece_query": "beethoven fur elise",
        "duration_range": (60, 360),
        "searches": [
            "fur elise practice session piano",
            "fur elise learning piano slow",
            "fur elise beginner practicing",
            "fur elise piano progress practice",
            "fur elise hands separate practice",
        ],
    },
    "nocturne_op9no2": {
        "title": "Chopin Nocturne Op. 9 No. 2",
        "composer": "Chopin",
        "piece_query": "chopin nocturne op 9 no 2",
        "duration_range": (60, 420),
        "searches": [
            "chopin nocturne op 9 no 2 practice piano",
            "chopin nocturne practicing slow",
            "chopin nocturne piano progress learning",
            "chopin nocturne beginner practice session",
        ],
    },
}

SKIP_KEYWORDS = [
    "tutorial", "synthesia", "sheet music", "how to play",
    "easy piano", "piano lesson", "slow tutorial", "learn to play",
    "midi", "piano tiles", "roblox",
]


def search_youtube(query: str, max_results: int = 15) -> list[dict]:
    """Search YouTube via yt-dlp."""
    cmd = [
        "yt-dlp", f"ytsearch{max_results}:{query}",
        "--dump-json", "--flat-playlist", "--no-download",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return []
    entries = []
    for line in result.stdout.strip().split("\n"):
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def collect_candidates(piece_id: str) -> list[dict]:
    """Search YouTube and collect practice video candidates."""
    piece = PIECES[piece_id]
    min_dur, max_dur = piece["duration_range"]
    seen: set[str] = set()
    candidates = []

    for query in piece["searches"]:
        print(f"  Searching: {query}")
        for entry in search_youtube(query):
            vid = entry.get("id", "")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            title = entry.get("title", "")
            dur = entry.get("duration", 0)
            if dur and (dur < min_dur or dur > max_dur):
                continue
            if any(kw in title.lower() for kw in SKIP_KEYWORDS):
                continue
            candidates.append({
                "video_id": vid,
                "title": title,
                "channel": entry.get("channel", entry.get("uploader", "")),
                "duration_seconds": dur or 0,
                "url": f"https://youtube.com/watch?v={vid}",
                "include": False,
                "skill_level": 0,
                "general_notes": "",
                "audio_quality": "",
                "expected_stop": True,
            })
        time.sleep(2)

    print(f"  Found {len(candidates)} candidates")
    return candidates


def save_candidates(piece_id: str, candidates: list[dict]) -> Path:
    """Save candidates to YAML for human review."""
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    path = SCENARIOS_DIR / f"{piece_id}.yaml"
    data = {
        "piece": piece_id,
        "title": PIECES[piece_id]["title"],
        "composer": PIECES[piece_id]["composer"],
        "piece_query": PIECES[piece_id]["piece_query"],
        "candidates": candidates,
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  Saved to {path}")
    print("  NEXT: Review YAML, set include=true on practice videos, fill skill_level/notes")
    return path


def download_included(piece_id: str):
    """Download audio for included recordings."""
    path = SCENARIOS_DIR / f"{piece_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    included = [c for c in data["candidates"] if c.get("include")]
    print(f"  Downloading {len(included)} included recordings...")
    for i, rec in enumerate(included):
        vid = rec["video_id"]
        out = AUDIO_DIR / f"{vid}.wav"
        if out.exists():
            print(f"  [{i+1}/{len(included)}] {vid} -- exists")
            continue
        print(f"  [{i+1}/{len(included)}] {vid} -- downloading...")
        cmd = [
            "yt-dlp", f"https://youtube.com/watch?v={vid}",
            "-x", "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 24000 -ac 1",
            "-o", str(out), "--no-playlist", "--quiet",
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"    Timeout downloading {vid}")
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--piece", required=True, choices=list(PIECES.keys()))
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()

    print(f"=== Practice: {PIECES[args.piece]['title']} ===")
    scenario_path = SCENARIOS_DIR / f"{args.piece}.yaml"
    if scenario_path.exists() and not args.search_only:
        download_included(args.piece)
    else:
        save_candidates(args.piece, collect_candidates(args.piece))
        if not args.search_only:
            print("\n  Review YAML first, then re-run without --search-only.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Search for practice candidates**

Run:
```bash
cd apps/evals
uv run python -m pipeline.practice_eval.collect_practice --piece fur_elise --search-only
uv run python -m pipeline.practice_eval.collect_practice --piece nocturne_op9no2 --search-only
```

- [ ] **Step 3: HUMAN STEP -- Review candidates and annotate**

For each candidate in the YAML files: watch 30-60s on YouTube, set `include: true` for practice sessions with audible mistakes, fill `skill_level` (1-3) and `general_notes`. Target 8-12 per piece.

- [ ] **Step 4: Download included recordings**

Run:
```bash
cd apps/evals
uv run python -m pipeline.practice_eval.collect_practice --piece fur_elise
uv run python -m pipeline.practice_eval.collect_practice --piece nocturne_op9no2
```

- [ ] **Step 5: HUMAN STEP -- Run local inference on practice recordings (~2-3 hours on MPS)**

Run: `cd apps/evals && CRESCEND_DEVICE=mps uv run python -m inference.eval_runner --audio-dir pipeline/practice_eval/audio`

Note: The `eval_runner.py` caches results to `model/data/eval/inference_cache/` (singular `eval`, not `evals`). The practice eval runner must look in this exact path.

- [ ] **Step 6: Commit**

```bash
git add apps/evals/pipeline/practice_eval/
git commit -m "feat(evals): add practice recording collection + scenarios"
```

---

### Task 7: Extend judge to forward all context keys

**Files:**
- Modify: `apps/evals/shared/judge.py`

The v2 judge prompt may use additional template variables (e.g., `{teaching_moment}`, `{subagent_output}`, `{scenario_notes}`). The current `judge_observation` hard-codes the format kwargs. Extend it to forward all context keys.

- [ ] **Step 1: Modify judge_observation to pass all context keys**

In `apps/evals/shared/judge.py`, change `judge_observation` (lines 56-87):

Replace the hard-coded `template.format(...)` call with:

```python
def judge_observation(
    observation_text: str,
    context: dict[str, Any],
    prompt_file: str = "observation_quality_judge_v1.txt",
    model: str = DEFAULT_MODEL,
) -> JudgeResult:
    """Judge a teacher observation using the specified prompt and model."""
    template = load_prompt(prompt_file)

    # Build format kwargs from context, with defaults for standard fields
    format_kwargs = {
        "piece_name": context.get("piece_name", "Unknown"),
        "bar_range": context.get("bar_range", "Unknown"),
        "predictions": context.get("predictions", "{}"),
        "baselines": context.get("baselines", "{}"),
        "recent_observations": context.get("recent_observations", "[]"),
        "analysis_facts": context.get("analysis_facts", "None"),
        "observation_text": observation_text,
    }
    # Forward any additional context keys (for v2 prompts)
    for key, value in context.items():
        if key not in format_kwargs:
            format_kwargs[key] = value

    user_message = template.format(**format_kwargs)
    # ... rest unchanged
```

- [ ] **Step 2: Verify v1 prompt still works**

Run: `cd apps/evals && uv run python -c "from shared.judge import load_prompt; print(load_prompt('observation_quality_judge_v1.txt')[:100])"`
Expected: v1 prompt loads without error.

- [ ] **Step 3: Commit**

```bash
git add apps/evals/shared/judge.py
git commit -m "feat(evals): forward all context keys to judge prompt template"
```

---

### Task 8: Build practice eval runner

**Files:**
- Create: `apps/evals/pipeline/practice_eval/eval_practice.py`

- [ ] **Step 1: Create the practice eval runner**

```python
"""Practice recording evaluation.

Runs practice recordings through the full pipeline (wrangler dev),
judges each observation with derived criteria (v2 judge), produces
a segmented report by tier, framing, and skill level.

Requires wrangler dev running at localhost:8787.

Usage:
    cd apps/evals/
    uv run python -m pipeline.practice_eval.eval_practice
    uv run python -m pipeline.practice_eval.eval_practice --piece fur_elise
"""

from __future__ import annotations

import asyncio
import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from paths import MODEL_DATA
from shared.judge import judge_observation
from shared.pipeline_client import run_recording, SessionResult
from shared.reporting import EvalReport, MetricResult

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
REPORTS_DIR = Path(__file__).parents[2] / "reports"

# eval_runner.py caches to model/data/eval/inference_cache/ (singular "eval")
INFERENCE_CACHE_BASE = MODEL_DATA / "eval" / "inference_cache"

JUDGE_PROMPT = "observation_quality_judge_v2.txt"


def load_scenarios(piece_id: str) -> list[dict]:
    """Load included scenarios for a piece."""
    path = SCENARIOS_DIR / f"{piece_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No scenarios at {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    included = [c for c in data.get("candidates", []) if c.get("include")]
    for c in included:
        c["piece_query"] = data.get("piece_query", "")
    return included


def find_inference_cache() -> dict[str, dict]:
    """Find and load the most recent inference cache."""
    if not INFERENCE_CACHE_BASE.exists():
        raise FileNotFoundError(
            f"No inference cache at {INFERENCE_CACHE_BASE}. Run eval_runner.py first."
        )
    cache_dirs = sorted(INFERENCE_CACHE_BASE.iterdir())
    if not cache_dirs:
        raise FileNotFoundError("Inference cache directory is empty.")

    recordings: dict[str, dict] = {}
    for cache_dir in cache_dirs:
        for json_file in cache_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
            rec_id = data.get("recording_id", json_file.stem)
            recordings[rec_id] = data
    return recordings


def main():
    parser = argparse.ArgumentParser(description="Run practice recording eval")
    parser.add_argument("--wrangler-url", default="http://localhost:8787")
    parser.add_argument("--piece", default="all", choices=["fur_elise", "nocturne_op9no2", "all"])
    args = parser.parse_args()

    pieces = ["fur_elise", "nocturne_op9no2"] if args.piece == "all" else [args.piece]

    all_scores: dict[str, list[bool]] = {}
    all_observations: list[dict] = []
    total_judge_calls = 0
    total_recordings = 0
    recordings_with_obs = 0
    tier_counts: dict[str, int] = {}
    framing_counts: dict[str, int] = {}

    print("Loading inference cache...")
    cache = find_inference_cache()
    print(f"  {len(cache)} recordings in cache")

    for piece_id in pieces:
        scenarios = load_scenarios(piece_id)
        print(f"\n=== {piece_id}: {len(scenarios)} practice recordings ===")

        for i, scenario in enumerate(scenarios):
            video_id = scenario["video_id"]
            total_recordings += 1
            print(f"  [{i+1}/{len(scenarios)}] {video_id}...", end=" ", flush=True)

            if video_id not in cache:
                print("not in cache, skipping")
                continue

            recording = cache[video_id]
            result: SessionResult = asyncio.run(
                run_recording(
                    args.wrangler_url,
                    recording,
                    piece_query=scenario.get("piece_query"),
                )
            )

            if result.errors:
                print(f"ERRORS: {result.errors}")
                continue

            if not result.observations:
                print("no observations (STOP did not trigger)")
                continue

            recordings_with_obs += 1

            for obs in result.observations:
                eval_ctx = obs.raw_message.get("eval_context", {})
                tier = str(eval_ctx.get("tier", "3"))
                framing = obs.framing or "unknown"
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                framing_counts[framing] = framing_counts.get(framing, 0) + 1

                context = {
                    "predictions": eval_ctx.get("predictions", {}),
                    "baselines": eval_ctx.get("baselines", {}),
                    "recent_observations": eval_ctx.get("recent_observations", []),
                    "analysis_facts": eval_ctx.get("analysis_facts", {}),
                    "piece_name": eval_ctx.get("piece_name", scenario.get("title", video_id)),
                    "bar_range": eval_ctx.get("bar_range", "full recording"),
                    "teaching_moment": json.dumps(eval_ctx.get("teaching_moment", {})),
                    "subagent_output": json.dumps(eval_ctx.get("subagent_output", {})),
                    "scenario_notes": scenario.get("general_notes", ""),
                }

                judge_result = judge_observation(obs.text, context, prompt_file=JUDGE_PROMPT)
                total_judge_calls += 1

                skill_level = scenario.get("skill_level", 0)
                for score in judge_result.scores:
                    if score.passed is not None:
                        all_scores.setdefault(score.criterion, []).append(score.passed)

                all_observations.append({
                    "video_id": video_id,
                    "piece": piece_id,
                    "skill_level": skill_level,
                    "tier": tier,
                    "framing": framing,
                    "observation": obs.text[:300],
                    "judge_scores": {
                        s.criterion: s.passed for s in judge_result.scores
                    },
                    "judge_evidence": {
                        s.criterion: s.evidence for s in judge_result.scores
                    },
                })

            print(f"{len(result.observations)} obs")

    # Build report
    report = EvalReport(
        eval_name="practice_eval",
        eval_version="2.0",
        dataset=f"practice_{total_recordings}",
        metrics={},
    )
    for criterion, scores in all_scores.items():
        if scores:
            mean = sum(scores) / len(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            report.metrics[criterion] = MetricResult(mean=mean, std=std, n=len(scores))

    report.metadata["total_recordings"] = total_recordings
    report.metadata["recordings_with_observations"] = recordings_with_obs
    report.metadata["stop_trigger_rate"] = (
        recordings_with_obs / total_recordings if total_recordings > 0 else 0
    )
    report.metadata["tier_distribution"] = tier_counts
    report.metadata["framing_distribution"] = framing_counts
    report.metadata["total_observations"] = len(all_observations)
    report.cost = {
        "judge_calls": total_judge_calls,
        "estimated_usd": round(total_judge_calls * 0.003, 2),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save(REPORTS_DIR / "practice_eval.json")
    report.print_summary()

    obs_path = REPORTS_DIR / "practice_eval_observations.json"
    with open(obs_path, "w") as f:
        json.dump(all_observations, f, indent=2)
    print(f"  Detailed observations: {obs_path}")


if __name__ == "__main__":
    main()
```

**Known limitation (v1):** The Worker does not support a `set_baselines` WebSocket message, so the eval-student's baselines default to SCALER_MEAN. This means STOP triggers only when the model scores are genuinely low (below the population mean), not relative to the student's personal history. For beginner/intermediate practice recordings, this should still trigger STOP on weak dimensions. If STOP trigger rate is too low, adding `set_baselines` to the Worker becomes a priority fix.

- [ ] **Step 2: HUMAN STEP -- Start wrangler dev in a separate terminal**

```bash
cd apps/api && wrangler dev
```
Verify: `curl http://localhost:8787/health`

- [ ] **Step 3: Run practice eval**

Run: `cd apps/evals && uv run python -m pipeline.practice_eval.eval_practice`
Expected: Observations generated and judged. Report at `reports/practice_eval.json`.

- [ ] **Step 4: HUMAN STEP -- Hand-score 10-15 observations for calibration**

Read `reports/practice_eval_observations.json`. Pick 10-15 observations. Rate each criterion independently. Compare with LLM judge. If disagreement > 30% on any criterion, revise the v2 judge prompt before trusting results.

- [ ] **Step 5: Commit**

```bash
git add apps/evals/pipeline/practice_eval/eval_practice.py
git add apps/evals/reports/practice_eval.json
git add apps/evals/reports/practice_eval_observations.json
git commit -m "feat(evals): add practice recording eval runner + first results"
```

---

### Task 9: Analyze results and decide priorities

- [ ] **Step 1: HUMAN STEP -- Review skill eval results**

Check analysis output from Task 5 Step 5. Key questions:
- Spearman rho > 0.3? (model tracks skill)
- Confusion rate < 0.40? (better than chance)
- Which dimensions separate best/worst?
- Any inversions at specific levels?

- [ ] **Step 2: HUMAN STEP -- Review practice eval results**

Check `apps/evals/reports/practice_eval.json`. Key questions:
- STOP trigger rate (should be > 50% on practice recordings)
- Tier distribution (most should be Tier 1)
- Per-criterion pass rates
- Corrective vs positive framing quality difference

- [ ] **Step 3: Write analysis summary in conversation**

Synthesize both eval results:
- Eval 1 verdict: does the model separate skill levels?
- Eval 2 verdict: does the pipeline produce useful feedback?
- Bottleneck: model, pipeline, or both?
- Recommended next priorities

- [ ] **Step 4: Commit**

```bash
git add apps/evals/reports/
git commit -m "data: eval system redesign complete, results analyzed"
```
