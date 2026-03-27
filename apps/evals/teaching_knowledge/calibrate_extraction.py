"""Compare LLM extraction against manual annotations.

Measures agreement on categorical fields (dimension_focus, student_skill_estimate,
feedback_type) and qualitative alignment on what_teacher_said.

Usage:
  uv run python -m apps.evals.teaching_knowledge.calibrate_extraction \
    --manual-dir data/manual_annotations \
    --llm-output data/raw_teaching_db.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def load_manual_annotations(manual_dir: Path) -> dict[str, list[dict]]:
    """Load manual annotations keyed by video_id."""
    annotations = {}
    for f in manual_dir.glob("*.json"):
        video_id = f.stem
        annotations[video_id] = json.loads(f.read_text())
    return annotations


def load_llm_extractions(llm_path: Path) -> dict[str, list[dict]]:
    """Load LLM extractions keyed by source_id."""
    all_moments = json.loads(llm_path.read_text())
    by_source = {}
    for m in all_moments:
        sid = m.get("source_id", "unknown")
        by_source.setdefault(sid, []).append(m)
    return by_source


def compare_field(manual_vals: list[str], llm_vals: list[str], field_name: str) -> dict:
    """Compare a categorical field between manual and LLM extractions."""
    # Use Counter-based comparison (order-independent)
    manual_counts = Counter(manual_vals)
    llm_counts = Counter(llm_vals)
    all_vals = set(manual_counts.keys()) | set(llm_counts.keys())

    matches = sum(min(manual_counts.get(v, 0), llm_counts.get(v, 0)) for v in all_vals)
    total = max(len(manual_vals), len(llm_vals))
    agreement = matches / total if total > 0 else 0.0

    return {
        "field": field_name,
        "manual_count": len(manual_vals),
        "llm_count": len(llm_vals),
        "agreement": round(agreement, 3),
        "manual_distribution": dict(manual_counts),
        "llm_distribution": dict(llm_counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-dir", type=Path, required=True)
    parser.add_argument("--llm-output", type=Path, required=True)
    parser.add_argument("--target-agreement", type=float, default=0.8)
    args = parser.parse_args()

    manual = load_manual_annotations(args.manual_dir)
    llm = load_llm_extractions(args.llm_output)

    if not manual:
        print("ERROR: No manual annotations found. Annotate at least 20 transcripts first.")
        print(f"  Directory: {args.manual_dir}")
        return

    overlap_ids = set(manual.keys()) & set(llm.keys())
    print(f"Manual annotations: {len(manual)} videos")
    print(f"LLM extractions: {len(llm)} videos")
    print(f"Overlap: {len(overlap_ids)} videos\n")

    if not overlap_ids:
        print("ERROR: No overlapping video IDs between manual and LLM extractions.")
        return

    # Aggregate field comparisons
    for field in ["dimension_focus", "student_skill_estimate", "feedback_type"]:
        all_manual = []
        all_llm = []
        for vid in overlap_ids:
            all_manual.extend(m.get(field, "unknown") for m in manual[vid])
            all_llm.extend(m.get(field, "unknown") for m in llm[vid])

        result = compare_field(all_manual, all_llm, field)
        status = "PASS" if result["agreement"] >= args.target_agreement else "FAIL"
        print(f"{field}: {result['agreement']:.1%} agreement [{status}]")
        print(f"  Manual: {result['manual_distribution']}")
        print(f"  LLM:    {result['llm_distribution']}")
        print()

    # Moment count comparison
    manual_total = sum(len(v) for v in manual.values())
    llm_total = sum(len(llm.get(vid, [])) for vid in overlap_ids)
    print(f"Moment count: manual={manual_total}, LLM={llm_total}")
    print(f"  Ratio: {llm_total/manual_total:.2f}x" if manual_total > 0 else "  No manual data")


if __name__ == "__main__":
    main()
