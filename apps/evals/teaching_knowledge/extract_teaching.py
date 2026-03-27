"""Extract teaching moments from transcripts using LLM.

Two passes:
  Pass 1 (Filter): Is this transcript a teaching/feedback moment?
  Pass 2 (Extract): 4-field structured extraction on confirmed moments.

Usage:
  uv run python -m apps.evals.teaching_knowledge.extract_teaching \
    --manifest data/transcripts/manifest_t2.json \
    --output data/raw_teaching_db.json
"""
import argparse
import json
from pathlib import Path

import anthropic

DATA_DIR = Path(__file__).parent / "data"

FILTER_PROMPT = """You are analyzing a transcript from a piano-related YouTube video.
Determine if this transcript contains real teaching or feedback moments where a teacher
gives specific musical guidance to a student.

Classify as:
- TEACHING: Contains specific musical feedback, critique, or instruction from teacher to student
- PERFORMANCE_ONLY: Just a performance with no teaching dialogue
- GENERAL: Discussion, interview, or Q&A without specific musical instruction
- TUTORIAL: Generic tutorial/how-to without real teacher-student interaction

Respond with a JSON object:
{
  "classification": "TEACHING" | "PERFORMANCE_ONLY" | "GENERAL" | "TUTORIAL",
  "confidence": 0.0-1.0,
  "reason": "one sentence explaining why"
}

TRANSCRIPT:
"""

EXTRACT_PROMPT = """You are extracting structured teaching moments from a piano masterclass
or lesson transcript. For each distinct teaching moment (where the teacher gives specific
musical feedback), extract these 4 fields:

1. what_teacher_said: Verbatim or close paraphrase of the teaching moment
2. dimension_focus: Which musical dimension is being addressed?
   Options: dynamics | timing | pedaling | articulation | phrasing | interpretation | general
3. student_skill_estimate: Based on context clues, what level is the student?
   Options: beginner | early_intermediate | intermediate | advanced | professional
4. feedback_type: What kind of teaching behavior is this?
   Options: corrective | encouraging | modeling | guided_discovery | scaffolding | motivational

Return a JSON array of teaching moments. Each moment is one distinct piece of feedback.
If the transcript contains multiple teaching moments, extract all of them.

TRANSCRIPT:
"""


def filter_transcript(client: anthropic.Anthropic, text: str, model: str) -> dict:
    """Pass 1: Classify whether transcript contains teaching moments."""
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": FILTER_PROMPT + text[:8000]}],
    )
    try:
        return json.loads(response.content[0].text)
    except (json.JSONDecodeError, IndexError):
        return {"classification": "GENERAL", "confidence": 0.0, "reason": "Parse error"}


def extract_moments(client: anthropic.Anthropic, text: str, model: str) -> list[dict]:
    """Pass 2: Extract 4-field teaching moments from confirmed teaching transcript."""
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": EXTRACT_PROMPT + text[:12000]}],
    )
    try:
        result = json.loads(response.content[0].text)
        if isinstance(result, list):
            return result
        return [result]
    except (json.JSONDecodeError, IndexError):
        return []


VALID_DIMENSIONS = {"dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation", "general"}
VALID_SKILLS = {"beginner", "early_intermediate", "intermediate", "advanced", "professional"}
VALID_TYPES = {"corrective", "encouraging", "modeling", "guided_discovery", "scaffolding", "motivational"}


def validate_moment(moment: dict) -> bool:
    """Validate that extracted moment has all required fields with valid values."""
    required = ["what_teacher_said", "dimension_focus", "student_skill_estimate", "feedback_type"]
    if not all(k in moment for k in required):
        return False
    if moment["dimension_focus"] not in VALID_DIMENSIONS:
        return False
    if moment["student_skill_estimate"] not in VALID_SKILLS:
        return False
    if moment["feedback_type"] not in VALID_TYPES:
        return False
    if len(moment["what_teacher_said"]) < 10:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract teaching moments from transcripts")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to transcript manifest JSON")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "raw_teaching_db.json")
    parser.add_argument("--filter-model", type=str, default="claude-haiku-4-5-20251001", help="Model for Pass 1 filtering")
    parser.add_argument("--extract-model", type=str, default="claude-sonnet-4-6-20250514", help="Model for Pass 2 extraction")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of transcripts to process")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if args.limit:
        manifest = manifest[:args.limit]

    client = anthropic.Anthropic()
    all_moments = []
    stats = {"total": len(manifest), "teaching": 0, "filtered_out": 0, "moments_extracted": 0, "validation_failed": 0}

    for i, entry in enumerate(manifest):
        video_id = entry["video_id"]
        transcript_path = Path(entry["path"])
        print(f"[{i+1}/{len(manifest)}] {video_id}...", end=" ")

        if not transcript_path.exists():
            print("SKIP (file missing)")
            continue

        # Read and parse transcript
        from apps.evals.teaching_knowledge.download_transcripts import parse_vtt
        text = parse_vtt(transcript_path)

        # Pass 1: Filter
        filter_result = filter_transcript(client, text, args.filter_model)
        if filter_result["classification"] != "TEACHING" or filter_result["confidence"] < 0.6:
            stats["filtered_out"] += 1
            print(f"FILTERED ({filter_result['classification']}, {filter_result['confidence']:.1f})")
            continue

        stats["teaching"] += 1

        # Pass 2: Extract
        moments = extract_moments(client, text, args.extract_model)
        valid_moments = []
        for m in moments:
            if validate_moment(m):
                m["source_id"] = video_id
                m["source_type"] = entry.get("source", "unknown")
                valid_moments.append(m)
            else:
                stats["validation_failed"] += 1

        all_moments.extend(valid_moments)
        stats["moments_extracted"] += len(valid_moments)
        print(f"EXTRACTED {len(valid_moments)} moments")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_moments, indent=2))
    print(f"\nStats: {json.dumps(stats, indent=2)}")
    print(f"Total moments saved: {len(all_moments)} -> {args.output}")


if __name__ == "__main__":
    main()
