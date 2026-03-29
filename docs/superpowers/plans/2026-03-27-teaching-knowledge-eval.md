# Teaching Knowledge Extraction & Research-Driven Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract teaching expertise from masterclass transcripts and pedagogy research, synthesize into a Teaching Playbook, then evaluate the CrescendAI pipeline against research-derived rubrics instead of proxy metrics.

**Architecture:** Two parallel tracks. Track A extracts teaching knowledge from YouTube transcripts + pedagogy literature and synthesizes a playbook. Track B fixes eval infrastructure plumbing. They converge at Phase 1 (pipeline alignment) and Phase 2 (eval execution). See spec: `docs/superpowers/specs/2026-03-26-teaching-knowledge-eval-design.md`.

**Tech Stack:** Python (uv), yt-dlp, Anthropic SDK (Claude Sonnet), existing Rust API (Cloudflare Workers), existing eval framework (pytest, asyncio WebSocket client).

---

## File Structure

### Track A: New scripts (teaching knowledge extraction)

```
apps/evals/teaching_knowledge/
  download_transcripts.py     # yt-dlp wrapper, quality gate, batch download
  extract_teaching.py         # LLM Pass 1 (filter) + Pass 2 (4-field extraction)
  calibrate_extraction.py     # Compare LLM vs manual annotations
  synthesize_playbook.py      # 3-round LLM synthesis -> playbook YAML
  derive_rubrics.py           # Playbook -> judge prompt templates
  pedagogy_research.py        # Web search + LLM synthesis of literature

apps/evals/teaching_knowledge/data/
  manual_annotations/         # 20 hand-annotated transcripts (calibration)
  transcripts/                # Downloaded transcript files
  raw_teaching_db.json        # Extracted teaching moments
  playbook.yaml               # Final Teaching Playbook
```

### Track B: Modified files (eval infrastructure)

```
apps/api/src/practice/session_finalization.rs   # Add eval query param flag
apps/evals/shared/pipeline_client.py            # Capture eval_context
apps/evals/pipeline/practice_eval/eval_practice.py  # Fix student ID, STOP config
config/stop_config.json                         # Shared STOP weights (new)
apps/api/src/services/stop.rs                   # Read from config instead of hardcoded
```

### Phase 1: Modified files (pipeline alignment)

```
apps/api/src/services/prompts.rs                # Tier-aware synthesis system prompt
apps/api/src/practice/accumulator.rs            # Piece-style dimension weights in top_moments()
apps/evals/shared/prompts/                      # New research-derived judge prompts
apps/evals/pipeline/practice_eval/analyze_e2e.py # Failure attribution logic
```

---

## TRACK A: Teaching Knowledge Research

### Task 1: Transcript Download Pipeline

**Files:**
- Create: `apps/evals/teaching_knowledge/__init__.py`
- Create: `apps/evals/teaching_knowledge/download_transcripts.py`

- [ ] **Step 1: Create the teaching_knowledge package**

```bash
mkdir -p apps/evals/teaching_knowledge/data/manual_annotations
mkdir -p apps/evals/teaching_knowledge/data/transcripts
touch apps/evals/teaching_knowledge/__init__.py
```

- [ ] **Step 2: Write the transcript downloader**

```python
"""Download YouTube transcripts with quality gating.

Usage:
  uv run python -m apps.evals.teaching_knowledge.download_transcripts \
    --source t2 --limit 50
  uv run python -m apps.evals.teaching_knowledge.download_transcripts \
    --source search --query "piano masterclass feedback" --limit 100
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "transcripts"
T2_MANIFEST_DIR = Path("model/data/evals/skill_eval")

NOISE_TOKENS = re.compile(r"\[(?:inaudible|Music|Applause|Laughter)\]", re.IGNORECASE)


def download_transcript(video_id: str, output_dir: Path) -> Path | None:
    """Download auto-generated subtitles for a YouTube video.
    Returns path to transcript file, or None if download fails or quality gate fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / video_id

    # Skip if already downloaded
    srt_file = out_path.with_suffix(".en.vtt")
    if srt_file.exists():
        return srt_file

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--write-auto-sub",
                "--sub-lang", "en",
                "--skip-download",
                "--output", str(out_path),
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"  SKIP {video_id}: yt-dlp error: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  SKIP {video_id}: download timeout")
        return None

    # Find the downloaded subtitle file (yt-dlp may use .vtt or .srt)
    for ext in [".en.vtt", ".en.srt", ".vtt", ".srt"]:
        candidate = out_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    print(f"  SKIP {video_id}: no subtitle file found")
    return None


def parse_vtt(path: Path) -> str:
    """Extract plain text from VTT/SRT subtitle file."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    text_lines = []
    for line in lines:
        # Skip timestamps, headers, empty lines
        if "-->" in line or line.strip().isdigit() or line.startswith("WEBVTT") or not line.strip():
            continue
        # Remove VTT formatting tags
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if clean and clean not in text_lines[-1:]:  # deduplicate consecutive
            text_lines.append(clean)
    return " ".join(text_lines)


def quality_gate(text: str, min_words: int = 500, max_noise_ratio: float = 0.3) -> bool:
    """Check if transcript passes quality thresholds."""
    words = text.split()
    if len(words) < min_words:
        return False
    noise_count = len(NOISE_TOKENS.findall(text))
    if noise_count / max(len(words), 1) > max_noise_ratio:
        return False
    return True


def get_t2_video_ids(limit: int = 50) -> list[str]:
    """Extract video IDs from T2 masterclass manifest files."""
    video_ids = []
    for manifest_dir in sorted(T2_MANIFEST_DIR.iterdir()):
        manifest = manifest_dir / "manifest.yaml"
        if not manifest.exists():
            continue
        # Simple YAML parsing for video_id fields
        text = manifest.read_text()
        for match in re.finditer(r"video_id:\s*([A-Za-z0-9_-]+)", text):
            video_ids.append(match.group(1))
    # Deduplicate (same video may appear in multiple pieces)
    seen = set()
    unique = []
    for vid in video_ids:
        if vid not in seen:
            seen.add(vid)
            unique.append(vid)
    return unique[:limit]


def search_youtube(query: str, limit: int = 100) -> list[str]:
    """Search YouTube for video IDs matching a query."""
    try:
        result = subprocess.run(
            ["yt-dlp", f"ytsearch{limit}:{query}", "--get-id", "--flat-playlist"],
            capture_output=True, text=True, timeout=120,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"  WARNING: yt-dlp search failed for '{query}'")
        return []


def main():
    parser = argparse.ArgumentParser(description="Download YouTube transcripts")
    parser.add_argument("--source", choices=["t2", "search"], required=True)
    parser.add_argument("--query", type=str, help="Search query (for --source search)")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    if args.source == "t2":
        video_ids = get_t2_video_ids(args.limit)
        print(f"Found {len(video_ids)} T2 video IDs (limit={args.limit})")
    else:
        if not args.query:
            print("ERROR: --query required for --source search")
            sys.exit(1)
        video_ids = search_youtube(args.query, args.limit)
        print(f"Found {len(video_ids)} videos for '{args.query}'")

    results = {"downloaded": 0, "quality_pass": 0, "quality_fail": 0, "error": 0}
    manifest = []

    for i, vid in enumerate(video_ids):
        print(f"[{i+1}/{len(video_ids)}] {vid}...", end=" ")
        path = download_transcript(vid, args.output_dir)
        if path is None:
            results["error"] += 1
            continue

        results["downloaded"] += 1
        text = parse_vtt(path)

        if quality_gate(text):
            results["quality_pass"] += 1
            manifest.append({
                "video_id": vid,
                "path": str(path),
                "word_count": len(text.split()),
                "source": args.source,
            })
            print(f"PASS ({len(text.split())} words)")
        else:
            results["quality_fail"] += 1
            print(f"FAIL (quality gate)")

    # Save manifest
    manifest_path = args.output_dir / f"manifest_{args.source}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nResults: {results}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test with 5 T2 videos**

```bash
cd apps/evals && uv run python -m teaching_knowledge.download_transcripts --source t2 --limit 5
```

Expected: Downloads 5 transcripts, reports quality gate pass/fail for each.

- [ ] **Step 4: Commit**

```bash
git add apps/evals/teaching_knowledge/
git commit -m "$(cat <<'EOF'
feat: transcript download pipeline with quality gating

Downloads YouTube auto-captions via yt-dlp, parses VTT/SRT to plain text,
applies quality gate (min 500 words, max 30% noise tokens). Supports T2
masterclass video IDs from manifests and YouTube search queries.
EOF
)"
```

---

### Task 2: Teaching Moment Extraction Pipeline

**Files:**
- Create: `apps/evals/teaching_knowledge/extract_teaching.py`

- [ ] **Step 1: Write the extraction script**

```python
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
import sys
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
```

- [ ] **Step 2: Test extraction on 3 transcripts**

```bash
cd apps/evals && uv run python -m teaching_knowledge.extract_teaching \
  --manifest teaching_knowledge/data/transcripts/manifest_t2.json \
  --limit 3
```

Expected: Filters transcripts, extracts teaching moments with 4 fields each, saves to `raw_teaching_db.json`.

- [ ] **Step 3: Commit**

```bash
git add apps/evals/teaching_knowledge/extract_teaching.py
git commit -m "$(cat <<'EOF'
feat: two-pass teaching moment extraction pipeline

Pass 1 (Haiku): filters transcripts for teaching content.
Pass 2 (Sonnet): extracts 4-field structured moments (what_teacher_said,
dimension_focus, student_skill_estimate, feedback_type) with schema validation.
EOF
)"
```

---

### Task 3: Extraction Calibration

**Files:**
- Create: `apps/evals/teaching_knowledge/calibrate_extraction.py`

- [ ] **Step 1: Create manual annotation template**

```bash
cat > apps/evals/teaching_knowledge/data/manual_annotations/README.md << 'EOF'
# Manual Annotations for Extraction Calibration

Annotate 20 transcripts (10 masterclass, 10 lesson) using the 4-field schema.
Save each as a JSON file named {video_id}.json with this structure:

[
  {
    "what_teacher_said": "verbatim or close paraphrase",
    "dimension_focus": "dynamics|timing|pedaling|articulation|phrasing|interpretation|general",
    "student_skill_estimate": "beginner|early_intermediate|intermediate|advanced|professional",
    "feedback_type": "corrective|encouraging|modeling|guided_discovery|scaffolding|motivational"
  }
]

Target: 20 transcripts annotated by founder (Jai).
EOF
```

- [ ] **Step 2: Write calibration comparison script**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add apps/evals/teaching_knowledge/calibrate_extraction.py
git add apps/evals/teaching_knowledge/data/manual_annotations/
git commit -m "feat: extraction calibration script (compares LLM vs manual annotations)"
```

**CHECKPOINT: Human task.** Jai must manually annotate 20 transcripts before calibration can run. This is ~2-3 hours of listening and annotating. Run calibration after annotations are complete:

```bash
cd apps/evals && uv run python -m teaching_knowledge.calibrate_extraction \
  --manual-dir teaching_knowledge/data/manual_annotations \
  --llm-output teaching_knowledge/data/raw_teaching_db.json
```

If agreement < 80% on any field, simplify the schema and re-extract.

---

### Task 4: Pedagogy Literature Research

**Files:**
- Create: `apps/evals/teaching_knowledge/pedagogy_research.py`

- [ ] **Step 1: Write the pedagogy research synthesizer**

```python
"""Synthesize piano pedagogy principles from literature and web research.

Produces a structured summary of teaching frameworks, principles, and
evidence-based strategies for piano instruction across skill levels.

Usage:
  uv run python -m apps.evals.teaching_knowledge.pedagogy_research \
    --output data/pedagogy_principles.json
"""
import argparse
import json
from pathlib import Path

import anthropic

DATA_DIR = Path(__file__).parent / "data"

RESEARCH_PROMPT = """You are a music education researcher. Synthesize the key principles
from piano pedagogy literature and teaching methodology. Cover these frameworks:

1. Suzuki Method -- key principles, how teachers interact with students
2. ABRSM / RCM exam structure -- how skill levels are defined, what's tested at each grade
3. Taubman Approach -- technical teaching methodology
4. Deliberate Practice (Ericsson) -- how skill acquisition works, what practice is effective
5. Zone of Proximal Development (Vygotsky) -- how to pitch feedback at the right level
6. Motor Learning Theory -- how physical skills are acquired, role of feedback timing
7. Self-Determination Theory (Deci/Ryan) -- intrinsic motivation in music learning
8. Flow Theory (Csikszentmihalyi) -- optimal challenge level for engagement

For each framework, extract:
- Core teaching principles (what the teacher should DO)
- What the teacher should NOT do (common mistakes)
- How feedback should differ by student skill level
- Evidence for effectiveness

Also synthesize cross-framework patterns:
- What do all great teaching methodologies agree on?
- Where do they disagree?
- What's the consensus on feedback timing, frequency, and type?

Return a JSON object with this structure:
{
  "frameworks": [
    {
      "name": "string",
      "core_principles": ["string"],
      "teacher_should_not": ["string"],
      "skill_level_adaptation": {"beginner": "string", "intermediate": "string", "advanced": "string"},
      "evidence": "string"
    }
  ],
  "cross_framework_patterns": {
    "consensus": ["string"],
    "disagreements": ["string"],
    "feedback_principles": ["string"]
  }
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA_DIR / "pedagogy_principles.json")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6-20250514")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    print("Synthesizing pedagogy literature...")

    response = client.messages.create(
        model=args.model,
        max_tokens=8000,
        messages=[{"role": "user", "content": RESEARCH_PROMPT}],
    )

    try:
        result = json.loads(response.content[0].text)
    except json.JSONDecodeError:
        # Save raw text if JSON parsing fails
        result = {"raw_response": response.content[0].text}
        print("WARNING: Response was not valid JSON. Saving raw text.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Saved {len(result.get('frameworks', []))} frameworks to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
cd apps/evals && uv run python -m teaching_knowledge.pedagogy_research
```

Expected: Produces `data/pedagogy_principles.json` with 8+ framework summaries.

- [ ] **Step 3: Commit**

```bash
git add apps/evals/teaching_knowledge/pedagogy_research.py
git commit -m "feat: pedagogy literature synthesis (8 frameworks, cross-framework patterns)"
```

---

### Task 5: Playbook Synthesis

**Files:**
- Create: `apps/evals/teaching_knowledge/synthesize_playbook.py`

- [ ] **Step 1: Write the playbook synthesizer**

```python
"""Synthesize Teaching Playbook from raw teaching database + pedagogy principles.

Three rounds:
  Round 1: Cluster teaching moments by skill level and feedback type
  Round 2: Extract patterns, dimension priorities, language patterns
  Round 3: Founder review checkpoint (manual)

Usage:
  uv run python -m apps.evals.teaching_knowledge.synthesize_playbook \
    --teaching-db data/raw_teaching_db.json \
    --pedagogy data/pedagogy_principles.json \
    --output data/playbook.yaml
"""
import argparse
import json
from pathlib import Path

import anthropic
import yaml

DATA_DIR = Path(__file__).parent / "data"

ROUND1_PROMPT = """You are analyzing a database of {n_moments} teaching moments extracted
from real piano masterclasses and lessons, plus {n_frameworks} pedagogical frameworks.

TEACHING MOMENTS (sample of {sample_size}):
{moments_sample}

PEDAGOGY PRINCIPLES:
{pedagogy_summary}

TASK: Group these teaching moments into natural clusters. Do NOT pre-assume an organizing
principle (tiers, moves, etc.). Let the patterns emerge from the data.

For each cluster you identify:
1. Name it descriptively (what teaching behavior defines this cluster?)
2. List 3-5 representative examples from the data
3. Note which skill levels this cluster appears at
4. Note which dimensions this cluster focuses on
5. Note the distribution of feedback types (corrective vs encouraging etc.)

Return a JSON object with clusters as an array.
"""

ROUND2_PROMPT = """You are building a Teaching Playbook for an AI piano teacher.

Round 1 identified these clusters:
{clusters}

Raw teaching data ({n_moments} moments) and pedagogy research ({n_frameworks} frameworks)
are available.

FULL TEACHING DATA:
{all_moments}

For each cluster, now extract:
1. Dominant teaching strategies and when to use them
2. Dimension priorities by repertoire style (e.g., pedaling for Chopin vs articulation for Bach)
3. Language patterns and register (warm, direct, technical, metaphorical)
4. Good feedback examples (with evidence citations from the data)
5. Bad feedback patterns (what mediocre teachers do instead)
6. What distinguishes great feedback from mediocre in this cluster

Also produce cross-cluster insights:
- What do great piano teachers do that mediocre ones don't?
- What should the AI teacher NEVER say at each skill level?
- Piece-style dimension rules (which dimensions matter for which composers/styles?)

Return a YAML document that could serve as a Teaching Playbook.
"""


def sample_moments(moments: list[dict], max_sample: int = 50) -> str:
    """Sample moments for prompt context, preserving diversity."""
    if len(moments) <= max_sample:
        return json.dumps(moments, indent=2)

    # Stratified sample by skill level
    by_skill = {}
    for m in moments:
        skill = m.get("student_skill_estimate", "unknown")
        by_skill.setdefault(skill, []).append(m)

    per_skill = max_sample // max(len(by_skill), 1)
    sample = []
    for skill_moments in by_skill.values():
        sample.extend(skill_moments[:per_skill])

    return json.dumps(sample[:max_sample], indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teaching-db", type=Path, default=DATA_DIR / "raw_teaching_db.json")
    parser.add_argument("--pedagogy", type=Path, default=DATA_DIR / "pedagogy_principles.json")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "playbook.yaml")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6-20250514")
    args = parser.parse_args()

    moments = json.loads(args.teaching_db.read_text())
    pedagogy = json.loads(args.pedagogy.read_text())
    client = anthropic.Anthropic()

    print(f"Loaded {len(moments)} teaching moments, {len(pedagogy.get('frameworks', []))} frameworks")

    # Round 1: Clustering
    print("\n--- Round 1: Clustering ---")
    r1_prompt = ROUND1_PROMPT.format(
        n_moments=len(moments),
        n_frameworks=len(pedagogy.get("frameworks", [])),
        sample_size=min(50, len(moments)),
        moments_sample=sample_moments(moments),
        pedagogy_summary=json.dumps(pedagogy.get("cross_framework_patterns", {}), indent=2),
    )
    r1_response = client.messages.create(model=args.model, max_tokens=8000, messages=[{"role": "user", "content": r1_prompt}])
    clusters = r1_response.content[0].text
    print(f"Clusters identified. Response length: {len(clusters)} chars")

    # Save Round 1 output for review
    r1_path = DATA_DIR / "playbook_round1_clusters.json"
    r1_path.write_text(clusters)
    print(f"Round 1 saved to {r1_path}")

    # Round 2: Pattern extraction
    print("\n--- Round 2: Pattern Extraction ---")
    r2_prompt = ROUND2_PROMPT.format(
        clusters=clusters,
        n_moments=len(moments),
        n_frameworks=len(pedagogy.get("frameworks", [])),
        all_moments=json.dumps(moments, indent=2)[:50000],  # Cap at 50K chars
    )
    r2_response = client.messages.create(model=args.model, max_tokens=16000, messages=[{"role": "user", "content": r2_prompt}])
    playbook_text = r2_response.content[0].text
    print(f"Playbook draft generated. Response length: {len(playbook_text)} chars")

    # Try to parse as YAML, fall back to saving raw text
    try:
        # Strip markdown code fences if present
        clean = playbook_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        playbook = yaml.safe_load(clean)
        args.output.write_text(yaml.dump(playbook, default_flow_style=False, allow_unicode=True))
    except yaml.YAMLError:
        args.output.write_text(playbook_text)
        print("WARNING: Could not parse as YAML. Saved raw text.")

    print(f"\nPlaybook saved to {args.output}")
    print("\n--- Round 3: FOUNDER REVIEW REQUIRED ---")
    print("Review the playbook at:")
    print(f"  {args.output}")
    print("\nQuality gate checklist:")
    print("  [ ] At least 3 distinct feedback patterns per skill level")
    print("  [ ] At least 1 pattern that surprises you (something the pipeline doesn't do)")
    print("  [ ] Piece-style dimension rules present (Bach vs Chopin etc.)")
    print("  [ ] Good AND bad examples for each pattern")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add apps/evals/teaching_knowledge/synthesize_playbook.py
git commit -m "feat: 3-round playbook synthesis (cluster -> extract -> founder review)"
```

**CHECKPOINT: Founder review.** After running, Jai reviews the playbook YAML. If it fails the quality gate (< 3 patterns per level, no surprises), add more source data and re-run.

---

### Task 6: Rubric Derivation

**Files:**
- Create: `apps/evals/teaching_knowledge/derive_rubrics.py`

- [ ] **Step 1: Write the rubric derivation script**

```python
"""Derive eval rubrics from the Teaching Playbook.

Reads the playbook, identifies quality dimensions that distinguish great from
mediocre teaching, and generates judge prompt templates with real examples.

Usage:
  uv run python -m apps.evals.teaching_knowledge.derive_rubrics \
    --playbook data/playbook.yaml \
    --output-dir ../shared/prompts/
"""
import argparse
import json
from pathlib import Path

import anthropic
import yaml

RUBRIC_PROMPT = """You are designing an evaluation rubric for an AI piano teacher.

Here is the Teaching Playbook (derived from real masterclass analysis and pedagogy research):
{playbook}

TASK: Design evaluation dimensions and a scoring rubric for judging whether the AI
teacher's synthesis output is good. The dimensions and scales should emerge from the
playbook -- what does the playbook say distinguishes great teaching from mediocre?

For each dimension:
1. Name it (something specific, not generic like "quality")
2. Define what each score level looks like (with concrete examples from the playbook)
3. Include a "zero" score that flags critical failures (would damage student trust)
4. Include calibration examples: one good and one bad synthesis example per dimension

Also produce:
- A system prompt for the LLM judge that includes the rubric
- Instructions for how to use the playbook's piece-style rules in scoring
- The format for judge output (criterion name, score, evidence)

Return a JSON object with:
{{
  "dimensions": [
    {{
      "name": "string",
      "description": "string",
      "scale": {{"0": "string", "1": "string", "2": "string", "3": "string"}},
      "good_example": {{"context": "string", "synthesis": "string", "score": 3, "why": "string"}},
      "bad_example": {{"context": "string", "synthesis": "string", "score": 0, "why": "string"}}
    }}
  ],
  "judge_system_prompt": "string",
  "scoring_format": "string"
}}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--playbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("apps/evals/shared/prompts"))
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6-20250514")
    args = parser.parse_args()

    playbook_text = args.playbook.read_text()
    client = anthropic.Anthropic()

    print("Deriving eval rubrics from playbook...")
    response = client.messages.create(
        model=args.model,
        max_tokens=8000,
        messages=[{"role": "user", "content": RUBRIC_PROMPT.format(playbook=playbook_text[:30000])}],
    )

    try:
        rubrics = json.loads(response.content[0].text)
    except json.JSONDecodeError:
        print("ERROR: Could not parse rubrics as JSON")
        print(response.content[0].text[:2000])
        return

    # Write judge system prompt
    judge_prompt_path = args.output_dir / "synthesis_quality_judge_v2.txt"
    judge_prompt_path.write_text(rubrics["judge_system_prompt"])
    print(f"Judge prompt saved to {judge_prompt_path}")

    # Write full rubric definition (for reference and calibration)
    rubric_path = args.output_dir / "rubric_definition.json"
    rubric_path.write_text(json.dumps(rubrics, indent=2))
    print(f"Full rubric definition saved to {rubric_path}")

    # Summary
    dims = rubrics.get("dimensions", [])
    print(f"\nDerived {len(dims)} evaluation dimensions:")
    for d in dims:
        print(f"  - {d['name']}: {d['description'][:80]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add apps/evals/teaching_knowledge/derive_rubrics.py
git commit -m "feat: derive eval rubrics from Teaching Playbook (playbook -> judge prompts)"
```

---

## TRACK B: Eval Infrastructure (parallel with Track A)

### Task 7: Shared STOP Config

**Files:**
- Create: `config/stop_config.json`
- Modify: `apps/api/src/services/stop.rs`
- Modify: `apps/evals/pipeline/practice_eval/eval_practice.py`

- [ ] **Step 1: Create shared config file**

```bash
mkdir -p config
cat > config/stop_config.json << 'STOPEOF'
{
  "scaler_mean": [0.5450, 0.4848, 0.4594, 0.5369, 0.5188, 0.5064],
  "scaler_std": [0.0689, 0.0388, 0.0791, 0.0154, 0.0186, 0.0555],
  "weights": [-0.5266, 0.3681, -0.5483, 0.4884, 0.2427, -0.1541],
  "bias": 0.1147,
  "threshold": 0.5,
  "dimensions": ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]
}
STOPEOF
```

- [ ] **Step 2: Update Python eval runner to read from config**

Read `apps/evals/pipeline/practice_eval/eval_practice.py` and replace the hardcoded STOP constants with config file loading. Find the `stop_probability()` function and replace the hardcoded arrays with values loaded from `config/stop_config.json`.

- [ ] **Step 3: Verify Rust stop.rs constants match config**

Read `apps/api/src/services/stop.rs` and verify the hardcoded values match `config/stop_config.json`. Note: Rust side reads from config in a future task (requires build system change). For now, add a comment referencing the config file.

- [ ] **Step 4: Commit**

```bash
git add config/stop_config.json apps/evals/pipeline/practice_eval/eval_practice.py apps/api/src/services/stop.rs
git commit -m "$(cat <<'EOF'
feat: shared STOP config file (eliminates Python/Rust constant duplication)

Moves STOP classifier weights to config/stop_config.json. Python eval runner
reads from config. Rust side still hardcoded but references config file in comment.
EOF
)"
```

---

### Task 8: Eval Context Query Parameter

**Files:**
- Modify: `apps/api/src/practice/session_finalization.rs`
- Modify: `apps/evals/shared/pipeline_client.py`

- [ ] **Step 1: Read session_finalization.rs to understand current eval_context export**

Read `apps/api/src/practice/session_finalization.rs` lines 150-200 to understand the current `ENVIRONMENT=development` check.

- [ ] **Step 2: Add eval flag to session start**

Modify the session start endpoint to accept an `eval=true` query parameter. Store this flag on the Durable Object session state. In `session_finalization.rs`, check this flag instead of the ENVIRONMENT variable when deciding whether to include `eval_context`.

- [ ] **Step 3: Update pipeline_client.py to pass eval flag**

Read `apps/evals/shared/pipeline_client.py` and update the session start POST request to include `?eval=true` in the URL.

- [ ] **Step 4: Test with wrangler dev**

```bash
cd apps/api && npx wrangler dev &
cd apps/evals && uv run python -c "
from shared.pipeline_client import PracticeEvalClient
import asyncio
async def test():
    client = PracticeEvalClient()
    await client.authenticate()
    print('Auth OK')
asyncio.run(test())
"
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/practice/session_finalization.rs apps/evals/shared/pipeline_client.py
git commit -m "feat: eval_context export via query param flag (decoupled from dev environment)"
```

---

### Task 9: Fix Hardcoded Student ID

**Files:**
- Modify: `apps/evals/pipeline/practice_eval/eval_practice.py`

- [ ] **Step 1: Fix the hardcoded student ID**

Read `eval_practice.py` and find the hardcoded `"eval-student-001"`. Replace with a unique ID per recording to prevent baseline contamination:

```python
# Replace hardcoded "eval-student-001" with per-recording ID
student_id = f"eval-{recording['video_id'][:12]}"
```

- [ ] **Step 2: Commit**

```bash
git add apps/evals/pipeline/practice_eval/eval_practice.py
git commit -m "fix: unique student ID per recording in eval runner (prevents baseline contamination)"
```

---

### Task 10: Complete Inference Cache

**Prerequisite:** AMT server crashed at ~2/269 recordings cached. Must be resolved before Phase 2.

- [ ] **Step 1: Diagnose AMT server crash**

Check `apps/inference/amt_local_server.py` or the HF endpoint status. Restart the server and verify it responds.

- [ ] **Step 2: Run cache generation for remaining T5 recordings**

```bash
cd apps/evals && uv run python -m inference.eval_runner --auto-t5
```

Monitor progress. Target: all 361 T5 recordings have both MuQ + AMT cache entries in `model/data/evals/inference_cache/`.

- [ ] **Step 3: Verify cache completeness**

```bash
ls model/data/evals/inference_cache/a1-max-muq-lora*/ | wc -l
```

Expected: 361 files (one per recording).

- [ ] **Step 4: Commit cache manifest**

```bash
git add model/data/evals/inference_cache/
git commit -m "feat: complete inference cache for all 361 T5 recordings"
```

---

### Task 11: Smoke Test (10 Recordings)

- [ ] **Step 1: Start wrangler dev**

```bash
cd apps/api && npx wrangler dev
```

- [ ] **Step 2: Run smoke test**

```bash
cd apps/evals && uv run python -m pipeline.practice_eval.eval_practice \
  --scenarios pipeline/practice_eval/scenarios/t5_bach_invention_1.yaml \
  --limit 10 \
  --pass-a-only
```

Expected: 10 recordings run through pipeline. Each produces synthesis output + accumulator state. Fix any plumbing errors found.

- [ ] **Step 3: Verify accumulator state captured**

Check that the output JSON includes eval_context with teaching_moments, mode_transitions, and baselines for each recording.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: eval plumbing fixes from smoke test (10 recordings validated)"
```

---

## CONVERGENCE: Phase 1 (Pipeline Alignment)

These tasks run after Track A playbook is complete AND Track B plumbing is validated.

### Task 12: Revise Synthesis Prompt

**Files:**
- Modify: `apps/api/src/services/prompts.rs`

- [ ] **Step 1: Read current synthesis prompt**

Read `apps/api/src/services/prompts.rs` and find `SESSION_SYNTHESIS_SYSTEM`.

- [ ] **Step 2: Revise prompt based on playbook**

Update `SESSION_SYNTHESIS_SYSTEM` to incorporate playbook-derived teaching strategies. The prompt should:
- Accept a `skill_tier` parameter (passed from T5 metadata in eval, from runtime detection in production)
- Include piece-style dimension rules from the playbook
- Encode teaching posture appropriate for each tier
- Include good/bad examples from the playbook as few-shot calibration

The exact prompt content depends on what the playbook produces. Write the prompt revision after reviewing `data/playbook.yaml`.

- [ ] **Step 3: Add regression test**

Write a test that verifies `build_synthesis_prompt()` still works correctly when:
- `skill_tier` is None (fallback to current generic behavior)
- `piece_context` is None (no piece-style rules applied)

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/services/prompts.rs
git commit -m "feat: tier-aware synthesis prompt (aligned to Teaching Playbook)"
```

---

### Task 13: Piece-Style Dimension Weights in top_moments()

**Files:**
- Modify: `apps/api/src/practice/accumulator.rs`

- [ ] **Step 1: Read current top_moments() implementation**

Read `apps/api/src/practice/accumulator.rs` lines 88-147.

- [ ] **Step 2: Add piece-style dimension weighting**

Add an optional `dimension_weights: Option<HashMap<String, f64>>` parameter to `top_moments()`. When provided, multiply each moment's `|deviation|` by the dimension's weight before ranking. This allows piece-style rules (e.g., pedaling weight=2.0 for Chopin, weight=0.5 for Bach) to influence which moments are selected.

- [ ] **Step 3: Add regression test**

Write a test that verifies `top_moments()` produces identical results when `dimension_weights` is None (existing behavior preserved).

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/practice/accumulator.rs
git commit -m "feat: piece-style dimension weights in top_moments() (optional, no-op when absent)"
```

---

### Task 14: Failure Attribution in Analysis Script

**Files:**
- Modify: `apps/evals/pipeline/practice_eval/analyze_e2e.py`

- [ ] **Step 1: Add failure attribution function**

```python
def attribute_failures(results: list[dict]) -> list[dict]:
    """For each synthesis that scored poorly, trace the failure to a pipeline capability.

    Rules-based attribution:
    - Synthesis references wrong bars -> score_following
    - Synthesis focuses on irrelevant dimension for piece style -> teaching_moment_selection
    - Synthesis tone inappropriate for skill level -> tier_detection
    - Synthesis is generic/undifferentiated -> synthesis_prompt
    - Synthesis contradicts model scores -> model_accuracy
    """
    attributions = []
    for r in results:
        synthesis = r.get("synthesis", {})
        judge_scores = r.get("judge_scores", {})
        accumulator = r.get("eval_context", {})

        # Only attribute failures (judge scored below threshold)
        failed_criteria = [
            c for c in judge_scores.get("scores", [])
            if c.get("score", 3) == 0  # Critical failures only
        ]

        for criterion in failed_criteria:
            attribution = {
                "recording_id": r.get("recording_id"),
                "criterion": criterion.get("name"),
                "score": criterion.get("score"),
                "evidence": criterion.get("evidence", ""),
                "attributed_to": "unknown",
                "reasoning": "",
            }

            evidence = criterion.get("evidence", "").lower()

            if "bar" in evidence and ("wrong" in evidence or "incorrect" in evidence):
                attribution["attributed_to"] = "score_following"
                attribution["reasoning"] = "Synthesis references incorrect bar numbers"
            elif "dimension" in evidence and ("irrelevant" in evidence or "inappropriate" in evidence):
                attribution["attributed_to"] = "teaching_moment_selection"
                attribution["reasoning"] = "Selected dimension not relevant for this piece style"
            elif "skill" in evidence or "level" in evidence or "tone" in evidence:
                attribution["attributed_to"] = "tier_detection"
                attribution["reasoning"] = "Teaching posture mismatched to student skill level"
            elif "generic" in evidence or "any" in evidence or "differentiat" in evidence:
                attribution["attributed_to"] = "synthesis_prompt"
                attribution["reasoning"] = "Output not grounded in specific performance data"
            else:
                attribution["attributed_to"] = "unattributed"
                attribution["reasoning"] = "Could not determine root capability from evidence"

            attributions.append(attribution)

    return attributions
```

- [ ] **Step 2: Integrate into analysis output**

Add a `print_failure_attribution()` function that calls `attribute_failures()` and prints a summary table:

```
FAILURE ATTRIBUTION
═══════════════════════════════════════
Capability               | Failures | %
─────────────────────────|──────────|────
score_following          |    5     | 25%
teaching_moment_selection|    8     | 40%
synthesis_prompt         |    4     | 20%
tier_detection           |    2     | 10%
unattributed             |    1     |  5%
```

- [ ] **Step 3: Commit**

```bash
git add apps/evals/pipeline/practice_eval/analyze_e2e.py
git commit -m "feat: rules-based failure attribution (traces synthesis failures to pipeline capability)"
```

---

### Task 15: Update Judge Framework for New Rubrics

**Files:**
- Modify: `apps/evals/shared/judge.py`

- [ ] **Step 1: Update judge_synthesis() to use new rubric format**

After rubrics are derived (Task 6), update `judge_synthesis()` in `judge.py` to:
- Load the new judge prompt from `prompts/synthesis_quality_judge_v2.txt`
- Parse scores on the research-derived dimensions (not the old 5 criteria)
- Return scores with the new dimension names and scale

The exact changes depend on the rubric output from Task 6. Read `prompts/rubric_definition.json` for the dimension names and scoring format.

- [ ] **Step 2: Update eval_practice.py to pass skill_bucket to judge**

The judge needs to know the student's skill level (from T5 metadata) to evaluate tier-appropriateness. Pass `skill_bucket` from the scenario YAML to the judge function.

- [ ] **Step 3: Commit**

```bash
git add apps/evals/shared/judge.py apps/evals/pipeline/practice_eval/eval_practice.py
git commit -m "feat: judge framework updated for research-derived rubrics"
```

---

## Phase 2: Eval Execution

### Task 16: Full Eval Run (Pass A)

- [ ] **Step 1: Run all 361 recordings**

```bash
cd apps/evals && uv run python -m pipeline.practice_eval.eval_practice \
  --pass-a-only \
  --judge synthesis \
  --output reports/practice_eval_v2.json
```

Expected: ~3 hours runtime. Produces synthesis + judge scores for all recordings.

- [ ] **Step 2: Run analysis**

```bash
cd apps/evals && uv run python -m pipeline.practice_eval.analyze_e2e \
  --input reports/practice_eval_v2.json
```

Review: composite scores, per-dimension breakdown, failure attribution.

- [ ] **Step 3: Export stratified sample for human review**

```bash
cd apps/evals && uv run python -c "
import json
data = json.loads(open('reports/practice_eval_v2.json').read())
# Sort by composite score, take 10 worst + 10 best + 10 mid
sorted_data = sorted(data['results'], key=lambda x: x.get('composite_score', 0))
sample = sorted_data[:10] + sorted_data[-10:] + sorted_data[len(sorted_data)//2 - 5:len(sorted_data)//2 + 5]
json.dump(sample, open('reports/human_review_sample.json', 'w'), indent=2)
print(f'Exported {len(sample)} recordings for human review')
"
```

**CHECKPOINT: Human review.** Jai listens to 50 recordings and annotates each synthesis output. This is ~8-10 hours. Annotations inform judge calibration and identify systematic failures.

---

### Task 17: Judge Calibration + Pass B

- [ ] **Step 1: Compare human annotations to judge scores**

After Jai completes the 50-output review, compare his annotations against the LLM judge scores. Identify where they diverge. Adjust judge prompts or rubric language as needed.

- [ ] **Step 2: Run Pass B (without piece context)**

```bash
cd apps/evals && uv run python -m pipeline.practice_eval.eval_practice \
  --pass-b-only \
  --judge synthesis \
  --output reports/practice_eval_v2_passb.json
```

- [ ] **Step 3: Compute piece-context delta**

```bash
cd apps/evals && uv run python -m pipeline.practice_eval.analyze_e2e \
  --input reports/practice_eval_v2.json \
  --pass-b reports/practice_eval_v2_passb.json \
  --compare
```

- [ ] **Step 4: Beta readiness assessment**

Review: no critical failures (score 0 on any dimension), composite score meets threshold, clear differentiation from generic LLM on 80%+ of outputs.
