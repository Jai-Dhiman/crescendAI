"""Generate synthetic teaching scenarios to fill gaps in the masterclass data.

The masterclass data skews heavily advanced and lacks beginner/intermediate
content. This module generates structured scenarios using Claude, which
are then human-filtered before entering the benchmark set.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import anthropic

from .records import TeachingRecord, save_records
from .benchmark import _load_api_key

# Scenario templates: each defines a teaching situation to generate
SCENARIO_TEMPLATES = [
    # Beginner dynamics
    {
        "dimension": "dynamics",
        "student_level": "beginner",
        "piece": "Fur Elise",
        "composer": "Beethoven",
        "passage_description": "opening A minor theme",
        "feedback_type": "suggestion",
        "context": "The student plays everything at the same volume -- no contrast between the gentle opening melody and the louder middle section.",
    },
    {
        "dimension": "dynamics",
        "student_level": "intermediate",
        "piece": "Clair de Lune",
        "composer": "Debussy",
        "passage_description": "climactic passage in the middle section",
        "feedback_type": "correction",
        "context": "The student rushes through the crescendo leading to the climax, arriving at forte too early and losing the gradual build.",
    },
    # Beginner timing
    {
        "dimension": "timing",
        "student_level": "beginner",
        "piece": "Minuet in G Major",
        "composer": "Bach",
        "passage_description": "repeated quarter note passage",
        "feedback_type": "correction",
        "context": "The student's pulse is uneven -- rushing through easy bars and slowing down at difficult ones.",
    },
    {
        "dimension": "timing",
        "student_level": "intermediate",
        "piece": "Nocturne Op. 9 No. 2",
        "composer": "Chopin",
        "passage_description": "rubato in the opening melody",
        "feedback_type": "suggestion",
        "context": "The student plays with strict metronomic time. The melody needs rubato but they're afraid to bend the tempo.",
    },
    # Beginner pedaling
    {
        "dimension": "pedaling",
        "student_level": "beginner",
        "piece": "Moonlight Sonata, 1st movement",
        "composer": "Beethoven",
        "passage_description": "opening arpeggiated triplets",
        "feedback_type": "correction",
        "context": "The student holds the sustain pedal down for entire phrases, creating harmonic blur when the bass changes.",
    },
    {
        "dimension": "pedaling",
        "student_level": "intermediate",
        "piece": "Ballade No. 1",
        "composer": "Chopin",
        "passage_description": "waltz-like second theme",
        "feedback_type": "suggestion",
        "context": "The pedaling is technically correct but mechanical -- changing on every beat rather than following the musical phrase.",
    },
    # Beginner articulation
    {
        "dimension": "articulation",
        "student_level": "beginner",
        "piece": "Sonatina in C Major",
        "composer": "Clementi",
        "passage_description": "staccato passage in the exposition",
        "feedback_type": "correction",
        "context": "The student's staccato sounds heavy and clunky -- more like dropping the keys than releasing them.",
    },
    {
        "dimension": "articulation",
        "student_level": "intermediate",
        "piece": "Invention No. 1 in C Major",
        "composer": "Bach",
        "passage_description": "sixteenth note runs",
        "feedback_type": "suggestion",
        "context": "All notes are played with identical touch. Bach's counterpoint needs varying articulation to distinguish the voices.",
    },
    # Beginner phrasing
    {
        "dimension": "phrasing",
        "student_level": "beginner",
        "piece": "Prelude in C Major, WTC I",
        "composer": "Bach",
        "passage_description": "arpeggiated chord pattern",
        "feedback_type": "suggestion",
        "context": "The student plays each bar as an isolated unit. There's no sense of larger phrase direction across multiple bars.",
    },
    {
        "dimension": "phrasing",
        "student_level": "intermediate",
        "piece": "Pathetique Sonata, 2nd movement",
        "composer": "Beethoven",
        "passage_description": "opening Adagio cantabile melody",
        "feedback_type": "suggestion",
        "context": "The melody is technically smooth but flat -- no sense of where the phrase is going, no arrival point, no breathing between phrases.",
    },
    # Beginner interpretation
    {
        "dimension": "interpretation",
        "student_level": "beginner",
        "piece": "Waltz in A minor",
        "composer": "Chopin",
        "passage_description": "melancholic opening theme",
        "feedback_type": "encouragement",
        "context": "The student is playing the notes correctly but hasn't thought about what the music expresses. It sounds like an exercise.",
    },
    {
        "dimension": "interpretation",
        "student_level": "intermediate",
        "piece": "Prelude Op. 28 No. 4",
        "composer": "Chopin",
        "passage_description": "descending chromatic inner voice",
        "feedback_type": "correction",
        "context": "The right hand melody is well-voiced but the left hand chromatic descent -- which drives the emotional tension -- is barely audible.",
    },
    # Advanced scenarios to diversify beyond masterclass
    {
        "dimension": "dynamics",
        "student_level": "advanced",
        "piece": "Ravel Piano Concerto in G, 2nd movement",
        "composer": "Ravel",
        "passage_description": "opening solo melody",
        "feedback_type": "suggestion",
        "context": "The student's pianissimo is beautiful but static. Ravel's long melodic line needs micro-dynamic shaping within the pp.",
    },
    {
        "dimension": "pedaling",
        "student_level": "advanced",
        "piece": "Images, Book 1: Reflets dans l'eau",
        "composer": "Debussy",
        "passage_description": "overlapping sonorities in the development",
        "feedback_type": "suggestion",
        "context": "The student is using full pedal changes where half-pedal and flutter techniques would create the water-like shimmer Debussy intended.",
    },
    {
        "dimension": "timing",
        "student_level": "advanced",
        "piece": "Goldberg Variations, Aria",
        "composer": "Bach",
        "passage_description": "ornamented sarabande rhythm",
        "feedback_type": "correction",
        "context": "The ornaments are rhythmically precise but sound academic. The sarabande pulse needs to breathe around the ornaments, not accommodate them mechanically.",
    },
    {
        "dimension": "articulation",
        "student_level": "advanced",
        "piece": "Transcendental Etude No. 10",
        "composer": "Liszt",
        "passage_description": "fortissimo chord sequences",
        "feedback_type": "correction",
        "context": "The chords are powerful but every attack sounds the same. The top voice of each chord should project above the rest.",
    },
    {
        "dimension": "phrasing",
        "student_level": "advanced",
        "piece": "Ballade No. 4",
        "composer": "Chopin",
        "passage_description": "contrapuntal development section",
        "feedback_type": "suggestion",
        "context": "The student manages the complex counterpoint technically but each voice has the same phrasing. The soprano line needs independent shaping from the inner voices.",
    },
    {
        "dimension": "interpretation",
        "student_level": "advanced",
        "piece": "Piano Sonata No. 32, Op. 111, 2nd movement",
        "composer": "Beethoven",
        "passage_description": "Arietta theme and first variation",
        "feedback_type": "suggestion",
        "context": "The Arietta is played beautifully but the transition into the first variation doesn't feel like a natural evolution -- there's a disconnect between the theme's simplicity and the variation's energy.",
    },
    # Encouragement scenarios (underrepresented in masterclass data)
    {
        "dimension": "phrasing",
        "student_level": "intermediate",
        "piece": "Nocturne Op. 48 No. 1",
        "composer": "Chopin",
        "passage_description": "C minor opening section",
        "feedback_type": "encouragement",
        "context": "The student has been struggling with this piece for weeks but today the opening section finally has a real sense of line and direction.",
    },
    {
        "dimension": "dynamics",
        "student_level": "beginner",
        "piece": "Gymnopdie No. 1",
        "composer": "Satie",
        "passage_description": "the entire piece",
        "feedback_type": "encouragement",
        "context": "The student is maintaining a consistent pp throughout with gentle dynamic shaping. For a beginner, this is unusual control.",
    },
]

GENERATION_PROMPT = """You are generating training data for a piano teaching AI. Given a teaching scenario, write what an experienced piano teacher would say to the student.

Requirements:
- 1-3 sentences only
- Specific to the exact musical moment described
- Use natural, embodied language (physical metaphors, tactile descriptions)
- Be actionable -- suggest what to try
- Sound like a real teacher talking to a student, not an AI writing about music
- Match the feedback type: {feedback_type}

Scenario:
Piece: {piece} by {composer}
Passage: {passage_description}
Student level: {student_level}
Dimension: {dimension}
What happened: {context}

Write the teacher's response:"""


def generate_synthetic_records(
    templates: list[dict] | None = None,
    output_path: Path | None = None,
) -> list[TeachingRecord]:
    """Generate synthetic TeachingRecords from templates using Claude.

    Returns records that should be human-reviewed before use.
    """
    if templates is None:
        templates = SCENARIO_TEMPLATES

    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found. Set it in the environment "
            "or in apps/api/.dev.vars"
        )
    client = anthropic.Anthropic(api_key=api_key)
    records: list[TeachingRecord] = []

    for i, tmpl in enumerate(templates):
        prompt = GENERATION_PROMPT.format(**tmpl)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )

        teacher_response = response.content[0].text.strip()
        # Strip quotes if the model wraps the response
        if teacher_response.startswith('"') and teacher_response.endswith('"'):
            teacher_response = teacher_response[1:-1]

        record = TeachingRecord(
            id=str(uuid.uuid4()),
            source="synthetic",
            piece=tmpl.get("piece"),
            composer=tmpl.get("composer"),
            passage_description=tmpl.get("passage_description"),
            dimension=tmpl["dimension"],
            student_level=tmpl["student_level"],
            feedback_type=tmpl["feedback_type"],
            scenario=tmpl["context"],
            teacher_response=teacher_response,
            source_id=f"synthetic_template_{i}",
            has_embodied_language=True,  # will be verified during review
            has_piece_specificity=True,
            is_actionable=True,
            metadata={"template": tmpl, "needs_review": True},
        )

        errors = record.validate()
        if errors:
            print(f"  Validation errors for template {i}: {errors}")
            continue

        records.append(record)
        print(
            f"  [{i + 1}/{len(templates)}] {tmpl['piece']} ({tmpl['dimension']}) "
            f"- {teacher_response[:60]}..."
        )

    if output_path:
        save_records(records, output_path)
        print(f"\nSaved {len(records)} synthetic records to {output_path}")

    return records


if __name__ == "__main__":
    from src.paths import Results

    output_path = Results.root / "teacher_voice" / "synthetic_records.jsonl"

    print(f"Generating {len(SCENARIO_TEMPLATES)} synthetic scenarios...\n")
    records = generate_synthetic_records(output_path=output_path)
    print(f"\nGenerated {len(records)} records. Review before use in benchmark.")
