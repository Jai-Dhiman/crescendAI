//! Prompt templates for the two-stage teacher pipeline.
//!
//! Stage 1 (Subagent): Analyzes teaching moments, selects the most important one,
//! decides framing. Outputs structured JSON + narrative reasoning.
//!
//! Stage 2 (Teacher): Converts subagent analysis into a natural 1-3 sentence
//! observation in the teacher persona voice.

/// Subagent system prompt (Stage 1 -- Groq, Llama 70B)
pub const SUBAGENT_SYSTEM: &str = r#"You are a piano pedagogy analyst. You receive structured data about a student's practice session -- teaching moments identified by an audio analysis model, the student's history, and musical context.

Your job is to reason about which teaching moment matters most for this student right now and decide how to frame it. You are NOT talking to the student. You are preparing a handoff for a teacher who will deliver the observation.

Reason through these steps:
1. LEARNING ARC: Where is the student with this piece? (new/mid-learning/polishing) What feedback is appropriate for this phase?
2. DELTA VS HISTORY: Compare scores against baselines and recent observations. Is this a blind spot (usually strong, dipped today)? A known weakness? An improvement?
3. MUSICAL CONTEXT: What does this music demand? Which dimensions matter most for this composer/style?
4. SELECTION: Pick the single highest-leverage moment. What will move the needle most?
5. FRAMING: Choose one: correction, recognition, encouragement, or question.

Output EXACTLY this JSON followed by a narrative paragraph:

```json
{
    "selected_moment": {
        "chunk_index": <int>,
        "dimension": "<string>",
        "dimension_score": <float>,
        "student_baseline": <float>,
        "bar_range": "<string or null>",
        "section_label": "<string or null>"
    },
    "framing": "<correction|recognition|encouragement|question>",
    "learning_arc": "<new|mid-learning|polishing>",
    "is_positive": <bool>,
    "musical_context": "<one sentence about what this music demands>"
}
```

Then write a narrative paragraph (3-5 sentences) explaining your reasoning for the teacher. Include what you heard, why it matters, and how to frame the observation."#;

/// Teacher system prompt (Stage 2 -- Anthropic, Sonnet 4.6)
pub const TEACHER_SYSTEM: &str = r#"You are a piano teacher who has been listening to your student practice. You have years of experience and deep knowledge of piano pedagogy, repertoire, and technique.

Your role is to give ONE specific observation about what you just heard. Not a report. Not a lesson plan. One thing -- the thing the student most needs to hear right now.

How you speak:
- Specific and grounded: reference the exact musical moment, not generalities
- Natural and warm: you're talking to a student you know, not writing a review
- Actionable: if you point out a problem, suggest what to try
- Honest but encouraging: don't sugarcoat, but don't discourage
- Brief: 1-3 sentences. A teacher's aside, not a lecture.

What you DON'T do:
- List multiple issues (pick ONE)
- Give scores or ratings
- Use jargon without explanation
- Say "great job!" without substance
- Cite sources or references
- Use bullet points or structured formatting
- Use markdown formatting of any kind"#;

/// Build the subagent user prompt from request data, observation history, and memory context.
pub fn build_subagent_user_prompt(
    teaching_moment: &serde_json::Value,
    student: &serde_json::Value,
    session: &serde_json::Value,
    piece_context: &Option<serde_json::Value>,
    recent_observations: &[ObservationRow],
    memory_context: &str,
) -> String {
    let mut prompt = String::with_capacity(2000);

    // Teaching moment data
    prompt.push_str("## Teaching Moment\n\n");
    prompt.push_str(&format!(
        "Chunk {} at {}s into session.\n",
        teaching_moment.get("chunk_index").and_then(|v| v.as_i64()).unwrap_or(0),
        teaching_moment.get("start_offset_sec").and_then(|v| v.as_f64()).unwrap_or(0.0),
    ));
    prompt.push_str(&format!(
        "Dimension flagged: {} (score: {:.2}, stop probability: {:.2})\n\n",
        teaching_moment.get("dimension").and_then(|v| v.as_str()).unwrap_or("unknown"),
        teaching_moment.get("dimension_score").and_then(|v| v.as_f64()).unwrap_or(0.0),
        teaching_moment.get("stop_probability").and_then(|v| v.as_f64()).unwrap_or(0.0),
    ));

    if let Some(scores) = teaching_moment.get("all_scores") {
        prompt.push_str("All 6 dimension scores for this chunk:\n");
        for dim in &["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"] {
            if let Some(score) = scores.get(*dim).and_then(|v| v.as_f64()) {
                prompt.push_str(&format!("- {}: {:.2}\n", dim, score));
            }
        }
        prompt.push('\n');
    }

    // Piece context
    if let Some(piece) = piece_context {
        prompt.push_str("## Piece Context\n\n");
        if let Some(composer) = piece.get("composer").and_then(|v| v.as_str()) {
            prompt.push_str(&format!("Composer: {}\n", composer));
        }
        if let Some(title) = piece.get("title").and_then(|v| v.as_str()) {
            prompt.push_str(&format!("Title: {}\n", title));
        }
        if let Some(section) = piece.get("section").and_then(|v| v.as_str()) {
            prompt.push_str(&format!("Section: {}\n", section));
        }
        if let Some(bar_range) = piece.get("bar_range").and_then(|v| v.as_str()) {
            prompt.push_str(&format!("Bar range: {}\n", bar_range));
        }
        prompt.push('\n');
    }

    // Session context
    prompt.push_str("## Session Context\n\n");
    prompt.push_str(&format!(
        "Duration: {} minutes, {} chunks analyzed, {} teaching moments found.\n\n",
        session.get("duration_min").and_then(|v| v.as_i64()).unwrap_or(0),
        session.get("total_chunks").and_then(|v| v.as_i64()).unwrap_or(0),
        session.get("chunks_above_threshold").and_then(|v| v.as_i64()).unwrap_or(0),
    ));

    // Student context
    prompt.push_str("## Student Context\n\n");
    let session_count = student.get("session_count").and_then(|v| v.as_i64()).unwrap_or(0);
    if session_count <= 1 {
        prompt.push_str("This is a new student. No history yet.\n");
        if let Some(level) = student.get("level").and_then(|v| v.as_str()) {
            prompt.push_str(&format!("Repertoire suggests {} level.\n", level));
        }
    } else {
        if let Some(level) = student.get("level").and_then(|v| v.as_str()) {
            prompt.push_str(&format!("Level: {}\n", level));
        }
        if let Some(goals) = student.get("goals").and_then(|v| v.as_str()) {
            if !goals.is_empty() {
                prompt.push_str(&format!("Goals: {}\n", goals));
            }
        }
        if let Some(baselines) = student.get("baselines") {
            prompt.push_str(&format!("Baselines (over {} sessions):\n", session_count));
            for dim in &["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"] {
                if let Some(val) = baselines.get(*dim).and_then(|v| v.as_f64()) {
                    prompt.push_str(&format!("- {}: {:.2}\n", dim, val));
                }
            }
        }
    }
    prompt.push('\n');

    // Memory context (synthesized facts, engagement history, piece-specific)
    if !memory_context.is_empty() {
        prompt.push_str(memory_context);
    }

    // Recent observation history
    if !recent_observations.is_empty() {
        prompt.push_str("## Recent Observations (newest first)\n\n");
        for obs in recent_observations {
            prompt.push_str(&format!(
                "- [{}] {}: \"{}\" (framing: {})\n",
                obs.created_at, obs.dimension, obs.observation_text, obs.framing
            ));
        }
        prompt.push('\n');
    }

    prompt.push_str("## Task\n\n");
    prompt.push_str("Analyze the teaching moment above. Select the best observation to make and decide how to frame it. Output the JSON + narrative as specified.");

    prompt
}

/// Build the teacher user prompt from the subagent's analysis.
pub fn build_teacher_user_prompt(
    subagent_json: &str,
    subagent_narrative: &str,
    student_level: &str,
    student_goals: &str,
) -> String {
    let mut prompt = String::with_capacity(1000);

    prompt.push_str("## Analysis from my teaching assistant\n\n");
    prompt.push_str(subagent_json);
    prompt.push_str("\n\n");
    prompt.push_str(subagent_narrative);
    prompt.push_str("\n\n");

    prompt.push_str("## Student\n\n");
    prompt.push_str(&format!("Level: {}\n", student_level));
    if !student_goals.is_empty() {
        prompt.push_str(&format!("Goals: {}\n", student_goals));
    }
    prompt.push_str("\n## What to say\n\n");
    prompt.push_str("Based on the analysis above, give one observation to the student. Be specific about what you heard and what to try. 1-3 sentences, no formatting.");

    prompt
}

/// Build the elaboration prompt for "Tell me more" follow-up.
pub fn build_elaboration_prompt(
    original_observation: &str,
    reasoning_trace: &str,
) -> String {
    format!(
        r#"The student just read this observation and tapped "Tell me more":

"{}"

Your earlier analysis:
{}

Elaborate with:
1. Why this matters for this piece/style
2. A specific practice technique they can try right now
3. What "fixed" would sound/feel like

Still conversational. 2-4 sentences. No formatting."#,
        original_observation, reasoning_trace
    )
}

/// A row from the observations table used to build context.
pub struct ObservationRow {
    pub dimension: String,
    pub observation_text: String,
    pub framing: String,
    pub created_at: String,
}

/// Synthesis system prompt (Groq, Llama 70B)
/// Called after session sync to update synthesized facts.
pub const SYNTHESIS_SYSTEM: &str = r#"You are a memory consolidation system for a piano teaching app. You receive:
1. Current active facts about a student (what the system currently believes)
2. New observations since the last synthesis (what was recently observed)
3. Teaching approach records (what feedback was given and whether the student engaged)
4. Student baselines (current dimension scores)

Your job is to update the student's fact base. Output ONLY valid JSON with three arrays:

```json
{
  "new_facts": [
    {
      "fact_text": "One sentence describing the pattern or insight",
      "fact_type": "dimension|approach|arc|student_reported",
      "dimension": "dynamics|timing|pedaling|articulation|phrasing|interpretation|null",
      "piece_context": {"composer": "...", "title": "..."} or null,
      "trend": "improving|stable|declining|new|resolved",
      "confidence": "high|medium|low",
      "evidence": ["obs_id_1", "obs_id_2"]
    }
  ],
  "invalidated_facts": [
    {
      "fact_id": "id of fact to invalidate",
      "reason": "Why this fact is no longer true",
      "invalid_at": "ISO date when it stopped being true"
    }
  ],
  "unchanged_facts": ["fact_id_1", "fact_id_2"]
}
```

Rules:
- Every current active fact must appear in either invalidated_facts or unchanged_facts
- Create approach facts when engagement patterns are clear (e.g., "student engages most with correction-framed feedback")
- Invalidate facts that are contradicted by new evidence (e.g., a "persistent weakness" that has improved for 3+ sessions)
- Set trend to "resolved" when a previously flagged issue is no longer appearing
- Be conservative: only create high-confidence facts when supported by 3+ observations
- Review student_reported facts for staleness (goals older than 90 days with no related observations)"#;

/// Build the synthesis user prompt from student data.
pub fn build_synthesis_prompt(
    active_facts: &[super::memory::SynthesizedFact],
    new_observations: &[serde_json::Value],
    teaching_approaches: &[serde_json::Value],
    baselines: &serde_json::Value,
) -> String {
    let mut prompt = String::with_capacity(3000);

    prompt.push_str("## Current Active Facts\n\n");
    if active_facts.is_empty() {
        prompt.push_str("No facts yet (first synthesis).\n\n");
    } else {
        for fact in active_facts {
            let dim = fact.dimension.as_deref().unwrap_or("general");
            let trend = fact.trend.as_deref().unwrap_or("unknown");
            prompt.push_str(&format!(
                "- [id: {}, type: {}, dim: {}, trend: {}, confidence: {}, since: {}] {}\n",
                fact.id, fact.fact_type, dim, trend, fact.confidence, fact.valid_at, fact.fact_text
            ));
        }
        prompt.push('\n');
    }

    prompt.push_str("## New Observations Since Last Synthesis\n\n");
    if new_observations.is_empty() {
        prompt.push_str("No new observations.\n\n");
    } else {
        for obs in new_observations {
            let id = obs.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let dim = obs.get("dimension").and_then(|v| v.as_str()).unwrap_or("");
            let text = obs.get("observation_text").and_then(|v| v.as_str()).unwrap_or("");
            let framing = obs.get("framing").and_then(|v| v.as_str()).unwrap_or("");
            let score = obs.get("dimension_score").and_then(|v| v.as_f64());
            let baseline = obs.get("student_baseline").and_then(|v| v.as_f64());
            let created = obs.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
            let trace = obs.get("reasoning_trace").and_then(|v| v.as_str()).unwrap_or("");

            prompt.push_str(&format!("- [id: {}, dim: {}, framing: {}, date: {}]\n", id, dim, framing, created));
            prompt.push_str(&format!("  Text: \"{}\"\n", text));
            if let (Some(s), Some(b)) = (score, baseline) {
                prompt.push_str(&format!("  Score: {:.2} (baseline: {:.2}, delta: {:+.2})\n", s, b, s - b));
            }
            if !trace.is_empty() && trace != "{}" {
                prompt.push_str(&format!("  Reasoning: {}\n", trace));
            }
        }
        prompt.push('\n');
    }

    if !teaching_approaches.is_empty() {
        prompt.push_str("## Teaching Approaches\n\n");
        for ta in teaching_approaches {
            let dim = ta.get("dimension").and_then(|v| v.as_str()).unwrap_or("");
            let framing = ta.get("framing").and_then(|v| v.as_str()).unwrap_or("");
            let summary = ta.get("approach_summary").and_then(|v| v.as_str()).unwrap_or("");
            let engaged = ta.get("engaged").and_then(|v| v.as_i64()).unwrap_or(0) == 1;
            prompt.push_str(&format!(
                "- {}: {} (engaged: {})\n",
                dim, summary, if engaged { "yes" } else { "no" }
            ));
        }
        prompt.push('\n');
    }

    prompt.push_str("## Student Baselines\n\n");
    for dim in &["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"] {
        if let Some(val) = baselines.get(format!("baseline_{}", dim)).and_then(|v| v.as_f64()) {
            prompt.push_str(&format!("- {}: {:.2}\n", dim, val));
        }
    }
    prompt.push('\n');

    prompt.push_str("## Task\n\nAnalyze the new observations against current facts and baselines. Output the JSON update.");

    prompt
}
