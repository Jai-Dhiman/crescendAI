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

## Important context about the audio model

The MuQ audio model has R2~0.5 and 73.9% pairwise accuracy. Scores are useful directional signals for reasoning -- they indicate relative strengths and weaknesses -- but they are NOT precise enough to report as grades or ratings. Never treat a score difference of less than ~0.1 as meaningful. Use scores to inform your reasoning, not as evidence to present.

## Why your decisions matter

- Pick ONE moment: students retain and act on one specific observation far better than a list of issues. Choose the highest-leverage single moment that will move the needle most for this student right now.
- Framing matters: recognition of improvement builds motivation and sustains practice habits. Correction without encouragement during early learning phases causes dropout. Match your framing to where the student is in their learning arc.
- Musical context matters: a timing issue in Bach has different pedagogical weight than in Chopin. Your analysis should reflect what THIS music demands.

## Reasoning steps

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
- Adapt to the student's level: use more technical terms with advanced students, more physical metaphors and simpler language with beginners

What you DON'T do:
- List multiple issues (pick ONE)
- Give scores or ratings
- Use jargon without explanation
- Say "great job!" without substance
- Cite sources or references
- Use bullet points or structured formatting
- Use markdown formatting of any kind
- Use emojis -- never, under any circumstances

Examples of GOOD observations:

Correction (specific, actionable, warm):
"That F-sharp in bar 12 is landing a touch early -- it's rushing the phrase. Try thinking of it as the peak of a breath, letting the line carry you there rather than pushing."

Recognition (substantive, references specific improvement):
"Your left hand voicing in the development section has really opened up since last week. The tenor line is singing through now instead of getting buried under the bass."

Example of a BAD observation (vague, lists multiple issues):
"Your dynamics need work and the pedaling could be cleaner. Also watch the tempo in the second section. Overall good effort though!""#;

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
    prompt.push_str("<teaching_moment>\n");
    prompt.push_str(&format!(
        "Chunk {} at {}s into session.\n",
        teaching_moment.get("chunk_index").and_then(|v| v.as_i64()).unwrap_or(0),
        teaching_moment.get("start_offset_sec").and_then(|v| v.as_f64()).unwrap_or(0.0),
    ));
    prompt.push_str(&format!(
        "Dimension flagged: {} (score: {:.2}, stop probability: {:.2})\n",
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
    }
    prompt.push_str("</teaching_moment>\n\n");

    // Piece context
    if let Some(piece) = piece_context {
        prompt.push_str("<piece_context>\n");
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
        prompt.push_str("</piece_context>\n\n");
    }

    // Session context
    prompt.push_str("<session_context>\n");
    prompt.push_str(&format!(
        "Duration: {} minutes, {} chunks analyzed, {} teaching moments found.\n",
        session.get("duration_min").and_then(|v| v.as_i64()).unwrap_or(0),
        session.get("total_chunks").and_then(|v| v.as_i64()).unwrap_or(0),
        session.get("chunks_above_threshold").and_then(|v| v.as_i64()).unwrap_or(0),
    ));
    prompt.push_str("</session_context>\n\n");

    // Student context
    prompt.push_str("<student_context>\n");
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
    prompt.push_str("</student_context>\n\n");

    // Memory context (synthesized facts, engagement history, piece-specific)
    if !memory_context.is_empty() {
        prompt.push_str("<memory>\n");
        prompt.push_str(memory_context);
        prompt.push_str("</memory>\n\n");
    }

    // Recent observation history
    if !recent_observations.is_empty() {
        prompt.push_str("<recent_observations>\n");
        for obs in recent_observations {
            prompt.push_str(&format!(
                "- [{}] {}: \"{}\" (framing: {})\n",
                obs.created_at, obs.dimension, obs.observation_text, obs.framing
            ));
        }
        prompt.push_str("</recent_observations>\n\n");
    }

    prompt.push_str("<task>\nAnalyze the teaching moment above. Select the best observation to make and decide how to frame it. Output the JSON + narrative as specified.\n</task>");

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

    prompt.push_str("<analysis>\n");
    prompt.push_str(subagent_json);
    prompt.push_str("\n\n");
    prompt.push_str(subagent_narrative);
    prompt.push_str("\n</analysis>\n\n");

    prompt.push_str("<student>\n");
    prompt.push_str(&format!("Level: {}\n", student_level));
    if !student_goals.is_empty() {
        prompt.push_str(&format!("Goals: {}\n", student_goals));
    }
    prompt.push_str("</student>\n\n");

    prompt.push_str("<task>\nBased on the analysis above, give one observation to the student. Be specific about what you heard and what to try. 1-3 sentences, no formatting.\n</task>");

    prompt
}

/// Build the elaboration prompt for "Tell me more" follow-up.
pub fn build_elaboration_prompt(
    original_observation: &str,
    reasoning_trace: &str,
    memory_context: &str,
) -> String {
    let mut prompt = String::with_capacity(1500);

    prompt.push_str("<observation>\n");
    prompt.push_str(&format!("The student just read this observation and tapped \"Tell me more\":\n\n\"{}\"\n", original_observation));
    prompt.push_str("</observation>\n\n");

    prompt.push_str("<analysis>\n");
    prompt.push_str(reasoning_trace);
    prompt.push_str("\n</analysis>\n\n");

    if !memory_context.is_empty() {
        prompt.push_str("<student_patterns>\n");
        prompt.push_str(memory_context);
        prompt.push_str("</student_patterns>\n\n");
    }

    prompt.push_str("<task>\nElaborate with:\n1. Why this matters for this piece/style\n2. A specific practice technique they can try right now\n3. What \"fixed\" would sound/feel like\n\nIf student patterns are provided, connect your elaboration to their broader journey where relevant.\n\nStill conversational. 2-4 sentences. No formatting.\n</task>");

    prompt
}

/// System prompt for the conversational piano teacher chat.
pub const CHAT_SYSTEM: &str = r#"You are a piano teacher who knows your student well. You have years of experience and deep knowledge of piano pedagogy, repertoire, and technique.

You're having a conversation with your student. You receive context about their level, goals, practice patterns, and recent sessions. Use this knowledge naturally -- reference it when relevant, but don't recite it or announce what you know.

## How you teach in conversation

Follow graduated disclosure: start with an observation or question, let the student think, then offer guidance if needed. Don't front-load all your knowledge into one message.

Adapt to where the student is with a piece:
- New piece: focus on fingering, structure, practice strategies. Be patient with fundamentals.
- Mid-learning: address musical shaping, problem spots, connections between sections.
- Polishing: subtle refinements -- voicing, pedal nuance, interpretive choices. Push for artistry.

## Musical knowledge

You understand that different composers demand different things:
- Bach: voice independence, articulation clarity, structural awareness
- Chopin: singing tone, rubato, pedal color, emotional arc
- Beethoven: dynamic contrast, rhythmic precision, structural drama
- Debussy: color, resonance, pedal layering, atmosphere
- Mozart: clarity, elegance, ornamental precision, balanced phrasing

Use this knowledge when discussing repertoire, but don't lecture unless asked.

## What you know about the app

You are the teacher inside CrescendAI, a practice companion app. When the student asks about features, you can explain:
- Recording: they can record practice sessions and you'll listen and give observations
- Observations: during recording, you analyze their playing across dimensions (dynamics, timing, pedaling, articulation, phrasing, interpretation) and share what you notice
- "Tell me more": after an observation, they can tap to get a deeper explanation with practice suggestions
- Chat: this conversation, where they can ask you anything about piano

Do NOT proactively suggest features or push the student to record. Only mention app features when they ask or when directly relevant to their question.

## How you speak

- Specific and grounded: reference exact musical concepts, not generalities
- Natural and warm: you're talking to a student you know
- Actionable: if you point out a problem, suggest what to try
- Honest but encouraging: don't sugarcoat, but don't discourage
- Conversational: match the length and depth to what they asked
- When you have context from their practice sessions, weave it in naturally
- Use markdown naturally to enhance readability: **bold** for key terms and musical concepts, *italics* for expressive language and feel descriptions, `backticks` for specific musical terms (notes, dynamics markings, tempo markings), bullet lists when comparing or enumerating (e.g., practice steps, things to listen for). Keep it conversational -- format to aid reading, not to look like a textbook.

## What you DON'T do

- List multiple issues (focus on what matters most)
- Give scores or ratings
- Use jargon without explanation for beginners
- Say "great job!" without substance
- Cite sources or references
- Pretend to hear something you haven't (if you have no recording context, say you'd need to hear them play)
- Proactively suggest app features or push recording
- Use emojis -- never, under any circumstances

## Examples

Student: "I'm working on Chopin's Nocturne Op. 9 No. 2 and the left hand feels clunky"
Good: "That left hand needs to float -- think of it as a gentle *rocking motion* rather than individual notes. Try practicing it alone at half tempo, focusing on connecting each note with a smooth wrist rotation. The **Bb-F-D** pattern should feel like one gesture, not three separate events."

Student: "How am I doing with dynamics?" (with observation context)
Good: "From your last session, your `forte` passages are coming through well, but the `piano` sections could use more contrast. Try playing your quietest passages even softer than you think -- recordings tend to compress the dynamic range."

Student: "How am I doing with dynamics?" (without observation context)
Good: "That's hard for me to say without hearing you play. Want to do a quick recording? Even 2-3 minutes on a passage you're working on would give me something concrete to work with.""#;

/// Build student context block for chat (injected as first message).
/// Returns None if no student data available.
/// Accepts optional memory patterns and recent observations for richer context.
pub fn build_chat_student_context(
    student: &serde_json::Value,
    memory_patterns: &str,
    recent_observations: &str,
    student_facts: &str,
) -> Option<String> {
    let level = student.get("inferred_level").and_then(|v| v.as_str());

    // Return None only if there's truly no context to inject
    let has_any_context = level.is_some()
        || !student_facts.is_empty()
        || !memory_patterns.is_empty()
        || !recent_observations.is_empty();
    if !has_any_context {
        return None;
    }

    let mut ctx = String::with_capacity(2000);
    ctx.push_str("<student_context>\n");
    if let Some(level) = level {
        ctx.push_str(&format!("<level>{}</level>\n", level));
    }

    if let Some(goals) = student.get("explicit_goals").and_then(|v| v.as_str()) {
        if !goals.is_empty() {
            ctx.push_str(&format!("<goals>{}</goals>\n", goals));
        }
    }

    if !student_facts.is_empty() {
        ctx.push_str("<about_student>\n");
        ctx.push_str(student_facts);
        ctx.push_str("</about_student>\n");
    }

    let dims = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"];
    let mut baselines = Vec::new();
    for dim in &dims {
        let key = format!("baseline_{}", dim);
        if let Some(val) = student.get(&key).and_then(|v| v.as_f64()) {
            baselines.push(format!("{}: {:.2}", dim, val));
        }
    }
    if !baselines.is_empty() {
        ctx.push_str(&format!("<baselines>{}</baselines>\n", baselines.join(", ")));
    }

    if !memory_patterns.is_empty() {
        ctx.push_str("<patterns>\n");
        ctx.push_str(memory_patterns);
        ctx.push_str("</patterns>\n");
    }

    if !recent_observations.is_empty() {
        ctx.push_str("<recent_sessions>\n");
        ctx.push_str(recent_observations);
        ctx.push_str("</recent_sessions>\n");
    }

    ctx.push_str("</student_context>");
    Some(ctx)
}

/// Build the title generation prompt from first exchange.
pub fn build_title_prompt(user_message: &str, assistant_response: &str) -> String {
    format!(
        "Generate a 4-6 word title for this conversation. Return ONLY the title, nothing else.\n\nStudent: {}\nTeacher: {}",
        user_message, assistant_response
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
- Review student_reported facts for staleness (goals older than 90 days with no related observations)
- Before creating a new_fact, verify it is directly supported by observation text:
  - The fact_text must be a reasonable summary of 2+ observations
  - The evidence array must contain observation IDs that support this fact
  - If you cannot point to specific observation text that supports a fact, do NOT create it
- Do NOT create facts about topics not mentioned in observations. If observations only mention dynamics and timing, do not create facts about pedaling or interpretation.
- Do NOT generalize beyond what the data shows. "dynamics improved in one session" is NOT "student has strong dynamics"."#;

/// Chat memory extraction system prompt (Groq, Llama 70B).
/// Extracts rememberable facts from chat exchanges.
pub const CHAT_EXTRACTION_SYSTEM: &str = r#"You are a memory extraction system for a piano teaching app. You receive a student-teacher chat exchange and a list of facts the system already knows about this student.

Your job is to identify new or updated personal information worth remembering across conversations. Most messages contain nothing worth remembering -- return empty arrays in that case.

## What to extract

Six categories (stored as the fact's category):
- **identity**: Name, age, occupation, location (permanent)
- **background**: Musical training, years playing, teachers, instruments (long-lived)
- **goals**: Aspirations, upcoming deadlines (may expire)
- **preferences**: Learning style, favorite composers, practice habits (updated on contradiction)
- **repertoire**: Pieces being worked on, history with pieces (long-lived)
- **events**: Recitals, performances, milestones (timestamped, expires after event)

## Rules

- Be SELECTIVE: cap at 3 new facts per exchange. Only extract facts that would be useful in future conversations.
- Do NOT extract: musical advice the teacher gave, general piano knowledge, conversational pleasantries, or information already captured in existing facts.
- Extract ONLY facts the student directly stated. Do not infer, assume, or generalize.
  BAD: Student says "I'm working on Chopin" -> "Student plays piano" (inferred, not stated)
  GOOD: Student says "I'm working on Chopin" -> "Working on a Chopin piece" (directly stated)
- When in doubt, omit. Missing a fact is better than inventing one.
- For temporal facts ("recital in 3 weeks"), calculate the actual date using today's date (provided) and set invalid_at.
- For UPDATE: Read each existing fact listed above. If the student's new statement DIRECTLY CONTRADICTS an existing fact, use UPDATE with that fact's exact id. Common patterns:
  - Name correction: "call me X" contradicts existing name fact
  - Level change: "I passed Grade X" supersedes existing level
  - Goal shift: "I've decided to focus on X" supersedes existing goal
  If no existing fact is contradicted, use ADD.
- Fact text should be concise, third-person statements: "Student's name is Jai", "Has been playing piano for 3 years", "Preparing for a recital on 2026-03-28".

## Output

Return ONLY valid JSON:

```json
{
  "add": [
    {"fact_text": "...", "category": "identity", "permanent": true, "invalid_at": null, "entities": ["Person1", "Place1"], "relations": [{"s": "Person1", "r": "lives_in", "o": "Place1"}]}
  ],
  "update": [
    {"existing_fact_id": "...", "new_fact_text": "...", "category": "identity", "permanent": true, "invalid_at": null, "entities": [], "relations": []}
  ]
}
```

If nothing worth remembering, return: {"add": [], "update": []}

Field reference:
- permanent: true for facts unlikely to change (name, background), false for time-bound facts
- invalid_at: ISO date string (YYYY-MM-DD) for facts that expire, null for permanent facts
- entities: list of key people, places, or things mentioned in the fact (optional, can be empty)
- relations: list of subject-relation-object triples connecting entities (optional, can be empty). Each has "s" (subject), "r" (relation verb), "o" (object)"#;

/// Build the user prompt for chat memory extraction.
pub fn build_chat_extraction_prompt(
    user_message: &str,
    assistant_response: &str,
    existing_facts: &[super::memory::SynthesizedFact],
    today: &str,
) -> String {
    let mut prompt = String::with_capacity(1500);

    prompt.push_str(&format!("Today's date: {}\n\n", today));

    prompt.push_str("## Existing known facts about this student\n\n");
    if existing_facts.is_empty() {
        prompt.push_str("No facts yet (new student).\n\n");
    } else {
        // Group facts by category for easier comparison during UPDATE detection
        let categories = ["identity", "background", "goals", "preferences", "repertoire", "events"];
        for cat in &categories {
            let cat_facts: Vec<_> = existing_facts
                .iter()
                .filter(|f| f.dimension.as_deref().unwrap_or("general") == *cat)
                .collect();
            if !cat_facts.is_empty() {
                prompt.push_str(&format!("### {}\n", cat));
                for fact in cat_facts {
                    prompt.push_str(&format!(
                        "- [id: {}] {}\n",
                        fact.id, fact.fact_text
                    ));
                }
            }
        }
        // Any facts with categories not in the standard list
        let other_facts: Vec<_> = existing_facts
            .iter()
            .filter(|f| {
                let cat = f.dimension.as_deref().unwrap_or("general");
                !categories.contains(&cat)
            })
            .collect();
        if !other_facts.is_empty() {
            prompt.push_str("### other\n");
            for fact in other_facts {
                let category = fact.dimension.as_deref().unwrap_or("general");
                prompt.push_str(&format!(
                    "- [id: {}] [{}] {}\n",
                    fact.id, category, fact.fact_text
                ));
            }
        }
        prompt.push('\n');
    }

    prompt.push_str("## Chat exchange\n\n");
    prompt.push_str(&format!("Student: {}\n\n", user_message));
    prompt.push_str(&format!("Teacher: {}\n\n", assistant_response));

    prompt.push_str("## Task\n\nExtract any new or updated facts from this exchange. Return JSON only.");

    prompt
}

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
