export const SESSION_SYNTHESIS_SYSTEM = `You are a warm, perceptive piano teacher reviewing a practice session. You watched the entire session and now give your student one cohesive, encouraging response.

## What you receive

A JSON object with the full session context: duration, practice pattern (modes and transitions), top teaching moments (dimensions with scores and deviations from baseline), drilling progress, and student memory.

## How to respond

1. Start with what went well -- acknowledge effort and specific improvements.
2. Identify the 1-2 most important things to work on, grounded in the session data.
3. If drilling occurred, comment on the progression (first vs final scores).
4. Frame suggestions as actionable practice strategies, not abstract criticism.
5. Keep it conversational -- 3-6 sentences. You are talking TO the student.
6. Reference specific musical details (bars, sections, dimensions) when the data supports it.
7. Do NOT mention scores, numbers, or model outputs directly. Translate them into musical language.
8. Do NOT list all dimensions. Focus on what matters most for THIS session.

## Calibration

The MuQ audio model has R2~0.5 and 80% pairwise accuracy. Scores are directional signals, not precise measurements. A deviation of 0.1 is noise; 0.2+ is meaningful. Use deviations to identify patterns, not to make absolute claims.`;

export const SUBAGENT_SYSTEM = `You are a piano pedagogy analyst. You receive structured data about a student's practice session -- teaching moments identified by an audio analysis model, the student's history, and musical context.

Your job is to reason about which teaching moment matters most for this student right now and decide how to frame it. You are NOT talking to the student. You are preparing a handoff for a teacher who will deliver the observation.

## Important context about the audio model

The MuQ audio model has R2~0.5 and 80.8% pairwise accuracy (A1-Max 4-fold ensemble). Scores are useful directional signals for reasoning -- they indicate relative strengths and weaknesses -- but they are NOT precise enough to report as grades or ratings. Never treat a score difference of less than ~0.1 as meaningful. Use scores to inform your reasoning, not as evidence to present.

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

\`\`\`json
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
\`\`\`

Then write a narrative paragraph (3-5 sentences) explaining your reasoning for the teacher. Include what you heard, why it matters, and how to frame the observation.`;

export const TEACHER_SYSTEM = `You are a piano teacher who has been listening to your student practice. You have years of experience and deep knowledge of piano pedagogy, repertoire, and technique.

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
"Your dynamics need work and the pedaling could be cleaner. Also watch the tempo in the second section. Overall good effort though!"`;

export interface ExerciseTool {
  name: string;
  description: string;
  input_schema: unknown;
}

export function exerciseToolDefinition(): ExerciseTool {
  return {
    name: "create_exercise",
    description:
      "Create a focused practice exercise when the student would benefit from " +
      "structured practice on a specific passage or technique. Use sparingly -- only when " +
      "a concrete drill would be more helpful than verbal guidance alone. Most observations " +
      "should be text-only.",
    input_schema: {
      type: "object",
      properties: {
        source_passage: {
          type: "string",
          description:
            "The passage this exercise targets (e.g., 'measures 12-16' or 'the opening phrase')",
        },
        target_skill: {
          type: "string",
          description:
            "The specific skill being developed (e.g., 'Voice balancing between hands')",
        },
        exercises: {
          type: "array",
          items: {
            type: "object",
            properties: {
              title: {
                type: "string",
                description: "Short exercise name",
              },
              instruction: {
                type: "string",
                description:
                  "Concrete steps the student should follow. 2-4 sentences.",
              },
              focus_dimension: {
                type: "string",
                enum: [
                  "dynamics",
                  "timing",
                  "pedaling",
                  "articulation",
                  "phrasing",
                  "interpretation",
                ],
              },
              hands: {
                type: "string",
                enum: ["left", "right", "both"],
              },
            },
            required: ["title", "instruction", "focus_dimension"],
          },
          minItems: 1,
          maxItems: 3,
        },
      },
      required: ["source_passage", "target_skill", "exercises"],
    },
  };
}

export interface ObservationRow {
  dimension: string;
  observationText: string;
  framing: string;
  createdAt: string;
}

export function buildSubagentUserPrompt(
  teachingMoment: Record<string, unknown>,
  pieceContext: Record<string, unknown> | null,
  sessionContext: Record<string, unknown>,
  studentContext: Record<string, unknown>,
  memory: string,
  recentObservations: ObservationRow[],
): string {
  const parts: string[] = [];

  // Teaching moment data
  parts.push("<teaching_moment>");
  parts.push(
    `Chunk ${(teachingMoment["chunk_index"] as number | undefined) ?? 0} at ${(teachingMoment["start_offset_sec"] as number | undefined) ?? 0}s into session.`,
  );
  parts.push(
    `Dimension flagged: ${(teachingMoment["dimension"] as string | undefined) ?? "unknown"} (score: ${((teachingMoment["dimension_score"] as number | undefined) ?? 0).toFixed(2)}, stop probability: ${((teachingMoment["stop_probability"] as number | undefined) ?? 0).toFixed(2)})`,
  );

  const allScores = teachingMoment["all_scores"] as
    | Record<string, number>
    | undefined;
  if (allScores) {
    parts.push("All 6 dimension scores for this chunk:");
    for (const dim of [
      "dynamics",
      "timing",
      "pedaling",
      "articulation",
      "phrasing",
      "interpretation",
    ]) {
      if (typeof allScores[dim] === "number") {
        parts.push(`- ${dim}: ${allScores[dim].toFixed(2)}`);
      }
    }
  }
  parts.push("</teaching_moment>");
  parts.push("");

  // Piece context
  if (pieceContext) {
    parts.push("<piece_context>");
    if (typeof pieceContext["composer"] === "string") {
      parts.push(`Composer: ${pieceContext["composer"]}`);
    }
    if (typeof pieceContext["title"] === "string") {
      parts.push(`Title: ${pieceContext["title"]}`);
    }
    if (typeof pieceContext["bar_range"] === "string") {
      parts.push(`Bar range: ${pieceContext["bar_range"]}`);
    }
    if (typeof pieceContext["analysis_tier"] === "number") {
      parts.push(
        `Analysis tier: ${pieceContext["analysis_tier"]} (1=full score context, 2=absolute, 3=scores only)`,
      );
    }

    const musicalAnalysis = pieceContext["musical_analysis"] as
      | Array<Record<string, unknown>>
      | undefined;
    if (Array.isArray(musicalAnalysis) && musicalAnalysis.length > 0) {
      parts.push("");
      parts.push("<musical_analysis>");
      for (const dimAnalysis of musicalAnalysis) {
        const dim = dimAnalysis["dimension"] as string | undefined;
        if (dim) {
          parts.push(`<${dim}>`);
          if (typeof dimAnalysis["analysis"] === "string") {
            parts.push(`  ${dimAnalysis["analysis"]}`);
          }
          if (typeof dimAnalysis["score_marking"] === "string") {
            parts.push(`  Score marking: ${dimAnalysis["score_marking"]}`);
          }
          if (typeof dimAnalysis["reference_comparison"] === "string") {
            parts.push(
              `  Reference: ${dimAnalysis["reference_comparison"]}`,
            );
          }
          parts.push(`</${dim}>`);
        }
      }
      parts.push("</musical_analysis>");
    }
    parts.push("</piece_context>");
    parts.push("");
  }

  // Session context
  parts.push("<session_context>");
  parts.push(
    `Duration: ${(sessionContext["duration_min"] as number | undefined) ?? 0} minutes, ${(sessionContext["total_chunks"] as number | undefined) ?? 0} chunks analyzed, ${(sessionContext["chunks_above_threshold"] as number | undefined) ?? 0} teaching moments found.`,
  );
  parts.push("</session_context>");
  parts.push("");

  // Student context
  parts.push("<student_context>");
  const sessionCount =
    (studentContext["session_count"] as number | undefined) ?? 0;
  if (sessionCount <= 1) {
    parts.push("This is a new student. No history yet.");
    if (typeof studentContext["level"] === "string") {
      parts.push(`Repertoire suggests ${studentContext["level"]} level.`);
    }
  } else {
    if (typeof studentContext["level"] === "string") {
      parts.push(`Level: ${studentContext["level"]}`);
    }
    const goals = studentContext["goals"] as string | undefined;
    if (goals && goals.length > 0) {
      parts.push(`Goals: ${goals}`);
    }
    const baselines = studentContext["baselines"] as
      | Record<string, number>
      | undefined;
    if (baselines) {
      parts.push(`Baselines (over ${sessionCount} sessions):`);
      for (const dim of [
        "dynamics",
        "timing",
        "pedaling",
        "articulation",
        "phrasing",
        "interpretation",
      ]) {
        if (typeof baselines[dim] === "number") {
          parts.push(`- ${dim}: ${baselines[dim].toFixed(2)}`);
        }
      }
    }
  }
  parts.push("</student_context>");
  parts.push("");

  // Memory context
  if (memory.length > 0) {
    parts.push("<memory>");
    parts.push(memory);
    parts.push("</memory>");
    parts.push("");
  }

  // Recent observations
  if (recentObservations.length > 0) {
    parts.push("<recent_observations>");
    for (const obs of recentObservations) {
      parts.push(
        `- [${obs.createdAt}] ${obs.dimension}: "${obs.observationText}" (framing: ${obs.framing})`,
      );
    }
    parts.push("</recent_observations>");
    parts.push("");
  }

  parts.push(
    "<task>\nAnalyze the teaching moment above. Select the best observation to make and decide how to frame it. Output the JSON + narrative as specified.\n</task>",
  );

  return parts.join("\n");
}

export function buildTeacherUserPrompt(
  subagentJson: string,
  subagentNarrative: string,
  studentLevel: string,
  studentGoals: string,
): string {
  const parts: string[] = [];

  parts.push("<analysis>");
  parts.push(subagentJson);
  parts.push("");
  parts.push(subagentNarrative);
  parts.push("</analysis>");
  parts.push("");

  parts.push("<student>");
  parts.push(`Level: ${studentLevel}`);
  if (studentGoals.length > 0) {
    parts.push(`Goals: ${studentGoals}`);
  }
  parts.push("</student>");
  parts.push("");

  parts.push(
    "<task>\nBased on the analysis above, give one observation to the student. Be specific about what you heard and what to try. 1-3 sentences, no formatting.\n</task>",
  );

  return parts.join("\n");
}

export const CHAT_SYSTEM = `You are a warm, encouraging piano teacher. You help students improve their playing through thoughtful conversation. You give specific, actionable advice grounded in the student's actual playing data when available.

Key principles:
- Celebrate strengths before suggesting improvements
- Frame observations, not absolute judgments
- Give actionable practice strategies
- Be specific about musical elements (dynamics, timing, pedaling, articulation, phrasing, interpretation)
- Adapt to the student's level and goals`;

export function buildChatUserContext(student: {
  inferredLevel?: string | null;
  explicitGoals?: string | null;
  baselines?: Record<string, number | null>;
}): string {
  const parts: string[] = [];
  if (student.inferredLevel) {
    parts.push(`Student level: ${student.inferredLevel}`);
  }
  if (student.explicitGoals) {
    parts.push(`Student goals: ${student.explicitGoals}`);
  }
  if (student.baselines) {
    const dims = Object.entries(student.baselines)
      .filter(([, v]) => v != null)
      .map(([k, v]) => `${k}: ${(v as number).toFixed(2)}`)
      .join(", ");
    if (dims) parts.push(`Current baselines: ${dims}`);
  }
  return parts.join("\n");
}

export function buildTitlePrompt(firstMessage: string): string {
  return `Generate a concise title (3-6 words, no quotes) for a piano lesson conversation that starts with: "${firstMessage}"`;
}
