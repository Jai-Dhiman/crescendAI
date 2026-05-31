// Isolated teacher synthesis + artifacts test.
// Run: cd apps/api && bun test-pipeline-teacher.ts
//
// Calls the same Sonnet 4 model production uses, with the same UNIFIED_TEACHER_SYSTEM
// prompt (inlined to avoid prompts.ts JSON-import issues). Two A/B variants:
//   A) MINIMAL briefing — Tier 2 facts (no piece, no bar numbers). What real users get today.
//   B) RICH briefing    — Tier 1 facts (piece + bars 6-11). What they'd get with piece-ID working.
// Also wires real tool schemas to see if the teacher emits artifact tool_use.

import { readFileSync } from "node:fs";

// --- Load ANTHROPIC_API_KEY from .dev.vars ---
const envText = readFileSync(".dev.vars", "utf8");
for (const line of envText.split("\n")) {
	const m = line.match(/^([A-Z_]+)=(.*)$/);
	if (m && !process.env[m[1]]) {
		process.env[m[1]] = m[2].replace(/^"|"$/g, "");
	}
}
const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) throw new Error("ANTHROPIC_API_KEY missing from .dev.vars");

// --- Inlined UNIFIED_TEACHER_SYSTEM (from apps/api/src/services/prompts.ts) ---
const UNIFIED_TEACHER_SYSTEM = `You are a warm, direct piano teacher. You help students improve their playing through honest, grounded feedback based on their actual playing data when available.

## Pedagogy Principles
- Be honest first, warm second -- do not soften or bury corrections with excessive praise
- Acknowledge strengths when they are genuinely present, not as a reflexive opener
- Give concrete, actionable corrections -- name what needs to change and how
- Be specific about musical elements (dynamics, timing, pedaling, articulation, phrasing, interpretation)
- Adapt to the student's level and goals
- One key insight per response is better than a laundry list

## The Six Dimensions
- Dynamics: volume control, crescendo/decrescendo, accents, balance between hands
- Timing: tempo stability, rubato, rhythmic accuracy, pulse
- Pedaling: sustain pedal timing, half-pedaling, clarity vs. blur
- Articulation: staccato, legato, touch quality, note connections
- Phrasing: musical sentences, breathing, shape, direction
- Interpretation: style, character, emotional expression, faithfulness to score

## Tool Usage
You have tools available. Use them when they add value:
- create_exercise: When a concrete drill would help more than verbal guidance. Use sparingly.
- score_highlight: Render specific bars as notation in the chat. Requires piece_id.
- assign_segment_loop: Propose a loopable practice segment for a specific bar range.
Most responses should be text-only. Tools are supplements, not defaults.

## Voice
- 1-4 sentences for observations
- No markdown formatting, no bullets, no emojis
- Direct, warm, specific -- prioritize honesty over encouragement
- Speak as a teacher talking TO the student, not about them`;

// --- Real-ish tool schemas (simplified from teacher.ts) ---
const TOOLS = [
	{
		name: "create_exercise",
		description: "Generate a custom drill exercise targeting one dimension.",
		input_schema: {
			type: "object",
			properties: {
				title: { type: "string" },
				dimension: { type: "string", enum: ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"] },
				instructions: { type: "string" },
			},
			required: ["title", "dimension", "instructions"],
		},
	},
	{
		name: "score_highlight",
		description: "Show specific bars of a piece as notation.",
		input_schema: {
			type: "object",
			properties: {
				piece_id: { type: "string" },
				bar_start: { type: "integer" },
				bar_end: { type: "integer" },
				dimension: { type: "string" },
				annotation: { type: "string" },
			},
			required: ["piece_id", "bar_start", "bar_end"],
		},
	},
	{
		name: "assign_segment_loop",
		description: "Propose a loopable practice segment for a specific bar range.",
		input_schema: {
			type: "object",
			properties: {
				piece_id: { type: "string" },
				bar_start: { type: "integer" },
				bar_end: { type: "integer" },
				dimension: { type: "string" },
				why: { type: "string" },
			},
			required: ["piece_id", "bar_start", "bar_end", "dimension", "why"],
		},
	},
];

// --- Two briefings ---

const BRIEF_MINIMAL = `<session_data>
${JSON.stringify(
	{
		duration_minutes: 3,
		practice_pattern: "regular",
		top_moments: [],
		drilling_records: [],
		piece: null,
	},
	null,
	2,
)}
</session_data>

<chunk_analysis>
Tier 2 (passage-level, no piece identified):
- dynamics: Velocity range 30-70 (mean 52, range 40)
- timing: Inter-onset interval mean 0.228s, std 0.029s (very regular timing)
- pedaling: 20 pedal event(s) detected
- articulation: Mean note duration 0.889s
- phrasing: 109 notes in passage
</chunk_analysis>

<task>Write a short teacher response: 2-4 sentences, conversational, warm, specific. Use tools if they would add value. Do not mention scores or numbers.</task>`;

const BRIEF_RICH = `<session_data>
${JSON.stringify(
	{
		duration_minutes: 3,
		practice_pattern: "drilling",
		top_moments: [
			{
				chunk_index: 4,
				dimension: "articulation",
				bar_range: [6, 11],
				deviation: -0.18,
				is_positive: false,
			},
		],
		drilling_records: [
			{ bar_range: [6, 11], repetition_count: 4, first_score: 0.45, final_score: 0.52 },
		],
		piece: { piece_id: "bach.prelude.bwv_846", composer: "Bach", title: "Prelude in C major, BWV 846" },
	},
	null,
	2,
)}
</session_data>

<chunk_analysis>
Tier 1 (bar-aligned), bars 6-11:
- dynamics: Mean velocity 52/127 (softer than notated). Score mean: 80. Noticeable crescendo through the passage.
- timing: Mean onset deviation 12ms (std 8ms): close to score timing.
- pedaling: 10 pedal events detected, avg duration 1.63s. Score has 0 marking(s) - over-pedaled relative to Bach norms.
- articulation: Mean note duration 0.889s vs score 0.327s (ratio 2.72x): legato (notes held longer than written).
- phrasing: Consistent timing shape across the passage.
</chunk_analysis>

<task>Write a short teacher response: 2-4 sentences, conversational, warm, specific. Use tools if they would add value. Do not mention scores or numbers.</task>`;

async function callTeacher(label: string, brief: string) {
	console.log(`\n${"=".repeat(70)}\n${label}\n${"=".repeat(70)}`);
	const t0 = Date.now();
	const resp = await fetch("https://api.anthropic.com/v1/messages", {
		method: "POST",
		headers: {
			"x-api-key": API_KEY!,
			"anthropic-version": "2023-06-01",
			"content-type": "application/json",
		},
		body: JSON.stringify({
			model: "claude-sonnet-4-20250514",
			max_tokens: 1024,
			system: [{ type: "text", text: UNIFIED_TEACHER_SYSTEM }],
			messages: [{ role: "user", content: brief }],
			tools: TOOLS,
			tool_choice: { type: "auto" },
		}),
	});
	const elapsed = Date.now() - t0;
	if (!resp.ok) {
		console.log(`HTTP ${resp.status}: ${await resp.text()}`);
		return;
	}
	const data = (await resp.json()) as {
		content: Array<{ type: string; text?: string; name?: string; input?: unknown }>;
		stop_reason: string;
		usage: { input_tokens: number; output_tokens: number };
	};
	console.log(`(${elapsed}ms, in=${data.usage.input_tokens} out=${data.usage.output_tokens}, stop=${data.stop_reason})`);
	for (const block of data.content) {
		if (block.type === "text") {
			console.log(`\n[text]\n${block.text}`);
		} else if (block.type === "tool_use") {
			console.log(`\n[tool_use] ${block.name}:\n${JSON.stringify(block.input, null, 2)}`);
		}
	}
}

await callTeacher("A) MINIMAL briefing — Tier 2 (no piece, no bars). What users get TODAY.", BRIEF_MINIMAL);
await callTeacher("B) RICH briefing — Tier 1 (piece + bars 6-11). What they'd get with piece-ID working.", BRIEF_RICH);
