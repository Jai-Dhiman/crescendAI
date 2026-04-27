import styleRules from "../lib/style-rules.json";
import {
	deriveSignals,
	formatTeacherVoiceBlocks,
	selectClusters,
} from "./teacher_style";

type StyleRulesEra = {
	composer_patterns: string[];
	dimensions: Record<string, string>;
};
type StyleRulesFile = {
	eras: Record<string, StyleRulesEra>;
};

function composerToEra(composer: string): string {
	if (!composer) return "Unknown";
	const rules = styleRules as StyleRulesFile;
	const lowered = composer.toLowerCase();
	for (const [eraName, eraData] of Object.entries(rules.eras)) {
		for (const pattern of eraData.composer_patterns) {
			if (lowered.includes(pattern.toLowerCase())) {
				return eraName;
			}
		}
	}
	return "Unknown";
}

function getStyleGuidance(composer: string): string {
	const era = composerToEra(composer);
	if (era === "Unknown") return "";
	const rules = styleRules as StyleRulesFile;
	const dims = rules.eras[era].dimensions;
	const lines = [
		`<style_guidance era="${era}">`,
		`For ${era}-era repertoire, weight dimensions as follows when giving feedback:`,
	];
	for (const [dim, rule] of Object.entries(dims)) {
		lines.push(`- ${dim}: ${rule}`);
	}
	lines.push("Advice that contradicts these rules should not be given.");
	lines.push("</style_guidance>");
	return lines.join("\n");
}

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

export const TEACHER_PROMPT_DYNAMIC_BOUNDARY =
	"__TEACHER_PROMPT_DYNAMIC_BOUNDARY__";

export const UNIFIED_TEACHER_SYSTEM = `You are a warm, encouraging piano teacher. You help students improve their playing through thoughtful conversation, grounded in their actual playing data when available.

## Pedagogy Principles
- Celebrate strengths before suggesting improvements
- Frame observations, not absolute judgments
- Give actionable practice strategies, not abstract criticism
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

## Model Calibration
The MuQ audio model has R2~0.5 and 80% pairwise accuracy. Scores are directional signals, not precise measurements. A deviation of 0.1 from baseline is noise; 0.2+ is meaningful. Use deviations to identify patterns, not to make absolute claims. Never mention raw scores to the student -- translate into musical language.

## Tool Usage
You have tools available. Use them when they add value:
- search_catalog: When the student mentions a piece by name and you need its piece_id for other tools. Never ask the student for a piece ID -- use search_catalog to look it up yourself. If multiple matches are returned, present the options to the student and confirm before using a piece_id. Never invent a piece_id; always pass through the exact pieceId string returned in search_catalog results.
- create_exercise: When a concrete drill would help more than verbal guidance. Use sparingly.
- score_highlight: Whenever the student asks to see, print, look at, or work on specific bars -- this tool renders those bars as notation in the chat. Also use it to point at a passage visually during teaching. Requires piece_id -- use search_catalog first if you don't have it. Never say you cannot show printed music; call this tool instead.
- keyboard_guide: When fingering or hand position matters.
- show_session_data: When the student asks about progress or you want to reference their history.
- reference_browser: When suggesting the student listen to a specific performance.
Most responses should be text-only. Tools are supplements, not defaults.

## Voice
- 1-4 sentences for observations, up to a short paragraph for explanations
- No markdown formatting, no bullets, no emojis
- Conversational, warm, specific
- Speak as a teacher talking TO the student, not about them`;

export function buildSynthesisFraming(
	sessionDurationMs: number,
	practicePattern: unknown,
	topMoments: unknown,
	drillingRecords: unknown,
	pieceMetadata: unknown,
	memoryContext: string,
	composer: string,
): string {
	const parts: string[] = [];

	const sessionData = {
		duration_minutes: Math.round(sessionDurationMs / 60000),
		practice_pattern: practicePattern,
		top_moments: topMoments,
		drilling_records: drillingRecords,
		piece: pieceMetadata,
	};

	parts.push("<session_data>");
	parts.push(JSON.stringify(sessionData, null, 2));
	parts.push("</session_data>");

	const guidance = getStyleGuidance(composer);
	if (guidance.length > 0) {
		parts.push("");
		parts.push(guidance);
	}

	const signals = deriveSignals(
		topMoments,
		drillingRecords,
		sessionDurationMs,
		pieceMetadata,
		practicePattern,
	);
	const voiceBlocks = formatTeacherVoiceBlocks(selectClusters(signals));
	if (voiceBlocks.length > 0) {
		parts.push("");
		parts.push(voiceBlocks);
	}

	if (memoryContext.length > 0) {
		parts.push("");
		parts.push("<student_memory>");
		parts.push(memoryContext);
		parts.push("</student_memory>");
	}

	parts.push("");
	parts.push(
		"<task>Write <analysis>...</analysis> first as a reasoning scratchpad (this will be stripped before delivery). Then write your teacher response: 3-6 sentences, conversational, warm, specific. Use tools if they would add value. Do not mention scores or numbers. Do not list all dimensions -- focus on what matters most for this session.</task>",
	);

	return parts.join("\n");
}

export function buildChatFraming(
	studentLevel: string,
	studentGoals: string,
	memoryContext: string,
): string {
	const parts: string[] = [];

	parts.push("<student>");
	if (studentLevel.length > 0) {
		parts.push(`Level: ${studentLevel}`);
	}
	if (studentGoals.length > 0) {
		parts.push(`Goals: ${studentGoals}`);
	}
	parts.push("</student>");

	if (memoryContext.length > 0) {
		parts.push("");
		parts.push("<student_memory>");
		parts.push(memoryContext);
		parts.push("</student_memory>");
	}

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
