import { and, asc, desc, eq, sql } from "drizzle-orm";
import { z } from "zod";
import { pieces } from "../db/schema/catalog";
import { observations } from "../db/schema/observations";
import { sessions } from "../db/schema/sessions";
import { DIMS_6 } from "../lib/dims";
import type { ServiceContext } from "../lib/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface InlineComponent {
	type: string;
	config: Record<string, unknown>;
}

export interface ToolResult {
	name: string;
	componentsJson: InlineComponent[];
	isError: boolean;
	errorMessage?: string;
}

interface AnthropicToolSchema {
	name: string;
	description: string;
	input_schema: {
		type: "object";
		properties: Record<string, unknown>;
		required?: string[];
	};
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyZodSchema = z.ZodSchema<any>;

interface ToolDefinition {
	name: string;
	description: string;
	schema: AnyZodSchema;
	anthropicSchema: AnthropicToolSchema;
	concurrencySafe: boolean;
	maxResultChars?: number;
	process: (
		ctx: ServiceContext,
		studentId: string,
		input: unknown,
	) => Promise<InlineComponent[]>;
}

// ---------------------------------------------------------------------------
// Shared dimension enum
// ---------------------------------------------------------------------------

const dimensionEnum = z.enum(DIMS_6 as unknown as [string, ...string[]]);

// ---------------------------------------------------------------------------
// Tool: prescribe_exercise
// ---------------------------------------------------------------------------

const barRangeField = z
	.tuple([z.number().int().positive(), z.number().int().positive()])
	.refine(([start, end]) => start <= end, { message: "bar_range start must be <= end" });

const prescribeExerciseSchema = z.discriminatedUnion("kind", [
	z.object({
		kind: z.literal("own_passage_loop"),
		target_dimension: dimensionEnum,
		bar_range: barRangeField,
		tempo_factor: z.number().min(0.25).max(1.0),
		piece_id: z.string().min(1).nullable(),
	}),
	z.object({
		kind: z.literal("corpus_drill"),
		target_dimension: dimensionEnum,
		bar_range: barRangeField,
		tempo_factor: z.number().min(0.25).max(1.0),
		primitive_id: z.string().nullable().optional().default(null),
		piece_id: z.string().min(1).nullable(),
	}),
]);

async function processPrescribeExercise(
	_ctx: ServiceContext,
	_studentId: string,
	rawInput: unknown,
): Promise<InlineComponent[]> {
	const input = prescribeExerciseSchema.parse(rawInput);

	if (input.kind === "own_passage_loop") {
		const scoreClip =
			input.piece_id !== null
				? { pieceId: input.piece_id, bars: input.bar_range as [number, number] }
				: undefined;

		return [
			{
				type: "exercise_set",
				config: {
					sourcePassage: `bars ${input.bar_range[0]}-${input.bar_range[1]}`,
					targetSkill: `${input.target_dimension} focus`,
					scoreClip,
					exercises: [
						{
							title: `Own passage loop: ${input.target_dimension}`,
							instruction: `Loop bars ${input.bar_range[0]}-${input.bar_range[1]} at ${Math.round(input.tempo_factor * 100)}% tempo. Focus on ${input.target_dimension}.`,
							focusDimension: input.target_dimension,
						},
					],
				},
			},
		];
	}

	// corpus_drill — text stub
	const stubInstruction =
		`${input.target_dimension} drill coming soon` +
		(input.primitive_id ? ` (drill: ${input.primitive_id})` : "") +
		`. Practice bars ${input.bar_range[0]}-${input.bar_range[1]} at ${Math.round(input.tempo_factor * 100)}% tempo focusing on ${input.target_dimension}.`;

	return [
		{
			type: "exercise_set",
			config: {
				sourcePassage: `bars ${input.bar_range[0]}-${input.bar_range[1]}`,
				targetSkill: `${input.target_dimension} focus`,
				exercises: [
					{
						title: `${input.target_dimension} corpus drill`,
						instruction: stubInstruction,
						focusDimension: input.target_dimension,
					},
				],
			},
		},
	];
}

// ---------------------------------------------------------------------------
// Tool: score_highlight
// ---------------------------------------------------------------------------

const pieceSlugSchema = z
	.string()
	.min(1)
	.max(200)
	.regex(/^[a-z0-9._-]+$/, {
		message:
			"piece_id must be a catalog slug like 'chopin.ballades.1' (returned by search_catalog)",
	});

const scoreHighlightSchema = z.object({
	piece_id: pieceSlugSchema,
	highlights: z
		.array(
			z.object({
				bars: z
					.tuple([z.number().int().min(1), z.number().int().min(1)])
					.refine(([start, end]) => start <= end, {
						message: "bars start must be <= end",
					}),
				dimension: dimensionEnum,
				annotation: z.string().max(500).optional(),
			}),
		)
		.min(1)
		.max(5),
});

async function processScoreHighlight(
	ctx: ServiceContext,
	_studentId: string,
	rawInput: unknown,
): Promise<InlineComponent[]> {
	const input = scoreHighlightSchema.parse(rawInput);

	// Validate piece exists in catalog
	const catalogRow = await ctx.db
		.select({ pieceId: pieces.pieceId })
		.from(pieces)
		.where(eq(pieces.pieceId, input.piece_id))
		.limit(1);

	if (catalogRow.length === 0) {
		console.log(
			JSON.stringify({
				level: "warn",
				message: "score_highlight piece_id not found in catalog",
				pieceId: input.piece_id,
			}),
		);
	}

	return [
		{
			type: "score_highlight",
			config: {
				pieceId: input.piece_id,
				highlights: input.highlights.map((h) => ({
					bars: h.bars,
					dimension: h.dimension,
					...(h.annotation !== undefined ? { annotation: h.annotation } : {}),
				})),
			},
		},
	];
}

// ---------------------------------------------------------------------------
// Tool: keyboard_guide
// ---------------------------------------------------------------------------

const keyboardGuideSchema = z.object({
	title: z.string(),
	description: z.string(),
	fingering: z.string().optional(),
	hands: z.enum(["left", "right", "both"]),
});

async function processKeyboardGuide(
	_ctx: ServiceContext,
	_studentId: string,
	rawInput: unknown,
): Promise<InlineComponent[]> {
	const input = keyboardGuideSchema.parse(rawInput);

	const config: Record<string, unknown> = {
		title: input.title,
		description: input.description,
		hands: input.hands,
	};
	if (input.fingering !== undefined) {
		config.fingering = input.fingering;
	}

	return [{ type: "keyboard_guide", config }];
}

// ---------------------------------------------------------------------------
// Tool: show_session_data
// ---------------------------------------------------------------------------

const showSessionDataSchema = z.object({
	query_type: z.enum([
		"dimension_history",
		"recent_sessions",
		"session_detail",
	]),
	dimension: dimensionEnum.optional(),
	session_id: z.string().uuid().optional(),
	limit: z.number().int().min(1).max(50).default(20),
});

const DB_LIMIT_CEILING = 50;

async function processShowSessionData(
	ctx: ServiceContext,
	studentId: string,
	rawInput: unknown,
): Promise<InlineComponent[]> {
	const input = showSessionDataSchema.parse(rawInput);

	const effectiveLimit = Math.min(input.limit, DB_LIMIT_CEILING);

	let data: unknown;

	if (input.query_type === "dimension_history") {
		const whereConditions = input.dimension
			? and(
					eq(observations.studentId, studentId),
					eq(observations.dimension, input.dimension),
				)
			: eq(observations.studentId, studentId);

		const rows = await ctx.db
			.select({
				id: observations.id,
				dimension: observations.dimension,
				dimensionScore: observations.dimensionScore,
				observationText: observations.observationText,
				framing: observations.framing,
				createdAt: observations.createdAt,
				sessionId: observations.sessionId,
			})
			.from(observations)
			.where(whereConditions)
			.orderBy(desc(observations.createdAt))
			.limit(effectiveLimit);

		data = rows;
	} else if (input.query_type === "recent_sessions") {
		const rows = await ctx.db
			.select({
				id: sessions.id,
				startedAt: sessions.startedAt,
				endedAt: sessions.endedAt,
				avgDynamics: sessions.avgDynamics,
				avgTiming: sessions.avgTiming,
				avgPedaling: sessions.avgPedaling,
				avgArticulation: sessions.avgArticulation,
				avgPhrasing: sessions.avgPhrasing,
				avgInterpretation: sessions.avgInterpretation,
			})
			.from(sessions)
			.where(eq(sessions.studentId, studentId))
			.orderBy(desc(sessions.startedAt))
			.limit(effectiveLimit);

		data = rows;
	} else {
		// session_detail
		const whereConditions = input.session_id
			? and(
					eq(sessions.studentId, studentId),
					eq(sessions.id, input.session_id),
				)
			: eq(sessions.studentId, studentId);

		const rows = await ctx.db
			.select({
				id: sessions.id,
				startedAt: sessions.startedAt,
				endedAt: sessions.endedAt,
				avgDynamics: sessions.avgDynamics,
				avgTiming: sessions.avgTiming,
				avgPedaling: sessions.avgPedaling,
				avgArticulation: sessions.avgArticulation,
				avgPhrasing: sessions.avgPhrasing,
				avgInterpretation: sessions.avgInterpretation,
			})
			.from(sessions)
			.where(whereConditions)
			.orderBy(desc(sessions.startedAt))
			.limit(1);

		data = rows[0] ?? null;
	}

	return [
		{
			type: "session_data",
			config: {
				queryType: input.query_type,
				studentId,
				data,
			},
		},
	];
}

// ---------------------------------------------------------------------------
// Tool: play_passage
// ---------------------------------------------------------------------------

const playPassageSchema = z
	.object({
		session_id: z.string().uuid(),
		bars: z
			.tuple([z.number().int().min(1), z.number().int().min(1)])
			.refine(([s, e]) => s <= e, { message: "bars start must be <= end" }),
		focus_bars: z
			.tuple([z.number().int().min(1), z.number().int().min(1)])
			.refine(([s, e]) => s <= e, { message: "focus_bars start must be <= end" })
			.optional(),
		dimension: dimensionEnum,
		annotation: z.string().min(1).max(280),
	})
	.refine(
		(d) =>
			d.focus_bars === undefined ||
			(d.focus_bars[0] >= d.bars[0] && d.focus_bars[1] <= d.bars[1]),
		{ message: "focus_bars must be within bars", path: ["focus_bars"] },
	);

async function processPlayPassage(
	_ctx: ServiceContext,
	_studentId: string,
	rawInput: unknown,
): Promise<InlineComponent[]> {
	const input = playPassageSchema.parse(rawInput);
	const config: Record<string, unknown> = {
		sessionId: input.session_id,
		bars: input.bars,
		dimension: input.dimension,
		annotation: input.annotation,
	};
	if (input.focus_bars !== undefined) {
		config.focusBars = input.focus_bars;
	}
	return [{ type: "play_passage", config }];
}

const playPassageAnthropicSchema: AnthropicToolSchema = {
	name: "play_passage",
	description:
		"Play back a bar-bounded slice of the student's own recording, with the score visible. Use when you want the student to LISTEN to a specific passage they just played, not just read about it. Only emit when a piece is identified for the current session and score alignment covers the requested bars — otherwise rely on text. The artifact shows the score for `bars` with `focus_bars` tinted in the dimension color.",
	input_schema: {
		type: "object",
		properties: {
			session_id: {
				type: "string",
				format: "uuid",
				description: "UUID of the practice session whose recording to play.",
			},
			bars: {
				type: "array",
				items: { type: "integer", minimum: 1 },
				minItems: 2,
				maxItems: 2,
				description:
					"Outer passage range as [start, end]. The full clip plays from start to end.",
			},
			focus_bars: {
				type: "array",
				items: { type: "integer", minimum: 1 },
				minItems: 2,
				maxItems: 2,
				description:
					"Optional tinted sub-range inside `bars`. Use to draw attention to the specific moment within musical context.",
			},
			dimension: {
				type: "string",
				enum: DIMS_6,
				description: "The single musical dimension this observation is about.",
			},
			annotation: {
				type: "string",
				description:
					"One sentence (<=280 chars) that the student reads next to the playback control.",
			},
		},
		required: ["session_id", "bars", "dimension", "annotation"],
	},
};

// ---------------------------------------------------------------------------
// Tool: search_catalog
// ---------------------------------------------------------------------------

const searchCatalogSchema = z
	.object({
		composer: z.string().min(1).max(200).optional(),
		opus_number: z.number().int().min(1).max(9999).optional(),
		piece_number: z.number().int().min(1).max(9999).optional(),
		title_keywords: z
			.string()
			.min(3)
			.max(200)
			.refine(
				(s) =>
					s
						.trim()
						.split(/\s+/)
						.some((t) => t.length >= 2),
				{
					message:
						"title_keywords must contain at least one token of 2+ characters",
				},
			)
			.optional(),
		query: z.string().min(1).max(300).optional(),
	})
	.refine(
		(data) =>
			data.composer !== undefined ||
			data.opus_number !== undefined ||
			data.piece_number !== undefined ||
			data.title_keywords !== undefined ||
			data.query !== undefined,
		{ message: "At least one search field is required" },
	);

async function processSearchCatalog(
	ctx: ServiceContext,
	_studentId: string,
	rawInput: unknown,
): Promise<InlineComponent[]> {
	const input = searchCatalogSchema.parse(rawInput);

	const conditions = [];

	if (input.composer !== undefined) {
		conditions.push(
			sql`${pieces.composer} ILIKE ${"%" + input.composer + "%"}`,
		);
	}
	if (input.opus_number !== undefined) {
		conditions.push(eq(pieces.opusNumber, input.opus_number));
	}
	if (input.piece_number !== undefined) {
		conditions.push(eq(pieces.pieceNumber, input.piece_number));
	}
	if (input.title_keywords) {
		const tokens = input.title_keywords
			.trim()
			.split(/\s+/)
			.filter((t) => t.length >= 2);
		for (const token of tokens) {
			conditions.push(sql`${pieces.title} ILIKE ${"%" + token + "%"}`);
		}
	}
	// Free-form fallback: token-split across both fields.
	// Only applied when no structured fields provided.
	if (input.query && conditions.length === 0) {
		const tokens = input.query
			.trim()
			.split(/\s+/)
			.filter((t) => t.length >= 2);
		for (const token of tokens) {
			conditions.push(
				sql`(${pieces.composer} ILIKE ${"%" + token + "%"} OR ${pieces.title} ILIKE ${"%" + token + "%"})`,
			);
		}
	}

	if (conditions.length === 0) {
		throw new Error(
			"search_catalog: no conditions after parse (unreachable after validation)",
		);
	}

	const rows = await ctx.db
		.select({
			pieceId: pieces.pieceId,
			composer: pieces.composer,
			title: pieces.title,
			barCount: pieces.barCount,
		})
		.from(pieces)
		.where(and(...conditions))
		.orderBy(asc(pieces.composer), asc(pieces.title))
		.limit(5);

	if (rows.length === 0) {
		return [
			{
				type: "search_catalog_result",
				config: {
					matches: [],
					message:
						"No pieces found. The catalog contains ~242 ASAP pieces. Try fewer fields or use exact opus/piece numbers — e.g., { composer: 'Chopin', opus_number: 64, piece_number: 2 }.",
				},
			},
		];
	}

	return [
		{
			type: "search_catalog_result",
			config: {
				matches: rows.map((r) => ({
					pieceId: r.pieceId,
					composer: r.composer,
					title: r.title,
					barCount: r.barCount,
				})),
			},
		},
	];
}

// ---------------------------------------------------------------------------
// Anthropic JSON schemas (for API calls)
// ---------------------------------------------------------------------------

const prescribeExerciseAnthropicSchema: AnthropicToolSchema = {
	name: "prescribe_exercise",
	description:
		"Prescribe a single targeted practice exercise for the student. Use own_passage_loop when the student has been playing a specific piece and you want them to loop a bar range from it. Use corpus_drill for general technique drills. Use sparingly.",
	input_schema: {
		type: "object",
		properties: {
			kind: {
				type: "string",
				enum: ["own_passage_loop", "corpus_drill"],
				description:
					"own_passage_loop: loop from the student's piece. corpus_drill: general technique drill.",
			},
			target_dimension: {
				type: "string",
				enum: [...DIMS_6],
				description: "The musical dimension this exercise targets.",
			},
			bar_range: {
				type: "array",
				items: { type: "integer", minimum: 1 },
				minItems: 2,
				maxItems: 2,
				description: "Bar range [start, end]. Start must be <= end.",
			},
			tempo_factor: {
				type: "number",
				minimum: 0.25,
				maximum: 1.0,
				description: "Practice tempo as a fraction of performance tempo.",
			},
			primitive_id: {
				type: ["string", "null"],
				description:
					"For corpus_drill only: drill primitive identifier. Pass null in S1.",
			},
			piece_id: {
				type: ["string", "null"],
				description:
					"Catalog piece ID for own_passage_loop. Use search_catalog to find it. Pass null for corpus_drill.",
			},
		},
		required: ["kind", "target_dimension", "bar_range", "tempo_factor", "piece_id"],
	},
};

const scoreHighlightAnthropicSchema: AnthropicToolSchema = {
	name: "score_highlight",
	description:
		"Display specific bars from the score as rendered notation, with optional dimension-colored highlights and annotations. Use whenever the student asks to see, print, look at, or work on a specific passage — this is how you show printed music in the chat. Also use to point at a passage during teaching.",
	input_schema: {
		type: "object",
		properties: {
			piece_id: {
				type: "string",
				description:
					"Piece slug returned by search_catalog (e.g. 'chopin.ballades.1'). Pass through verbatim — do not transform or fabricate.",
			},
			highlights: {
				type: "array",
				description:
					"One to five highlight regions. Each targets a bar range with a dimension and optional annotation.",
				minItems: 1,
				maxItems: 5,
				items: {
					type: "object",
					properties: {
						bars: {
							type: "array",
							items: { type: "integer", minimum: 1 },
							minItems: 2,
							maxItems: 2,
							description:
								"Bar range as [start, end]. Use same number for a single bar (e.g. [4, 4]).",
						},
						dimension: {
							type: "string",
							enum: DIMS_6,
							description: "Which musical dimension this highlight targets.",
						},
						annotation: {
							type: "string",
							description:
								"Optional text annotation to display on the highlighted bars.",
						},
					},
					required: ["bars", "dimension"],
				},
			},
		},
		required: ["piece_id", "highlights"],
	},
};

const keyboardGuideAnthropicSchema: AnthropicToolSchema = {
	name: "keyboard_guide",
	description:
		"Display an interactive keyboard diagram with fingering guidance or hand position notes.",
	input_schema: {
		type: "object",
		properties: {
			title: {
				type: "string",
				description: "Short label for this keyboard guide.",
			},
			description: {
				type: "string",
				description: "Explanation of the technique or pattern being shown.",
			},
			fingering: {
				type: "string",
				description:
					"Optional fingering notation string (e.g. '1-2-3-1-2-3-4-5').",
			},
			hands: {
				type: "string",
				enum: ["left", "right", "both"],
				description: "Which hand(s) this guide applies to.",
			},
		},
		required: ["title", "description", "hands"],
	},
};

const showSessionDataAnthropicSchema: AnthropicToolSchema = {
	name: "show_session_data",
	description:
		"Retrieve practice session data for the student. Always scoped to the current student.",
	input_schema: {
		type: "object",
		properties: {
			query_type: {
				type: "string",
				enum: ["dimension_history", "recent_sessions", "session_detail"],
				description:
					"'dimension_history': past observations for a dimension. 'recent_sessions': list of recent sessions. 'session_detail': full data for a specific session.",
			},
			dimension: {
				type: "string",
				enum: DIMS_6,
				description: "For dimension_history: which dimension to query.",
			},
			session_id: {
				type: "string",
				format: "uuid",
				description: "For session_detail: UUID of the session to retrieve.",
			},
			limit: {
				type: "integer",
				minimum: 1,
				maximum: 50,
				default: 20,
				description:
					"Maximum number of records to return (default 20, ceiling 50).",
			},
		},
		required: ["query_type"],
	},
};

const searchCatalogAnthropicSchema: AnthropicToolSchema = {
	name: "search_catalog",
	description:
		"Search the piece catalog to find a piece's UUID. PREFER structured fields: use composer, opus_number, and piece_number when you can extract them from the student's words — { composer: 'Chopin', opus_number: 64, piece_number: 2 } is exact and unambiguous. Use title_keywords for genre words when opus/number are unknown. Only use query as a last resort.",
	input_schema: {
		type: "object",
		properties: {
			composer: {
				type: "string",
				description:
					"Composer last name. 'Chopin', 'Bach', 'Beethoven'. Case-insensitive substring match.",
			},
			opus_number: {
				type: "integer",
				description:
					"Opus number as integer. 'Op. 64' → 64. Exact match — most important for disambiguation.",
			},
			piece_number: {
				type: "integer",
				description:
					"Piece number within the opus as integer. 'No. 2' → 2. Exact match — critical to distinguish pieces within an opus.",
			},
			title_keywords: {
				type: "string",
				description:
					"Genre or title keywords when opus/number are unknown. 'Waltz', 'Nocturne', 'Ballade'. Token-split substring match.",
			},
			query: {
				type: "string",
				description:
					"Free-form fallback only. Use when you cannot identify composer or structured numbers — e.g., 'that slow Bach prelude'.",
			},
		},
	},
};

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export const TOOL_REGISTRY: Record<string, ToolDefinition> = {
	prescribe_exercise: {
		name: "prescribe_exercise",
		description: prescribeExerciseAnthropicSchema.description,
		schema: prescribeExerciseSchema,
		anthropicSchema: prescribeExerciseAnthropicSchema,
		concurrencySafe: true,
		process: processPrescribeExercise,
	},
	score_highlight: {
		name: "score_highlight",
		description: scoreHighlightAnthropicSchema.description,
		schema: scoreHighlightSchema,
		anthropicSchema: scoreHighlightAnthropicSchema,
		concurrencySafe: true,
		maxResultChars: 5000,
		process: processScoreHighlight,
	},
	keyboard_guide: {
		name: "keyboard_guide",
		description: keyboardGuideAnthropicSchema.description,
		schema: keyboardGuideSchema,
		anthropicSchema: keyboardGuideAnthropicSchema,
		concurrencySafe: true,
		maxResultChars: 2000,
		process: processKeyboardGuide,
	},
	show_session_data: {
		name: "show_session_data",
		description: showSessionDataAnthropicSchema.description,
		schema: showSessionDataSchema,
		anthropicSchema: showSessionDataAnthropicSchema,
		concurrencySafe: true,
		maxResultChars: 10000,
		process: processShowSessionData,
	},
	play_passage: {
		name: "play_passage",
		description: playPassageAnthropicSchema.description,
		schema: playPassageSchema,
		anthropicSchema: playPassageAnthropicSchema,
		concurrencySafe: true,
		maxResultChars: 2000,
		process: processPlayPassage,
	},
	search_catalog: {
		name: "search_catalog",
		description: searchCatalogAnthropicSchema.description,
		schema: searchCatalogSchema,
		anthropicSchema: searchCatalogAnthropicSchema,
		concurrencySafe: true,
		maxResultChars: 3000,
		process: processSearchCatalog,
	},
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function getAnthropicToolSchemas(): AnthropicToolSchema[] {
	return Object.values(TOOL_REGISTRY).map((t) => t.anthropicSchema);
}

/**
 * Process a single tool call from the Anthropic API.
 *
 * Steps:
 * 1. Look up the tool by name.
 * 2. Validate input with Zod.
 * 3. Call the tool's process function.
 * 4. Return ToolResult.
 *
 * On any error: logs structured JSON, returns { isError: true }.
 */
export async function processToolUse(
	ctx: ServiceContext,
	studentId: string,
	toolName: string,
	toolInput: unknown,
): Promise<ToolResult> {
	const tool = TOOL_REGISTRY[toolName];

	if (!tool) {
		console.error(
			JSON.stringify({
				level: "error",
				message: "Unknown tool name",
				toolName,
				studentId,
			}),
		);
		return {
			name: toolName,
			componentsJson: [],
			isError: true,
			errorMessage: `Unknown tool '${toolName}'. This tool is not registered.`,
		};
	}

	const validation = tool.schema.safeParse(toolInput);
	if (!validation.success) {
		const errorMessage = validation.error.message;
		console.error(
			JSON.stringify({
				level: "error",
				message: "Tool input validation failed",
				toolName,
				studentId,
				issues: validation.error.issues,
			}),
		);
		return { name: toolName, componentsJson: [], isError: true, errorMessage };
	}

	try {
		const componentsJson = await tool.process(ctx, studentId, validation.data);

		// Apply maxResultChars truncation if needed
		if (tool.maxResultChars !== undefined) {
			const serialized = JSON.stringify(componentsJson);
			if (serialized.length > tool.maxResultChars) {
				const truncationNotice: InlineComponent = {
					type: "truncation_notice",
					config: {
						message: `Result truncated: exceeded ${tool.maxResultChars} character limit.`,
						originalLength: serialized.length,
					},
				};
				return {
					name: toolName,
					componentsJson: [truncationNotice],
					isError: false,
				};
			}
		}

		return { name: toolName, componentsJson, isError: false };
	} catch (err) {
		const errorMessage = err instanceof Error ? err.message : String(err);
		console.error(
			JSON.stringify({
				level: "error",
				message: "Tool process function threw",
				toolName,
				studentId,
				error: errorMessage,
				stack: err instanceof Error ? err.stack : undefined,
			}),
		);
		return { name: toolName, componentsJson: [], isError: true, errorMessage };
	}
}
