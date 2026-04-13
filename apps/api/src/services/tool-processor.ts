import { and, asc, desc, eq, sql } from "drizzle-orm";
import { z } from "zod";
import { pieces } from "../db/schema/catalog";
import { exerciseDimensions, exercises } from "../db/schema/exercises";
import { observations } from "../db/schema/observations";
import { sessions } from "../db/schema/sessions";
import { DIMS_6 } from "../lib/dims";
import { InferenceError } from "../lib/errors";
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
// Tool: create_exercise
// ---------------------------------------------------------------------------

const createExerciseSchema = z.object({
	source_passage: z.string().min(1).max(500),
	target_skill: z.string().min(1).max(500),
	exercises: z
		.array(
			z.object({
				title: z.string().min(1).max(200),
				instruction: z.string().min(1).max(4000),
				focus_dimension: dimensionEnum,
				hands: z.enum(["left", "right", "both"]).optional(),
			}),
		)
		.min(1)
		.max(3),
});

async function persistGeneratedExercise(
	ctx: ServiceContext,
	title: string,
	instruction: string,
	focusDimension: string,
	sourcePassage: string,
	targetSkill: string,
): Promise<string> {
	const [inserted] = await ctx.db
		.insert(exercises)
		.values({
			title,
			description: `${targetSkill} -- ${sourcePassage}`,
			instructions: instruction,
			difficulty: "intermediate",
			category: "generated",
			source: "teacher_llm",
		})
		.returning({ id: exercises.id });

	if (!inserted) {
		throw new InferenceError("Failed to insert generated exercise");
	}

	await ctx.db.insert(exerciseDimensions).values({
		exerciseId: inserted.id,
		dimension: focusDimension,
	});

	return inserted.id;
}

async function processCreateExercise(
	ctx: ServiceContext,
	_studentId: string,
	rawInput: unknown,
): Promise<InlineComponent[]> {
	const input = createExerciseSchema.parse(rawInput);

	const processed: unknown[] = [];

	for (const ex of input.exercises) {
		const exerciseId = await persistGeneratedExercise(
			ctx,
			ex.title,
			ex.instruction,
			ex.focus_dimension,
			input.source_passage,
			input.target_skill,
		);

		const exJson: Record<string, unknown> = {
			title: ex.title,
			instruction: ex.instruction,
			focusDimension: ex.focus_dimension,
			exerciseId,
		};
		if (ex.hands) {
			exJson.hands = ex.hands;
		}
		processed.push(exJson);
	}

	return [
		{
			type: "exercise_set",
			config: {
				sourcePassage: input.source_passage,
				targetSkill: input.target_skill,
				exercises: processed,
			},
		},
	];
}

// ---------------------------------------------------------------------------
// Tool: score_highlight
// ---------------------------------------------------------------------------

const scoreHighlightSchema = z.object({
	piece_id: z.string().uuid(),
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
// Tool: reference_browser
// ---------------------------------------------------------------------------

const referenceBrowserSchema = z.object({
	piece_id: z.string().uuid().optional(),
	passage: z.string().optional(),
	description: z.string(),
});

async function processReferenceBrowser(
	ctx: ServiceContext,
	_studentId: string,
	rawInput: unknown,
): Promise<InlineComponent[]> {
	const input = referenceBrowserSchema.parse(rawInput);

	const config: Record<string, unknown> = {
		description: input.description,
	};

	if (input.piece_id !== undefined) {
		const catalogRow = await ctx.db
			.select({ pieceId: pieces.pieceId })
			.from(pieces)
			.where(eq(pieces.pieceId, input.piece_id))
			.limit(1);

		if (catalogRow.length === 0) {
			console.log(
				JSON.stringify({
					level: "warn",
					message:
						"reference_browser piece_id not found in catalog, omitting from config",
					pieceId: input.piece_id,
				}),
			);
		} else {
			config.pieceId = input.piece_id;
		}
	}

	if (input.passage !== undefined) {
		config.passage = input.passage;
	}

	return [{ type: "reference_browser", config }];
}

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
				(s) => s.trim().split(/\s+/).some((t) => t.length >= 2),
				{ message: "title_keywords must contain at least one token of 2+ characters" },
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
		conditions.push(sql`${pieces.composer} ILIKE ${"%" + input.composer + "%"}`);
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

const createExerciseAnthropicSchema: AnthropicToolSchema = {
	name: "create_exercise",
	description:
		"Create one or more targeted practice exercises for a specific passage and skill. Persists exercises to the database.",
	input_schema: {
		type: "object",
		properties: {
			source_passage: {
				type: "string",
				description:
					"The musical passage or section the exercises target (e.g. 'bars 5-8', 'opening theme').",
			},
			target_skill: {
				type: "string",
				description: "The specific skill or technique the exercises develop.",
			},
			exercises: {
				type: "array",
				description: "One or more exercises to assign.",
				minItems: 1,
				items: {
					type: "object",
					properties: {
						title: {
							type: "string",
							description: "Short title for the exercise.",
						},
						instruction: {
							type: "string",
							description: "Step-by-step instruction the student follows.",
						},
						focus_dimension: {
							type: "string",
							enum: DIMS_6,
							description:
								"Which of the 6 musical dimensions this exercise targets.",
						},
						hands: {
							type: "string",
							description:
								"Optional: which hands to use ('left', 'right', or 'both').",
						},
					},
					required: ["title", "instruction", "focus_dimension"],
				},
			},
		},
		required: ["source_passage", "target_skill", "exercises"],
	},
};

const scoreHighlightAnthropicSchema: AnthropicToolSchema = {
	name: "score_highlight",
	description:
		"Highlight one or more bar ranges in the score viewer with dimension-colored annotations. Use to visually point at specific passages during teaching.",
	input_schema: {
		type: "object",
		properties: {
			piece_id: {
				type: "string",
				format: "uuid",
				description: "UUID of the piece being discussed. Required.",
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

const referenceBrowserAnthropicSchema: AnthropicToolSchema = {
	name: "reference_browser",
	description:
		"Display a reference panel with contextual information about a piece or passage.",
	input_schema: {
		type: "object",
		properties: {
			piece_id: {
				type: "string",
				format: "uuid",
				description: "Optional UUID of the piece to look up.",
			},
			passage: {
				type: "string",
				description: "Optional passage description (e.g. 'bars 10-15').",
			},
			description: {
				type: "string",
				description: "What information to surface in the reference panel.",
			},
		},
		required: ["description"],
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
	create_exercise: {
		name: "create_exercise",
		description: createExerciseAnthropicSchema.description,
		schema: createExerciseSchema,
		anthropicSchema: createExerciseAnthropicSchema,
		concurrencySafe: false,
		process: processCreateExercise,
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
	reference_browser: {
		name: "reference_browser",
		description: referenceBrowserAnthropicSchema.description,
		schema: referenceBrowserSchema,
		anthropicSchema: referenceBrowserAnthropicSchema,
		concurrencySafe: true,
		maxResultChars: 2000,
		process: processReferenceBrowser,
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
		return { name: toolName, componentsJson: [], isError: true };
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
