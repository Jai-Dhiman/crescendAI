import { describe, expect, it } from "vitest";
import { DIMS_6 } from "../lib/dims";
import {
	getAnthropicToolSchemas,
	TOOL_REGISTRY,
	type ToolResult,
} from "./tool-processor";

// ---------------------------------------------------------------------------
// Registry structure tests
// ---------------------------------------------------------------------------

describe("TOOL_REGISTRY structure", () => {
	const toolNames = [
		"create_exercise",
		"score_highlight",
		"keyboard_guide",
		"show_session_data",
		"reference_browser",
		"search_catalog",
	] as const;

	it("has all 6 tools", () => {
		expect(Object.keys(TOOL_REGISTRY)).toHaveLength(6);
		for (const name of toolNames) {
			expect(TOOL_REGISTRY[name]).toBeDefined();
		}
	});

	it("each tool has required fields", () => {
		for (const name of toolNames) {
			const tool = TOOL_REGISTRY[name];
			expect(typeof tool.name).toBe("string");
			expect(tool.schema).toBeDefined();
			expect(tool.anthropicSchema).toBeDefined();
			expect(typeof tool.description).toBe("string");
			expect(typeof tool.concurrencySafe).toBe("boolean");
			expect(typeof tool.process).toBe("function");
		}
	});

	it("create_exercise is not concurrencySafe", () => {
		expect(TOOL_REGISTRY.create_exercise.concurrencySafe).toBe(false);
	});

	it("score_highlight, keyboard_guide, show_session_data, reference_browser are concurrencySafe", () => {
		expect(TOOL_REGISTRY.score_highlight.concurrencySafe).toBe(true);
		expect(TOOL_REGISTRY.keyboard_guide.concurrencySafe).toBe(true);
		expect(TOOL_REGISTRY.show_session_data.concurrencySafe).toBe(true);
		expect(TOOL_REGISTRY.reference_browser.concurrencySafe).toBe(true);
		expect(TOOL_REGISTRY.search_catalog.concurrencySafe).toBe(true);
	});

	it("maxResultChars defined for bounded tools", () => {
		expect(TOOL_REGISTRY.score_highlight.maxResultChars).toBe(5000);
		expect(TOOL_REGISTRY.keyboard_guide.maxResultChars).toBe(2000);
		expect(TOOL_REGISTRY.show_session_data.maxResultChars).toBe(10000);
		expect(TOOL_REGISTRY.reference_browser.maxResultChars).toBe(2000);
		expect(TOOL_REGISTRY.search_catalog.maxResultChars).toBe(3000);
	});

	it("create_exercise has no maxResultChars", () => {
		expect(TOOL_REGISTRY.create_exercise.maxResultChars).toBeUndefined();
	});
});

// ---------------------------------------------------------------------------
// getAnthropicToolSchemas
// ---------------------------------------------------------------------------

describe("getAnthropicToolSchemas", () => {
	it("returns an array of 6 schemas", () => {
		const schemas = getAnthropicToolSchemas();
		expect(schemas).toHaveLength(6);
	});

	it("each schema has name, description, input_schema", () => {
		for (const schema of getAnthropicToolSchemas()) {
			expect(typeof schema.name).toBe("string");
			expect(typeof schema.description).toBe("string");
			expect(schema.input_schema).toBeDefined();
			expect(schema.input_schema.type).toBe("object");
		}
	});
});

// ---------------------------------------------------------------------------
// create_exercise Zod validation
// ---------------------------------------------------------------------------

describe("create_exercise schema validation", () => {
	const schema = TOOL_REGISTRY.create_exercise.schema;

	it("passes valid input", () => {
		const result = schema.safeParse({
			source_passage: "bars 1-4",
			target_skill: "legato phrasing",
			exercises: [
				{
					title: "Slow practice",
					instruction: "Play slowly with full bow",
					focus_dimension: "phrasing",
				},
			],
		});
		expect(result.success).toBe(true);
	});

	it("passes with optional hands field", () => {
		const result = schema.safeParse({
			source_passage: "opening",
			target_skill: "voicing",
			exercises: [
				{
					title: "Thumb drill",
					instruction: "Isolate thumb",
					focus_dimension: "dynamics",
					hands: "right",
				},
			],
		});
		expect(result.success).toBe(true);
	});

	it("rejects missing source_passage", () => {
		const result = schema.safeParse({
			target_skill: "timing",
			exercises: [{ title: "X", instruction: "Y", focus_dimension: "timing" }],
		});
		expect(result.success).toBe(false);
	});

	it("rejects empty exercises array", () => {
		const result = schema.safeParse({
			source_passage: "bars 1-4",
			target_skill: "timing",
			exercises: [],
		});
		expect(result.success).toBe(false);
	});

	it("rejects invalid focus_dimension", () => {
		const result = schema.safeParse({
			source_passage: "bars 1-4",
			target_skill: "timing",
			exercises: [
				{ title: "X", instruction: "Y", focus_dimension: "invalid_dim" },
			],
		});
		expect(result.success).toBe(false);
	});

	it("accepts all DIMS_6 values for focus_dimension", () => {
		for (const dim of DIMS_6) {
			const result = schema.safeParse({
				source_passage: "bar 1",
				target_skill: "test",
				exercises: [{ title: "T", instruction: "I", focus_dimension: dim }],
			});
			expect(result.success).toBe(true);
		}
	});
});

// ---------------------------------------------------------------------------
// score_highlight Zod validation
// ---------------------------------------------------------------------------

describe("score_highlight schema validation", () => {
	const schema = TOOL_REGISTRY.score_highlight.schema;

	it("passes valid single highlight", () => {
		const result = schema.safeParse({
			piece_id: "chopin.ballades.1",
			highlights: [
				{ bars: [1, 4], dimension: "dynamics" },
			],
		});
		expect(result.success).toBe(true);
	});

	it("passes multiple highlights with annotations", () => {
		const result = schema.safeParse({
			piece_id: "chopin.ballades.1",
			highlights: [
				{ bars: [1, 4], dimension: "dynamics", annotation: "crescendo here" },
				{ bars: [12, 16], dimension: "pedaling", annotation: "sustain bleeds" },
			],
		});
		expect(result.success).toBe(true);
	});

	it("rejects missing piece_id", () => {
		const result = schema.safeParse({
			highlights: [{ bars: [1, 4], dimension: "dynamics" }],
		});
		expect(result.success).toBe(false);
	});

	it("rejects empty highlights array", () => {
		const result = schema.safeParse({
			piece_id: "chopin.ballades.1",
			highlights: [],
		});
		expect(result.success).toBe(false);
	});

	it("rejects more than 5 highlights", () => {
		const highlights = Array.from({ length: 6 }, (_, i) => ({
			bars: [i + 1, i + 2],
			dimension: "dynamics",
		}));
		const result = schema.safeParse({
			piece_id: "chopin.ballades.1",
			highlights,
		});
		expect(result.success).toBe(false);
	});

	it("rejects invalid dimension", () => {
		const result = schema.safeParse({
			piece_id: "chopin.ballades.1",
			highlights: [{ bars: [1, 4], dimension: "rhythm" }],
		});
		expect(result.success).toBe(false);
	});

	it("rejects bars where start > end", () => {
		const result = schema.safeParse({
			piece_id: "chopin.ballades.1",
			highlights: [{ bars: [8, 4], dimension: "dynamics" }],
		});
		expect(result.success).toBe(false);
	});

	it("accepts catalog slug piece_id", () => {
		const result = schema.safeParse({
			piece_id: "chopin.ballades.1",
			highlights: [{ bars: [1, 4], dimension: "dynamics" }],
		});
		expect(result.success).toBe(true);
	});

	it("rejects piece_id with invalid characters", () => {
		const result = schema.safeParse({
			piece_id: "Chopin Ballade #1",
			highlights: [{ bars: [1, 4], dimension: "dynamics" }],
		});
		expect(result.success).toBe(false);
	});

	it("rejects empty piece_id", () => {
		const result = schema.safeParse({
			piece_id: "",
			highlights: [{ bars: [1, 4], dimension: "dynamics" }],
		});
		expect(result.success).toBe(false);
	});

	it("rejects old bars-string format", () => {
		const result = schema.safeParse({ bars: "1-4" });
		expect(result.success).toBe(false);
	});
});

// ---------------------------------------------------------------------------
// keyboard_guide Zod validation
// ---------------------------------------------------------------------------

describe("keyboard_guide schema validation", () => {
	const schema = TOOL_REGISTRY.keyboard_guide.schema;

	it("passes valid input", () => {
		const result = schema.safeParse({
			title: "Thumb under",
			description: "Practice thumb-under technique",
			hands: "right",
		});
		expect(result.success).toBe(true);
	});

	it("passes with optional fingering", () => {
		const result = schema.safeParse({
			title: "Scale fingering",
			description: "C major fingering pattern",
			fingering: "1-2-3-1-2-3-4-5",
			hands: "both",
		});
		expect(result.success).toBe(true);
	});

	it("rejects invalid hands value", () => {
		const result = schema.safeParse({
			title: "T",
			description: "D",
			hands: "one",
		});
		expect(result.success).toBe(false);
	});

	it("rejects missing title", () => {
		const result = schema.safeParse({ description: "D", hands: "left" });
		expect(result.success).toBe(false);
	});

	it("accepts all hands enum values", () => {
		for (const hands of ["left", "right", "both"] as const) {
			const result = schema.safeParse({
				title: "T",
				description: "D",
				hands,
			});
			expect(result.success).toBe(true);
		}
	});
});

// ---------------------------------------------------------------------------
// show_session_data Zod validation
// ---------------------------------------------------------------------------

describe("show_session_data schema validation", () => {
	const schema = TOOL_REGISTRY.show_session_data.schema;

	it("passes dimension_history query type with dimension", () => {
		const result = schema.safeParse({
			query_type: "dimension_history",
			dimension: "dynamics",
		});
		expect(result.success).toBe(true);
	});

	it("passes recent_sessions with default limit", () => {
		const result = schema.safeParse({ query_type: "recent_sessions" });
		expect(result.success).toBe(true);
		if (result.success) {
			expect(result.data.limit).toBe(20);
		}
	});

	it("passes session_detail with session_id", () => {
		const result = schema.safeParse({
			query_type: "session_detail",
			session_id: "123e4567-e89b-12d3-a456-426614174000",
		});
		expect(result.success).toBe(true);
	});

	it("rejects limit > 50", () => {
		const result = schema.safeParse({
			query_type: "recent_sessions",
			limit: 51,
		});
		expect(result.success).toBe(false);
	});

	it("rejects limit < 1", () => {
		const result = schema.safeParse({
			query_type: "recent_sessions",
			limit: 0,
		});
		expect(result.success).toBe(false);
	});

	it("rejects invalid query_type", () => {
		const result = schema.safeParse({ query_type: "all_data" });
		expect(result.success).toBe(false);
	});

	it("rejects invalid dimension", () => {
		const result = schema.safeParse({
			query_type: "dimension_history",
			dimension: "vibrato",
		});
		expect(result.success).toBe(false);
	});

	it("rejects invalid session_id (not uuid)", () => {
		const result = schema.safeParse({
			query_type: "session_detail",
			session_id: "not-a-uuid",
		});
		expect(result.success).toBe(false);
	});
});

// ---------------------------------------------------------------------------
// reference_browser Zod validation
// ---------------------------------------------------------------------------

describe("reference_browser schema validation", () => {
	const schema = TOOL_REGISTRY.reference_browser.schema;

	it("passes with description only", () => {
		const result = schema.safeParse({
			description: "Look up fingering for Chopin nocturne",
		});
		expect(result.success).toBe(true);
	});

	it("passes with all optional fields", () => {
		const result = schema.safeParse({
			piece_id: "chopin.ballades.1",
			passage: "bars 5-8",
			description: "Check phrasing",
		});
		expect(result.success).toBe(true);
	});

	it("rejects piece_id with invalid characters", () => {
		const result = schema.safeParse({
			description: "test",
			piece_id: "Chopin Ballade #1",
		});
		expect(result.success).toBe(false);
	});

	it("rejects missing description", () => {
		const result = schema.safeParse({ passage: "bars 1-4" });
		expect(result.success).toBe(false);
	});
});

// ---------------------------------------------------------------------------
// search_catalog Zod validation
// ---------------------------------------------------------------------------

describe("search_catalog schema validation", () => {
	const schema = TOOL_REGISTRY.search_catalog.schema;

	it("passes with composer only", () => {
		const result = schema.safeParse({ composer: "Chopin" });
		expect(result.success).toBe(true);
	});

	it("passes with opus_number only", () => {
		const result = schema.safeParse({ opus_number: 64 });
		expect(result.success).toBe(true);
	});

	it("passes with piece_number only", () => {
		const result = schema.safeParse({ piece_number: 2 });
		expect(result.success).toBe(true);
	});

	it("passes with title_keywords only", () => {
		const result = schema.safeParse({ title_keywords: "Waltz" });
		expect(result.success).toBe(true);
	});

	it("passes with query only", () => {
		const result = schema.safeParse({ query: "Chopin waltz" });
		expect(result.success).toBe(true);
	});

	it("passes with composer + opus_number + piece_number (primary use case)", () => {
		const result = schema.safeParse({
			composer: "Chopin",
			opus_number: 64,
			piece_number: 2,
		});
		expect(result.success).toBe(true);
	});

	it("rejects empty object", () => {
		const result = schema.safeParse({});
		expect(result.success).toBe(false);
	});

	it("rejects opus_number below 1", () => {
		const result = schema.safeParse({ opus_number: 0 });
		expect(result.success).toBe(false);
	});

	it("rejects opus_number above 9999", () => {
		const result = schema.safeParse({ opus_number: 10000 });
		expect(result.success).toBe(false);
	});

	it("rejects non-integer opus_number", () => {
		const result = schema.safeParse({ opus_number: 64.5 });
		expect(result.success).toBe(false);
	});

	it("rejects title_keywords with only single-char tokens", () => {
		const result = schema.safeParse({ title_keywords: "a b" });
		expect(result.success).toBe(false);
	});

	it("passes title_keywords with at least one 2-char token", () => {
		const result = schema.safeParse({ title_keywords: "Waltz" });
		expect(result.success).toBe(true);
	});
});

// ---------------------------------------------------------------------------
// processToolUse (no DB required -- pass-through tools)
// ---------------------------------------------------------------------------

describe("processToolUse pass-through tools", () => {
	// We need a minimal ServiceContext mock -- pass-through tools don't hit DB
	const mockCtx = {
		db: {
			select: () => ({
				from: () => ({
					where: () => ({
						limit: () => Promise.resolve([]),
					}),
				}),
			}),
		} as never,
		env: {} as never,
	};
	const studentId = "student-abc";

	it("keyboard_guide returns ToolResult with correct type", async () => {
		const { processToolUse } = await import("./tool-processor");
		const result: ToolResult = await processToolUse(
			mockCtx,
			studentId,
			"keyboard_guide",
			{
				title: "Thumb under",
				description: "Practice thumb-under passing",
				hands: "right",
			},
		);
		expect(result.isError).toBe(false);
		expect(result.name).toBe("keyboard_guide");
		expect(result.componentsJson).toHaveLength(1);
		expect(result.componentsJson[0].type).toBe("keyboard_guide");
	});

	it("reference_browser returns ToolResult with correct type", async () => {
		const { processToolUse } = await import("./tool-processor");
		const result: ToolResult = await processToolUse(
			mockCtx,
			studentId,
			"reference_browser",
			{
				description: "Look up phrasing for bar 10",
			},
		);
		expect(result.isError).toBe(false);
		expect(result.name).toBe("reference_browser");
		expect(result.componentsJson).toHaveLength(1);
		expect(result.componentsJson[0].type).toBe("reference_browser");
	});

	it("unknown tool name returns isError: true", async () => {
		const { processToolUse } = await import("./tool-processor");
		const result: ToolResult = await processToolUse(
			mockCtx,
			studentId,
			"nonexistent_tool",
			{},
		);
		expect(result.isError).toBe(true);
		expect(result.name).toBe("nonexistent_tool");
	});

	it("validation failure returns isError: true", async () => {
		const { processToolUse } = await import("./tool-processor");
		const result: ToolResult = await processToolUse(
			mockCtx,
			studentId,
			"keyboard_guide",
			{ hands: "right" }, // missing title and description
		);
		expect(result.isError).toBe(true);
	});

	it("score_highlight pass-through returns ToolResult", async () => {
		const { processToolUse } = await import("./tool-processor");
		const result: ToolResult = await processToolUse(
			mockCtx,
			studentId,
			"score_highlight",
			{
				piece_id: "chopin.ballades.1",
				highlights: [{ bars: [1, 8], dimension: "dynamics" }],
			},
		);
		// Note: will fail catalog lookup with mock ctx, but should not throw
		// because processScoreHighlight logs a warning and continues
		expect(result.isError).toBe(false);
		expect(result.componentsJson[0].type).toBe("score_highlight");
	});

	it("score_highlight config has no scoreData field", async () => {
		const { processToolUse } = await import("./tool-processor");
		const result: ToolResult = await processToolUse(
			mockCtx,
			studentId,
			"score_highlight",
			{
				piece_id: "chopin.ballades.1",
				highlights: [{ bars: [1, 8], dimension: "dynamics" }],
			},
		);
		expect(result.isError).toBe(false);
		const config = result.componentsJson[0].config;
		expect(config).not.toHaveProperty("scoreData");
		expect(config).toHaveProperty("pieceId");
		expect(config).toHaveProperty("highlights");
	});

	it("search_catalog returns matches for structured query", async () => {
		const searchMockCtx = {
			db: {
				select: () => ({
					from: () => ({
						where: () => ({
							orderBy: () => ({
								limit: () =>
									Promise.resolve([
										{
											pieceId: "abc-123",
											composer: "Chopin",
											title: "Waltz Op. 64 No. 2",
											barCount: 138,
										},
									]),
							}),
						}),
					}),
				}),
			} as never,
			env: {} as never,
		};

		const { processToolUse } = await import("./tool-processor");
		const result: ToolResult = await processToolUse(
			searchMockCtx,
			studentId,
			"search_catalog",
			{ composer: "Chopin", opus_number: 64, piece_number: 2 },
		);
		expect(result.isError).toBe(false);
		expect(result.name).toBe("search_catalog");
		expect(result.componentsJson).toHaveLength(1);
		expect(result.componentsJson[0].type).toBe("search_catalog_result");

		const config = result.componentsJson[0].config as {
			matches: Array<{ pieceId: string; composer: string; title: string }>;
		};
		expect(config.matches).toHaveLength(1);
		expect(config.matches[0].pieceId).toBe("abc-123");
		expect(config.matches[0].composer).toBe("Chopin");
	});

	it("search_catalog returns matches for query fallback", async () => {
		const fallbackMockCtx = {
			db: {
				select: () => ({
					from: () => ({
						where: () => ({
							orderBy: () => ({
								limit: () =>
									Promise.resolve([
										{
											pieceId: "def-456",
											composer: "Bach",
											title: "WTC I - Prelude - 1",
											barCount: 32,
										},
									]),
							}),
						}),
					}),
				}),
			} as never,
			env: {} as never,
		};

		const { processToolUse } = await import("./tool-processor");
		const result: ToolResult = await processToolUse(
			fallbackMockCtx,
			studentId,
			"search_catalog",
			{ query: "Bach WTC prelude" },
		);
		expect(result.isError).toBe(false);
		const config = result.componentsJson[0].config as {
			matches: Array<{ pieceId: string }>;
		};
		expect(config.matches).toHaveLength(1);
		expect(config.matches[0].pieceId).toBe("def-456");
	});

	it("search_catalog empty result message references opus_number and piece_number", async () => {
		const emptyMockCtx = {
			db: {
				select: () => ({
					from: () => ({
						where: () => ({
							orderBy: () => ({
								limit: () => Promise.resolve([]),
							}),
						}),
					}),
				}),
			} as never,
			env: {} as never,
		};

		const { processToolUse } = await import("./tool-processor");
		const result: ToolResult = await processToolUse(
			emptyMockCtx,
			studentId,
			"search_catalog",
			{ composer: "Nonexistent" },
		);
		expect(result.isError).toBe(false);
		const config = result.componentsJson[0].config as {
			matches: unknown[];
			message?: string;
		};
		expect(config.matches).toHaveLength(0);
		// New message guides caller toward structured fields
		expect(config.message).toContain("opus_number");
		expect(config.message).toContain("piece_number");
	});
});
