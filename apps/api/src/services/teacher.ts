import * as Sentry from "@sentry/cloudflare";
import { InferenceError } from "../lib/errors";
import type { ServiceContext } from "../lib/types";
import {
	type AnthropicContentBlock,
	type AnthropicSystemBlock,
	callAnthropic,
	callAnthropicStream,
} from "./llm";
import { buildMemoryContext } from "./memory";
import { buildSynthesisFraming, UNIFIED_TEACHER_SYSTEM } from "./prompts";
import {
	getAnthropicToolSchemas,
	type InlineComponent,
	processToolUse,
	type ToolResult,
} from "./tool-processor";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ToolCallRecord {
	id: string;
	name: string;
	input: unknown;
	result: ToolResult;
}

export type TeacherEvent =
	| { type: "delta"; text: string }
	| { type: "tool_start"; name: string }
	| { type: "tool_result"; name: string; componentsJson: InlineComponent[] }
	| { type: "tool_error"; name: string; message: string }
	| {
			type: "done";
			fullText: string;
			allComponents: InlineComponent[];
			toolCalls: ToolCallRecord[];
			stopReason: string;
	  };

export interface TeacherResponse {
	text: string;
	toolResults: ToolResult[];
}

export interface SynthesisInput {
	studentId: string;
	conversationId: string | null;
	sessionDurationMs: number;
	practicePattern: string;
	topMoments: unknown[];
	drillingRecords: unknown[];
	pieceMetadata: { composer?: string; title?: string } | null;
}

type ProcessToolFn = (name: string, input: unknown) => Promise<ToolResult>;

// ---------------------------------------------------------------------------
// Content block tracking
// ---------------------------------------------------------------------------

interface TextBlock {
	type: "text";
	textAccumulator: string;
}

interface ToolUseBlock {
	type: "tool_use";
	id: string;
	name: string;
	jsonAccumulator: string;
}

type ContentBlock = TextBlock | ToolUseBlock;

// ---------------------------------------------------------------------------
// SSE line parser
// ---------------------------------------------------------------------------

function* parseSSELines(
	chunk: string,
): Generator<{ event: string; data: string }> {
	// Split by double newline to get SSE messages
	const messages = chunk.split(/\n\n/);
	for (const message of messages) {
		if (message.trim() === "") continue;
		const lines = message.split("\n");
		let event = "";
		let data = "";
		for (const line of lines) {
			if (line.startsWith("event:")) {
				event = line.slice("event:".length).trim();
			} else if (line.startsWith("data:")) {
				data = line.slice("data:".length).trim();
			}
		}
		if (event && data) {
			yield { event, data };
		}
	}
}

// ---------------------------------------------------------------------------
// processSSEEvent — shared handler for both the main loop and flush path
// ---------------------------------------------------------------------------

async function processSSEEvent(
	event: string,
	data: string,
	blocks: Map<number, ContentBlock>,
	state: {
		fullText: string;
		allComponents: InlineComponent[];
		toolCalls: ToolCallRecord[];
		stopReason: string;
		pendingTextDeltas: string[];
		hasToolUseThisTurn: boolean;
	},
	processToolFn: ProcessToolFn,
): Promise<TeacherEvent[]> {
	let parsed: Record<string, unknown>;
	try {
		parsed = JSON.parse(data) as Record<string, unknown>;
	} catch (err) {
		console.error(
			JSON.stringify({
				level: "error",
				message: "Failed to parse SSE data JSON",
				event,
				data,
				error: err instanceof Error ? err.message : String(err),
			}),
		);
		Sentry.captureException(err, {
			tags: { service: "teacher", operation: "sse_parse" },
			extra: { sseEvent: event, rawData: data },
		});
		return [];
	}

	if (event === "content_block_start") {
		const index = parsed["index"] as number;
		const contentBlock = parsed["content_block"] as Record<string, unknown>;
		const blockType = contentBlock["type"] as string;

		if (blockType === "text") {
			blocks.set(index, { type: "text", textAccumulator: "" });
		} else if (blockType === "tool_use") {
			const id = contentBlock["id"] as string;
			const name = contentBlock["name"] as string;
			blocks.set(index, { type: "tool_use", id, name, jsonAccumulator: "" });
			// Discard any text buffered in this turn — it was intermediate narration.
			state.pendingTextDeltas = [];
			state.hasToolUseThisTurn = true;
			return [{ type: "tool_start", name }];
		}
		return [];
	}

	if (event === "content_block_delta") {
		const index = parsed["index"] as number;
		const delta = parsed["delta"] as Record<string, unknown>;
		const deltaType = delta["type"] as string;
		const block = blocks.get(index);

		if (!block) return [];

		if (deltaType === "text_delta" && block.type === "text") {
			const text = delta["text"] as string;
			block.textAccumulator += text;
			state.fullText += text;
			state.pendingTextDeltas.push(text);
			return [];
		} else if (deltaType === "input_json_delta" && block.type === "tool_use") {
			block.jsonAccumulator += delta["partial_json"] as string;
		}
		return [];
	}

	if (event === "content_block_stop") {
		const index = parsed["index"] as number;
		const block = blocks.get(index);

		if (!block || block.type !== "tool_use") return [];

		let toolInput: unknown;
		try {
			toolInput = JSON.parse(block.jsonAccumulator);
		} catch (err) {
			const parseMsg = err instanceof Error ? err.message : String(err);
			console.error(
				JSON.stringify({
					level: "error",
					message: "Failed to parse tool_use JSON accumulator",
					toolName: block.name,
					accumulated: block.jsonAccumulator,
					error: parseMsg,
				}),
			);
			return [
				{
					type: "tool_error",
					name: block.name,
					message: `The model sent malformed input for ${block.name}: ${parseMsg}`,
				},
			];
		}

		const result = await processToolFn(block.name, toolInput);
		state.toolCalls.push({ id: block.id, name: block.name, input: toolInput, result });
		if (!result.isError) {
			state.allComponents.push(...result.componentsJson);
			return [
				{
					type: "tool_result",
					name: result.name,
					componentsJson: result.componentsJson,
				},
			];
		}
		return [
			{
				type: "tool_error",
				name: result.name,
				message: result.errorMessage ?? "Tool call failed.",
			},
		];
	}

	if (event === "message_delta") {
		const delta = parsed["delta"] as Record<string, unknown> | undefined;
		if (delta && typeof delta["stop_reason"] === "string") {
			state.stopReason = delta["stop_reason"];
		}
		// Flush buffered text deltas only for final (non-tool-use) turns.
		if (state.stopReason !== "tool_use" && state.pendingTextDeltas.length > 0) {
			const flushed = state.pendingTextDeltas.map((text) => ({ type: "delta" as const, text }));
			state.pendingTextDeltas = [];
			return flushed;
		}
		state.pendingTextDeltas = [];
		return [];
	}

	// message_stop: flush any remaining buffered deltas (covers streams with no message_delta)
	if (event === "message_stop") {
		if (!state.hasToolUseThisTurn && state.pendingTextDeltas.length > 0) {
			const flushed = state.pendingTextDeltas.map((text) => ({ type: "delta" as const, text }));
			state.pendingTextDeltas = [];
			return flushed;
		}
		state.pendingTextDeltas = [];
		return [];
	}

	// message_start — no action needed
	return [];
}

// ---------------------------------------------------------------------------
// parseAnthropicStream
// ---------------------------------------------------------------------------

export async function* parseAnthropicStream(
	stream: ReadableStream,
	processToolFn: ProcessToolFn,
): AsyncGenerator<TeacherEvent> {
	const decoder = new TextDecoder();
	const reader = stream.getReader();

	const blocks = new Map<number, ContentBlock>();
	const state = {
		fullText: "",
		allComponents: [] as InlineComponent[],
		toolCalls: [] as ToolCallRecord[],
		stopReason: "end_turn",
		pendingTextDeltas: [] as string[],
		hasToolUseThisTurn: false,
	};
	let textBuffer = "";

	try {
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			textBuffer += decoder.decode(value, { stream: true });

			// Process complete SSE messages — keep partial message in buffer
			const lastDoubleNewline = textBuffer.lastIndexOf("\n\n");
			if (lastDoubleNewline === -1) continue;

			const toProcess = textBuffer.slice(0, lastDoubleNewline + 2);
			textBuffer = textBuffer.slice(lastDoubleNewline + 2);

			for (const { event, data } of parseSSELines(toProcess)) {
				const events = await processSSEEvent(
					event,
					data,
					blocks,
					state,
					processToolFn,
				);
				for (const e of events) yield e;
			}
		}

		// Flush any remaining buffer content
		if (textBuffer.trim()) {
			for (const { event, data } of parseSSELines(textBuffer)) {
				const events = await processSSEEvent(
					event,
					data,
					blocks,
					state,
					processToolFn,
				);
				for (const e of events) yield e;
			}
		}
	} finally {
		reader.releaseLock();
	}

	yield {
		type: "done",
		fullText: state.fullText,
		allComponents: state.allComponents,
		toolCalls: state.toolCalls,
		stopReason: state.stopReason,
	};
}

// ---------------------------------------------------------------------------
// stripAnalysis
// ---------------------------------------------------------------------------

export function stripAnalysis(text: string): string {
	return text.replace(/<analysis>[\s\S]*?<\/analysis>/g, "").trim();
}

// ---------------------------------------------------------------------------
// chat
// ---------------------------------------------------------------------------

const MAX_TOOL_TURNS = 5;

export async function* chat(
	ctx: ServiceContext,
	studentId: string,
	messages: Array<{ role: "user" | "assistant"; content: string | AnthropicContentBlock[] }>,
	dynamicContext: string,
): AsyncGenerator<TeacherEvent> {
	const systemBlocks: AnthropicSystemBlock[] = [
		{
			type: "text",
			text: UNIFIED_TEACHER_SYSTEM,
			cache_control: { type: "ephemeral" },
		},
		...(dynamicContext.trim()
			? [{ type: "text" as const, text: dynamicContext }]
			: []),
	];

	const processToolFn: ProcessToolFn = async (name, input) => {
		return processToolUse(ctx, studentId, name, input);
	};

	let currentMessages = messages as Array<{ role: "user" | "assistant"; content: string | AnthropicContentBlock[] }>;
	let accumulatedComponents: InlineComponent[] = [];

	for (let turn = 0; turn < MAX_TOOL_TURNS; turn++) {
		const stream = await callAnthropicStream(ctx.env, {
			model: "claude-sonnet-4-20250514",
			max_tokens: 2048,
			system: systemBlocks,
			messages: currentMessages,
			tools: getAnthropicToolSchemas(),
			tool_choice: { type: "auto" },
		});

		let doneEvent: TeacherEvent | null = null;

		for await (const event of parseAnthropicStream(stream, processToolFn)) {
			if (event.type === "done") {
				doneEvent = event;
			} else {
				yield event;
			}
		}

		if (!doneEvent || doneEvent.type !== "done") break;

		accumulatedComponents.push(...doneEvent.allComponents);

		// If no tool calls or stop_reason is not tool_use, this is the final turn.
		// Only the final turn's text is persisted — intermediate tool-calling turns
		// emit deltas for streaming UX but their narration is cleared by the frontend.
		if (doneEvent.toolCalls.length === 0 || doneEvent.stopReason !== "tool_use") {
			console.log(
				JSON.stringify({
					level: "info",
					message: "chat stream complete",
					turn: turn + 1,
					toolCallsThisTurn: doneEvent.toolCalls.length,
					totalComponents: accumulatedComponents.length,
					stopReason: doneEvent.stopReason,
					fullTextLength: doneEvent.fullText.length,
				}),
			);
			yield {
				type: "done",
				fullText: doneEvent.fullText,
				allComponents: accumulatedComponents,
				toolCalls: doneEvent.toolCalls,
				stopReason: doneEvent.stopReason,
			};
			return;
		}

		// Build continuation messages: assistant response + tool results
		const assistantContent: AnthropicContentBlock[] = [];
		if (doneEvent.fullText) {
			assistantContent.push({ type: "text", text: doneEvent.fullText });
		}
		for (const tc of doneEvent.toolCalls) {
			assistantContent.push({
				type: "tool_use",
				id: tc.id,
				name: tc.name,
				input: tc.input,
			});
		}

		const GENERIC_TOOL_ERROR =
			"Tool call failed validation. For piece_id, pass through the exact slug returned by search_catalog (e.g. 'chopin.ballades.1'); do not transform or invent it. Check all required fields and try again.";

		const toolResultContent: AnthropicContentBlock[] = doneEvent.toolCalls.map(
			(tc) => {
				if (tc.result.isError) {
					return {
						type: "tool_result" as const,
						tool_use_id: tc.id,
						is_error: true,
						content: tc.result.errorMessage ?? GENERIC_TOOL_ERROR,
					};
				}
				return {
					type: "tool_result" as const,
					tool_use_id: tc.id,
					content: JSON.stringify(tc.result.componentsJson),
				};
			},
		);

		currentMessages = [
			...currentMessages,
			{ role: "assistant", content: assistantContent },
			{ role: "user", content: toolResultContent },
		];

		console.log(
			JSON.stringify({
				level: "info",
				message: "chat tool continuation",
				turn: turn + 1,
				toolCount: doneEvent.toolCalls.length,
				toolNames: doneEvent.toolCalls.map((tc) => tc.name),
			}),
		);
	}

	// All tool turns exhausted — force one more call with tool_choice: none so the
	// model synthesizes a text response from the accumulated tool results.
	try {
		const forcedStream = await callAnthropicStream(ctx.env, {
			model: "claude-sonnet-4-20250514",
			max_tokens: 2048,
			system: systemBlocks,
			messages: currentMessages,
			tools: getAnthropicToolSchemas(),
			tool_choice: { type: "none" },
		});

		let forcedDone: TeacherEvent | null = null;

		for await (const event of parseAnthropicStream(forcedStream, processToolFn)) {
			if (event.type === "done") {
				forcedDone = event;
			} else {
				yield event;
			}
		}

		if (forcedDone && forcedDone.type === "done" && forcedDone.fullText) {
			yield {
				type: "done",
				fullText: forcedDone.fullText,
				allComponents: accumulatedComponents,
				toolCalls: [],
				stopReason: "forced_text_after_max_turns",
			};
			return;
		}
	} catch (err) {
		console.error(
			JSON.stringify({
				level: "error",
				message: "forced final call failed after max tool turns",
				error: err instanceof Error ? err.message : String(err),
				stack: err instanceof Error ? err.stack : undefined,
			}),
		);
		Sentry.captureException(err, {
			tags: { service: "teacher", operation: "forced_final_call" },
		});
	}

	yield {
		type: "done",
		fullText: "I had trouble putting that together — could you ask again?",
		allComponents: accumulatedComponents,
		toolCalls: [],
		stopReason: "max_tool_turns",
	};
}

// ---------------------------------------------------------------------------
// synthesize
// ---------------------------------------------------------------------------

const SYNTHESIS_TIMEOUT_MS = 25_000;

export async function synthesize(
	ctx: ServiceContext,
	input: SynthesisInput,
): Promise<TeacherResponse> {
	const controller = new AbortController();
	const timeoutId = setTimeout(() => controller.abort(), SYNTHESIS_TIMEOUT_MS);

	try {
		const memoryContext = await buildMemoryContext(ctx, input.studentId);

		const synthesisFraming = buildSynthesisFraming(
			input.sessionDurationMs,
			input.practicePattern,
			input.topMoments,
			input.drillingRecords,
			input.pieceMetadata,
			memoryContext,
		);

		const systemBlocks: AnthropicSystemBlock[] = [
			{
				type: "text",
				text: UNIFIED_TEACHER_SYSTEM,
				cache_control: { type: "ephemeral" },
			},
			{ type: "text", text: synthesisFraming },
		];

		const response = await callAnthropic(ctx.env, {
			model: "claude-sonnet-4-20250514",
			max_tokens: 2048,
			system: systemBlocks,
			messages: [
				{ role: "user", content: "Please provide your session synthesis." },
			],
			tools: getAnthropicToolSchemas(),
			tool_choice: { type: "auto" },
		});

		let rawText = "";
		const toolResults: ToolResult[] = [];

		for (const block of response.content) {
			if (block.type === "text" && block.text) {
				rawText += block.text;
			} else if (block.type === "tool_use" && block.name) {
				const result = await processToolUse(
					ctx,
					input.studentId,
					block.name,
					block.input,
				);
				toolResults.push(result);
			}
		}

		const text = stripAnalysis(rawText);

		console.log(
			JSON.stringify({
				level: "info",
				message: "synthesis complete",
				studentId: input.studentId,
				conversationId: input.conversationId,
				toolCount: toolResults.length,
				textLength: text.length,
			}),
		);

		return { text, toolResults };
	} catch (err) {
		if (err instanceof Error && err.name === "AbortError") {
			throw new InferenceError("Synthesis timed out after 25 seconds");
		}
		throw err;
	} finally {
		clearTimeout(timeoutId);
	}
}
