import * as Sentry from "@sentry/cloudflare";
import { InferenceError } from "../lib/errors";
import type { ServiceContext } from "../lib/types";
import { runHook } from "../harness/loop/runHook";
import { runStreamingHook } from "../harness/loop/runStreamingHook";
import { buildGroundedDigest } from "../harness/loop/grounded-digest";
import type {
	CompoundBinding,
	HookContext,
	HookEvent,
	PhaseContext,
} from "../harness/loop/types";
import type { SynthesisArtifact } from "../harness/artifacts/synthesis";
import {
	type AnthropicContentBlock,
	type AnthropicSystemBlock,
	callAnthropic,
	callAnthropicStream,
	callWorkersAIStream,
} from "./llm";
import { routeModel } from "../harness/loop/route-model";
import type { AnthropicChatRequest } from "../harness/loop/tool-format";
import { buildMemoryContext } from "./memory";
import { buildSynthesisFraming, UNIFIED_TEACHER_SYSTEM } from "./prompts";
import * as segmentLoopsService from "./segment-loops";
import {
	getAnthropicToolSchemas,
	type InlineComponent,
	processToolUse,
	type ToolResult,
	toolResultModelContent,
} from "./tool-processor";
import { assignSegmentLoopAtom } from "../harness/atoms/assign-segment-loop";

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

// Re-export EnrichedChunk from session-brain to avoid interface duplication.
// If toEnrichedChunk gains a new field, SynthesisInput automatically reflects it.
export type { EnrichedChunk as EnrichedChunkDigest } from '../do/session-brain'
import type { EnrichedChunk } from '../do/session-brain'

export interface SessionHistoryRecord {
	sessionId: string
	startedAt: string
	synthesis: string | null
}

export interface PastDiagnosisRecord {
	id: string
	sessionId: string
	primaryDimension: string
	barRangeStart: number | null
	barRangeEnd: number | null
	artifactJson: unknown
	createdAt: string
	pieceId: string | null
}

export interface SynthesisInput {
	studentId: string;
	conversationId: string | null;
	sessionDurationMs: number;
	practicePattern: string;
	topMoments: unknown[];
	drillingRecords: unknown[];
	pieceMetadata: { composer?: string; title?: string } | null;
	enrichedChunks: EnrichedChunk[];
	baselines: Record<string, number> | null;
	sessionHistory: SessionHistoryRecord[];
	pastDiagnoses: PastDiagnosisRecord[];
	pieceId?: string | null;
	referenceMode?: "within_session" | null;
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
		state.toolCalls.push({
			id: block.id,
			name: block.name,
			input: toolInput,
			result,
		});
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
			const flushed = state.pendingTextDeltas.map((text) => ({
				type: "delta" as const,
				text,
			}));
			state.pendingTextDeltas = [];
			return flushed;
		}
		state.pendingTextDeltas = [];
		return [];
	}

	// message_stop: flush any remaining buffered deltas (covers streams with no message_delta)
	if (event === "message_stop") {
		if (!state.hasToolUseThisTurn && state.pendingTextDeltas.length > 0) {
			const flushed = state.pendingTextDeltas.map((text) => ({
				type: "delta" as const,
				text,
			}));
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
// parseOpenAIStream
// ---------------------------------------------------------------------------

interface OpenAIStreamToolCallAccumulator {
	id: string;
	name: string;
	argumentsAccumulator: string;
}

export async function* parseOpenAIStream(
	stream: ReadableStream,
	processToolFn: ProcessToolFn,
): AsyncGenerator<TeacherEvent> {
	const decoder = new TextDecoder();
	const reader = stream.getReader();

	const toolAccumulators = new Map<number, OpenAIStreamToolCallAccumulator>();
	// Tracks which tool indices have already emitted a tool_start, so each tool emits
	// exactly one regardless of streamed-fragment vs single-final-chunk delivery (and
	// when the tool name arrives in a later fragment than the index first appeared).
	const startedTools = new Set<number>();
	const state = {
		fullText: "",
		allComponents: [] as InlineComponent[],
		toolCalls: [] as ToolCallRecord[],
		stopReason: "stop",
		hasToolCallThisTurn: false,
	};

	let textBuffer = "";

	function* parseLines(raw: string): Generator<Record<string, unknown>> {
		const messages = raw.split(/\n\n/);
		for (const message of messages) {
			for (const line of message.split("\n")) {
				const trimmed = line.trim();
				if (!trimmed.startsWith("data:")) continue;
				const payload = trimmed.slice("data:".length).trim();
				if (payload === "[DONE]") continue;
				let parsed: Record<string, unknown>;
				try {
					parsed = JSON.parse(payload) as Record<string, unknown>;
				} catch (err) {
					console.error(
						JSON.stringify({
							level: "error",
							message: "parseOpenAIStream: failed to parse SSE data",
							payload,
							error: err instanceof Error ? err.message : String(err),
						}),
					);
					Sentry.captureException(err, {
						tags: { service: "teacher", operation: "openai_sse_parse" },
						extra: { payload },
					});
					continue;
				}
				yield parsed;
			}
		}
	}

	async function* processChunk(
		parsed: Record<string, unknown>,
	): AsyncGenerator<TeacherEvent> {
		const choices = parsed["choices"] as
			| Array<Record<string, unknown>>
			| undefined;
		if (!choices || choices.length === 0) return;

		const choice = choices[0];
		const delta = choice["delta"] as Record<string, unknown> | undefined;
		const finishReason = choice["finish_reason"] as string | null | undefined;

		if (delta) {
			const content = delta["content"] as string | null | undefined;
			if (content && !state.hasToolCallThisTurn) {
				state.fullText += content;
				yield { type: "delta", text: content };
			}

			const toolCalls = delta["tool_calls"] as
				| Array<{
						index: number;
						id?: string;
						type?: string;
						function?: { name?: string; arguments?: string };
				  }>
				| undefined;

			if (toolCalls) {
				for (const tc of toolCalls) {
					const idx = tc.index;
					if (!toolAccumulators.has(idx)) {
						toolAccumulators.set(idx, {
							id: tc.id ?? "",
							name: tc.function?.name ?? "",
							argumentsAccumulator: "",
						});
						state.hasToolCallThisTurn = true;
						const accum = toolAccumulators.get(idx)!;
						if (accum.name && !startedTools.has(idx)) {
							startedTools.add(idx);
							yield { type: "tool_start", name: accum.name };
						}
					} else {
						const accum = toolAccumulators.get(idx)!;
						if (tc.id) accum.id = tc.id;
						if (tc.function?.name) accum.name = tc.function.name;
					}
					if (tc.function?.arguments) {
						toolAccumulators.get(idx)!.argumentsAccumulator +=
							tc.function.arguments;
					}
				}
			}
		}

		if (finishReason) {
			state.stopReason = finishReason;

			if (finishReason === "tool_calls" || toolAccumulators.size > 0) {
				const indices = Array.from(toolAccumulators.keys()).sort(
					(a, b) => a - b,
				);
				for (const idx of indices) {
					const accum = toolAccumulators.get(idx)!;

					state.hasToolCallThisTurn = true;
					if (!startedTools.has(idx)) {
						startedTools.add(idx);
						yield { type: "tool_start", name: accum.name };
					}

					let toolInput: unknown;
					try {
						toolInput = JSON.parse(accum.argumentsAccumulator);
					} catch (err) {
						const parseMsg =
							err instanceof Error ? err.message : String(err);
						console.error(
							JSON.stringify({
								level: "error",
								message: "parseOpenAIStream: failed to parse tool arguments",
								toolName: accum.name,
								accumulated: accum.argumentsAccumulator,
								error: parseMsg,
							}),
						);
						yield {
							type: "tool_error",
							name: accum.name,
							message: `The model sent malformed input for ${accum.name}: ${parseMsg}`,
						};
						continue;
					}

					const result = await processToolFn(accum.name, toolInput);
					state.toolCalls.push({
						id: accum.id,
						name: accum.name,
						input: toolInput,
						result,
					});
					if (!result.isError) {
						state.allComponents.push(...result.componentsJson);
						yield {
							type: "tool_result",
							name: result.name,
							componentsJson: result.componentsJson,
						};
					} else {
						yield {
							type: "tool_error",
							name: result.name,
							message: result.errorMessage ?? "Tool call failed.",
						};
					}
				}
				toolAccumulators.clear();
			}
		}
	}

	try {
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			textBuffer += decoder.decode(value, { stream: true });

			const lastDoubleNewline = textBuffer.lastIndexOf("\n\n");
			if (lastDoubleNewline === -1) continue;

			const toProcess = textBuffer.slice(0, lastDoubleNewline + 2);
			textBuffer = textBuffer.slice(lastDoubleNewline + 2);

			for (const parsed of parseLines(toProcess)) {
				for await (const ev of processChunk(parsed)) {
					yield ev;
				}
			}
		}

		if (textBuffer.trim()) {
			for (const parsed of parseLines(textBuffer)) {
				for await (const ev of processChunk(parsed)) {
					yield ev;
				}
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
		// Normalize OpenAI's "tool_calls" finish_reason to Anthropic's "tool_use" so the
		// shared runPhase1Streaming continuation guard (stopReason === "tool_use") works
		// identically on both provider paths. Without this, glm tool turns abort early.
		stopReason: state.stopReason === "tool_calls" ? "tool_use" : state.stopReason,
	};
}

// ---------------------------------------------------------------------------
// stripAnalysis
// ---------------------------------------------------------------------------

export function stripAnalysis(text: string): string {
	return text.replace(/<analysis>[\s\S]*?<\/analysis>/g, "").trim();
}


// ---------------------------------------------------------------------------
// runPhase1Streaming
// ---------------------------------------------------------------------------

export async function* runPhase1Streaming(
	ctx: PhaseContext,
	binding: CompoundBinding,
	systemBlocks: AnthropicSystemBlock[],
	initialMessages: Array<{
		role: "user" | "assistant";
		content: string | AnthropicContentBlock[];
	}>,
	processToolFn: ProcessToolFn,
): AsyncGenerator<TeacherEvent> {
	const toolSchemas = binding.tools.map((t) => ({
		name: t.name,
		description: t.description,
		input_schema: t.input_schema,
	}));

	let currentMessages = initialMessages;
	const accumulatedComponents: InlineComponent[] = [];

	for (let turn = 0; turn < ctx.turnCap; turn++) {
		const client = routeModel("phase1_analysis", ctx.env);
		const chatBody: AnthropicChatRequest = {
			model: client.model,
			max_tokens: 2048,
			system: systemBlocks,
			messages: currentMessages,
			tools: toolSchemas,
			tool_choice: { type: "auto" },
		};
		const stream =
			client.provider === "workers-ai"
				? await callWorkersAIStream(ctx.env, chatBody)
				: await callAnthropicStream(ctx.env, chatBody);

		let doneEvent: TeacherEvent | null = null;
		const parseStream =
			client.provider === "workers-ai"
				? parseOpenAIStream(stream, processToolFn)
				: parseAnthropicStream(stream, processToolFn);
		for await (const event of parseStream) {
			if (event.type === "done") {
				doneEvent = event;
			} else {
				yield event;
			}
		}

		if (!doneEvent || doneEvent.type !== "done") {
			console.error(
				JSON.stringify({
					level: "error",
					message:
						"runPhase1Streaming: stream parser did not yield done event",
					turn,
				}),
			);
			break;
		}

		accumulatedComponents.push(...doneEvent.allComponents);

		if (
			doneEvent.toolCalls.length === 0 ||
			doneEvent.stopReason !== "tool_use"
		) {
			yield { ...doneEvent, allComponents: accumulatedComponents };
			return;
		}

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
					content: toolResultModelContent(tc.result),
				};
			},
		);

		currentMessages = [
			...currentMessages,
			{ role: "assistant" as const, content: assistantContent },
			{ role: "user" as const, content: toolResultContent },
		];

		console.log(
			JSON.stringify({
				level: "info",
				message: "streaming chat tool continuation",
				turn: turn + 1,
				toolCount: doneEvent.toolCalls.length,
				toolNames: doneEvent.toolCalls.map((tc) => tc.name),
			}),
		);
	}

	// Turn cap exhausted — force a text response with tool_choice: none
	try {
		const forcedClient = routeModel("phase1_analysis", ctx.env);
		const forcedBody: AnthropicChatRequest = {
			model: forcedClient.model,
			max_tokens: 2048,
			system: systemBlocks,
			messages: currentMessages,
			tools: toolSchemas,
			tool_choice: { type: "none" },
		};
		const forcedStream =
			forcedClient.provider === "workers-ai"
				? await callWorkersAIStream(ctx.env, forcedBody)
				: await callAnthropicStream(ctx.env, forcedBody);

		let forcedDone: TeacherEvent | null = null;
		const parseForcedStream =
			forcedClient.provider === "workers-ai"
				? parseOpenAIStream(forcedStream, processToolFn)
				: parseAnthropicStream(forcedStream, processToolFn);
		for await (const event of parseForcedStream) {
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
			tags: { service: "teacher", operation: "streaming_forced_final_call" },
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
// chatV6
// ---------------------------------------------------------------------------

export async function* chatV6(
	ctx: ServiceContext,
	studentId: string,
	messages: Array<{
		role: "user" | "assistant";
		content: string | AnthropicContentBlock[];
	}>,
	dynamicContext: string,
	pieceId?: string,
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
		if (name === "assign_segment_loop") {
			try {
				const atomCtx: PhaseContext = {
					env: ctx.env,
					studentId,
					sessionId: "",
					conversationId: null,
					digest: {},
					waitUntil: () => {},
					pieceId,
					trigger: "chat",
					turnCap: 5,
				};
				const artifact = await assignSegmentLoopAtom(atomCtx, input);
				const component = segmentLoopsService.toLoopComponent(artifact);
				return { name, componentsJson: [component], isError: false };
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				return { name, componentsJson: [], isError: true, errorMessage: message };
			}
		}
		return processToolUse(ctx, studentId, name, input);
	};

	const hookCtx: HookContext = {
		env: ctx.env,
		studentId,
		sessionId: "",
		conversationId: null,
		digest: {},
		waitUntil: (_p: Promise<unknown>) => {},
		pieceId,
		trigger: "chat",
	};

	yield* runStreamingHook(
		"OnChatMessage",
		hookCtx,
		processToolFn,
		systemBlocks,
		messages,
	);
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

		const composer =
			(input.pieceMetadata as { composer?: string } | null | undefined)
				?.composer ?? "";
		const synthesisFraming = buildSynthesisFraming(
			input.sessionDurationMs,
			input.practicePattern,
			input.topMoments,
			input.drillingRecords,
			input.pieceMetadata,
			memoryContext,
			composer,
			input.referenceMode ?? null,
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

// ---------------------------------------------------------------------------
// synthesizeV6
// ---------------------------------------------------------------------------

const COHORT_TABLES: Record<string, { p: number; value: number }[]> = {
	dynamics:        [{ p: 25, value: 0.38 }, { p: 50, value: 0.55 }, { p: 75, value: 0.70 }, { p: 90, value: 0.82 }],
	timing:          [{ p: 25, value: 0.34 }, { p: 50, value: 0.48 }, { p: 75, value: 0.63 }, { p: 90, value: 0.77 }],
	pedaling:        [{ p: 25, value: 0.32 }, { p: 50, value: 0.46 }, { p: 75, value: 0.61 }, { p: 90, value: 0.75 }],
	articulation:    [{ p: 25, value: 0.37 }, { p: 50, value: 0.54 }, { p: 75, value: 0.68 }, { p: 90, value: 0.80 }],
	phrasing:        [{ p: 25, value: 0.36 }, { p: 50, value: 0.52 }, { p: 75, value: 0.66 }, { p: 90, value: 0.79 }],
	interpretation:  [{ p: 25, value: 0.35 }, { p: 50, value: 0.51 }, { p: 75, value: 0.65 }, { p: 90, value: 0.78 }],
}

/**
 * V6 adapter. Translates the legacy SynthesisInput shape into a HookContext
 * and yields the harness loop's event stream. Caller (DO) consumes events and
 * maps the final artifact to the existing WebSocket/persist pipeline.
 *
 * `sessionId` is passed alongside `input` because SynthesisInput does not carry
 * it today (legacy quirk: sessionId lives on the DO state, not in input).
 */
export async function* synthesizeV6(
	ctx: ServiceContext,
	input: SynthesisInput,
	sessionId: string,
	waitUntil?: (p: Promise<unknown>) => void,
): AsyncGenerator<HookEvent<SynthesisArtifact>> {
	const groundedDigest = await buildGroundedDigest(
		input,
		{ db: ctx.db, studentId: input.studentId },
		COHORT_TABLES,
	);

	const hookCtx: HookContext = {
		env: ctx.env,
		studentId: input.studentId,
		sessionId,
		conversationId: input.conversationId,
		digest: groundedDigest as unknown as Record<string, unknown>,
		waitUntil: waitUntil ?? ((_p: Promise<unknown>) => {}),
		pieceId: input.pieceId ?? undefined,
		trigger: "synthesis",
	};

	for await (const ev of runHook("OnSessionEnd", hookCtx)) {
		yield ev;
	}
}
