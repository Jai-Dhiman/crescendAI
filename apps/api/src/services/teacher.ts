import type { ServiceContext } from "../lib/types";
import { InferenceError } from "../lib/errors";
import { callAnthropic, callAnthropicStream, type AnthropicSystemBlock } from "./llm";
import { buildMemoryContext } from "./memory";
import { UNIFIED_TEACHER_SYSTEM, buildSynthesisFraming } from "./prompts";
import {
  processToolUse,
  getAnthropicToolSchemas,
  type ToolResult,
  type InlineComponent,
} from "./tool-processor";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type TeacherEvent =
  | { type: "delta"; text: string }
  | { type: "tool_result"; name: string; componentsJson: InlineComponent[] }
  | { type: "done"; fullText: string; allComponents: InlineComponent[] };

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
  name: string;
  jsonAccumulator: string;
}

type ContentBlock = TextBlock | ToolUseBlock;

// ---------------------------------------------------------------------------
// SSE line parser
// ---------------------------------------------------------------------------

function* parseSSELines(chunk: string): Generator<{ event: string; data: string }> {
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
// parseAnthropicStream
// ---------------------------------------------------------------------------

export async function* parseAnthropicStream(
  stream: ReadableStream,
  processToolFn: ProcessToolFn,
): AsyncGenerator<TeacherEvent> {
  const decoder = new TextDecoder();
  const reader = stream.getReader();

  const blocks = new Map<number, ContentBlock>();
  let fullText = "";
  const allComponents: InlineComponent[] = [];
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
        let parsed: Record<string, unknown>;
        try {
          parsed = JSON.parse(data) as Record<string, unknown>;
        } catch {
          console.log(JSON.stringify({
            level: "warn",
            message: "Failed to parse SSE data JSON",
            event,
            data,
          }));
          continue;
        }

        if (event === "content_block_start") {
          const index = parsed["index"] as number;
          const contentBlock = parsed["content_block"] as Record<string, unknown>;
          const blockType = contentBlock["type"] as string;

          if (blockType === "text") {
            blocks.set(index, { type: "text", textAccumulator: "" });
          } else if (blockType === "tool_use") {
            const name = contentBlock["name"] as string;
            blocks.set(index, { type: "tool_use", name, jsonAccumulator: "" });
          }
        } else if (event === "content_block_delta") {
          const index = parsed["index"] as number;
          const delta = parsed["delta"] as Record<string, unknown>;
          const deltaType = delta["type"] as string;
          const block = blocks.get(index);

          if (!block) continue;

          if (deltaType === "text_delta" && block.type === "text") {
            const text = delta["text"] as string;
            block.textAccumulator += text;
            fullText += text;
            yield { type: "delta", text };
          } else if (deltaType === "input_json_delta" && block.type === "tool_use") {
            const partialJson = delta["partial_json"] as string;
            block.jsonAccumulator += partialJson;
          }
        } else if (event === "content_block_stop") {
          const index = parsed["index"] as number;
          const block = blocks.get(index);

          if (!block || block.type !== "tool_use") continue;

          let toolInput: unknown;
          try {
            toolInput = JSON.parse(block.jsonAccumulator);
          } catch {
            console.log(JSON.stringify({
              level: "error",
              message: "Failed to parse tool_use JSON accumulator",
              toolName: block.name,
              accumulated: block.jsonAccumulator,
            }));
            continue;
          }

          const result = await processToolFn(block.name, toolInput);
          if (!result.isError) {
            allComponents.push(...result.componentsJson);
            yield { type: "tool_result", name: result.name, componentsJson: result.componentsJson };
          }
        }
        // message_start, message_delta, message_stop — no action needed
      }
    }

    // Flush any remaining buffer content
    if (textBuffer.trim()) {
      for (const { event, data } of parseSSELines(textBuffer)) {
        let parsed: Record<string, unknown>;
        try {
          parsed = JSON.parse(data) as Record<string, unknown>;
        } catch {
          continue;
        }

        if (event === "content_block_delta") {
          const index = parsed["index"] as number;
          const delta = parsed["delta"] as Record<string, unknown>;
          const deltaType = delta["type"] as string;
          const block = blocks.get(index);

          if (block && deltaType === "text_delta" && block.type === "text") {
            const text = delta["text"] as string;
            block.textAccumulator += text;
            fullText += text;
            yield { type: "delta", text };
          } else if (block && deltaType === "input_json_delta" && block.type === "tool_use") {
            block.jsonAccumulator += delta["partial_json"] as string;
          }
        } else if (event === "content_block_stop") {
          const index = parsed["index"] as number;
          const block = blocks.get(index);

          if (block && block.type === "tool_use") {
            let toolInput: unknown;
            try {
              toolInput = JSON.parse(block.jsonAccumulator);
            } catch {
              continue;
            }
            const result = await processToolFn(block.name, toolInput);
            if (!result.isError) {
              allComponents.push(...result.componentsJson);
              yield { type: "tool_result", name: result.name, componentsJson: result.componentsJson };
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  yield { type: "done", fullText, allComponents };
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

export async function* chat(
  ctx: ServiceContext,
  studentId: string,
  messages: Array<{ role: "user" | "assistant"; content: string }>,
  dynamicContext: string,
): AsyncGenerator<TeacherEvent> {
  const systemBlocks: AnthropicSystemBlock[] = [
    { type: "text", text: UNIFIED_TEACHER_SYSTEM, cache_control: { type: "ephemeral" } },
    ...(dynamicContext.trim()
      ? [{ type: "text" as const, text: dynamicContext }]
      : []),
  ];

  const stream = await callAnthropicStream(ctx.env, {
    model: "claude-sonnet-4-20250514",
    max_tokens: 2048,
    system: systemBlocks,
    messages,
    tools: getAnthropicToolSchemas(),
    tool_choice: { type: "auto" },
  });

  const processToolFn: ProcessToolFn = async (name, input) => {
    return processToolUse(ctx, studentId, name, input);
  };

  yield* parseAnthropicStream(stream, processToolFn);
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
      { type: "text", text: UNIFIED_TEACHER_SYSTEM, cache_control: { type: "ephemeral" } },
      { type: "text", text: synthesisFraming },
    ];

    const response = await callAnthropic(ctx.env, {
      model: "claude-sonnet-4-20250514",
      max_tokens: 2048,
      system: systemBlocks,
      messages: [{ role: "user", content: "Please provide your session synthesis." }],
      tools: getAnthropicToolSchemas(),
      tool_choice: { type: "auto" },
    });

    let rawText = "";
    const toolResults: ToolResult[] = [];

    for (const block of response.content) {
      if (block.type === "text" && block.text) {
        rawText += block.text;
      } else if (block.type === "tool_use" && block.name) {
        const result = await processToolUse(ctx, input.studentId, block.name, block.input);
        toolResults.push(result);
      }
    }

    const text = stripAnalysis(rawText);

    console.log(JSON.stringify({
      level: "info",
      message: "synthesis complete",
      studentId: input.studentId,
      conversationId: input.conversationId,
      toolCount: toolResults.length,
      textLength: text.length,
    }));

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
