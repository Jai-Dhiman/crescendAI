import { InferenceError } from "../lib/errors";
import type { Bindings } from "../lib/types";

interface LlmMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface AnthropicRequest {
  model: string;
  max_tokens: number;
  system?: string;
  messages: LlmMessage[];
  stream?: boolean;
  tools?: unknown[];
  tool_choice?: unknown;
}

interface AnthropicResponse {
  content: Array<{
    type: string;
    text?: string;
    id?: string;
    name?: string;
    input?: unknown;
  }>;
  stop_reason: string;
  usage: { input_tokens: number; output_tokens: number };
}

export async function callAnthropic(
  env: Bindings,
  request: AnthropicRequest,
): Promise<AnthropicResponse> {
  const url = `${env.AI_GATEWAY_TEACHER}/anthropic/v1/messages`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({ ...request, stream: false }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new InferenceError(
      `Anthropic request failed: ${res.status} ${text}`,
    );
  }

  return res.json() as Promise<AnthropicResponse>;
}

export async function callAnthropicStream(
  env: Bindings,
  request: AnthropicRequest,
): Promise<ReadableStream> {
  const url = `${env.AI_GATEWAY_TEACHER}/anthropic/v1/messages`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({ ...request, stream: true }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new InferenceError(
      `Anthropic stream request failed: ${res.status} ${text}`,
    );
  }

  if (!res.body) {
    throw new InferenceError("Anthropic stream response has no body");
  }

  return res.body;
}

export async function callGroq(
  env: Bindings,
  model: string,
  messages: LlmMessage[],
  maxTokens?: number,
): Promise<string> {
  const url = `${env.AI_GATEWAY_BACKGROUND}/groq/openai/v1/chat/completions`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model,
      messages,
      max_tokens: maxTokens ?? 1024,
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new InferenceError(`Groq request failed: ${res.status} ${text}`);
  }

  const data = (await res.json()) as {
    choices: Array<{ message: { content: string } }>;
  };

  return data.choices[0].message.content;
}
