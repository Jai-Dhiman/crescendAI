import { InferenceError } from "../../lib/errors";
import type { PhaseContext } from "./types";

/**
 * V6 stub: pass-through. Future versions redact student names from voice prompts.
 */
export function redactPii<T>(req: T): T {
	return req;
}

/**
 * V8a: accepts tool name and context for future permission gating. Currently pass-through.
 */
export async function wrapToolCall(
	_toolName: string,
	_ctx: PhaseContext,
	invoke: () => Promise<unknown>,
): Promise<unknown> {
	return invoke();
}

const RETRY_DELAY_MS = 100;

/**
 * Retry wrapper. Retries exactly once on InferenceError after a short delay.
 * Non-InferenceError exceptions propagate immediately.
 */
export async function withRetries<T>(fn: () => Promise<T>): Promise<T> {
	try {
		return await fn();
	} catch (err) {
		if (!(err instanceof InferenceError)) throw err;
		await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
		return fn();
	}
}

/**
 * V8+ replaces with a real reviewer agent. V6: log a breadcrumb on 10% sample.
 * `sample` is injectable so tests are deterministic.
 */
export function reviewArtifact(artifact: unknown, sample: () => boolean): void {
	if (!sample()) return;
	console.log(
		JSON.stringify({
			level: "info",
			fn: "reviewArtifact",
			message: "sampled",
			artifactPreview:
				typeof artifact === "object" && artifact !== null
					? Object.keys(artifact as Record<string, unknown>)
					: String(artifact),
		}),
	);
}
