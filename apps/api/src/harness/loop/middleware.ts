/**
 * V6 stub: pass-through. Future versions redact student names from voice prompts.
 */
export function redactPii<T>(req: T): T {
	return req;
}

/**
 * V6 stub: pass-through. V8a fills in action-tool permission gating.
 */
export async function wrapToolCall<T>(invoke: () => Promise<T>): Promise<T> {
	return invoke();
}

/**
 * Retry wrapper. V6: 1 retry on InferenceError or network error, exponential backoff.
 * Initial form (Task 7): no retry — pass-through. Task 9 adds retry-on-error.
 */
export async function withRetries<T>(fn: () => Promise<T>): Promise<T> {
	return fn();
}
