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
