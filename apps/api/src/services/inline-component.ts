// Shared shape for client-rendered inline components and the tool-result
// envelope that carries them. Split out of the now-deleted tool-processor.ts
// (chat-dispatch, #164 cleanup) because live callers only ever needed these
// two types, not the chat tool registry/dispatcher.

export interface InlineComponent {
	type: string;
	config: Record<string, unknown>;
}

export interface ToolResult {
	name: string;
	componentsJson: InlineComponent[];
	isError: boolean;
	errorMessage?: string;
	/**
	 * Distilled, model-facing prose. When set, the teacher model sees this
	 * string as the tool_result content instead of the raw componentsJson —
	 * the client still renders componentsJson.
	 */
	modelSummary?: string;
}
