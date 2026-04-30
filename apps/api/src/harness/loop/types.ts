import type { ZodTypeAny } from "zod";

export type HookKind =
	| "OnStop"
	| "OnPieceDetected"
	| "OnBarRegression"
	| "OnSessionEnd"
	| "OnWeeklyReview"
	| "OnChatMessage";

export interface ToolDefinition {
	name: string;
	description: string;
	input_schema: Record<string, unknown>;
	invoke: (input: unknown, ctx?: PhaseContext) => Promise<unknown>;
}

export interface CompoundBinding {
	compoundName: string;
	procedurePrompt: string;
	tools: ToolDefinition[];
	mode: "streaming" | "buffered";
	phases: 1 | 2;
	artifactSchema?: ZodTypeAny;
	artifactToolName?: string;
}

export type Phase2Binding = CompoundBinding & {
	artifactSchema: ZodTypeAny;
	artifactToolName: string;
};

export type Phase1Event =
	| { type: "phase1_tool_call"; id: string; tool: string; input: unknown }
	| {
			type: "phase1_tool_result";
			id: string;
			tool: string;
			ok: true;
			output: unknown;
	  }
	| {
			type: "phase1_tool_result";
			id: string;
			tool: string;
			ok: false;
			error: string;
	  }
	| { type: "phase1_done"; toolCallCount: number; turnCount: number };

export type HookEvent<TArtifact> =
	| Phase1Event
	| { type: "phase2_started" }
	| { type: "artifact"; value: TArtifact }
	| { type: "validation_error"; raw: unknown; zodError: string }
	| { type: "phase_error"; phase: 1 | 2; error: string };

export interface HookContext {
	env: import("../../lib/types").Bindings;
	studentId: string;
	sessionId: string;
	conversationId: string | null;
	digest: Record<string, unknown>;
	waitUntil: (p: Promise<unknown>) => void;
	pieceId?: string;
	trigger?: "chat" | "synthesis";
}

export interface PhaseContext extends HookContext {
	turnCap: number;
}
