import { ConfigError } from "../../lib/errors";
import type {
	AnthropicContentBlock,
	AnthropicSystemBlock,
} from "../../services/llm";
import type { TeacherEvent } from "../../services/teacher";
import { runPhase1Streaming } from "../../services/teacher";
import type { ToolResult } from "../../services/tool-processor";
import { getCompoundBinding } from "./compound-registry";
import type { HookContext, HookKind, PhaseContext } from "./types";

type ProcessToolFn = (name: string, input: unknown) => Promise<ToolResult>;

const DEFAULT_TURN_CAP = 5;

export async function* runStreamingHook(
	hook: HookKind,
	hookCtx: HookContext,
	processToolFn: ProcessToolFn,
	systemBlocks: AnthropicSystemBlock[],
	initialMessages: Array<{
		role: "user" | "assistant";
		content: string | AnthropicContentBlock[];
	}>,
): AsyncGenerator<TeacherEvent> {
	const binding = getCompoundBinding(hook);
	if (!binding) {
		throw new ConfigError(
			`runStreamingHook: no binding registered for hook "${hook}"`,
		);
	}
	if (binding.mode !== "streaming") {
		throw new ConfigError(
			`runStreamingHook: binding for "${hook}" has mode "${binding.mode}", expected "streaming"`,
		);
	}
	const phaseCtx: PhaseContext = {
		...hookCtx,
		turnCap: DEFAULT_TURN_CAP,
	};
	yield* runPhase1Streaming(
		phaseCtx,
		binding,
		systemBlocks,
		initialMessages,
		processToolFn,
	);
}
