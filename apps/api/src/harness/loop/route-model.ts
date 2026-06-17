// apps/api/src/harness/loop/route-model.ts

import type { Bindings } from "../../lib/types";
import type { ModelClient } from "./gateway-client";

export type { ModelClient } from "./gateway-client";

export type TaskKind = "phase1_analysis" | "phase2_voice";

const WORKERS_AI_CLIENT: ModelClient = {
	provider: "workers-ai",
	model: "@cf/qwen/qwen3-30b-a3b-fp8",
};

const ANTHROPIC_CLIENT: ModelClient = {
	provider: "anthropic",
	model: "claude-sonnet-4-20250514",
};

export function routeModel(_kind: TaskKind, env: Bindings): ModelClient {
	if (env.TEACHER_PROVIDER === "anthropic") {
		return ANTHROPIC_CLIENT;
	}
	if (env.TEACHER_MODEL) {
		return { provider: "workers-ai", model: env.TEACHER_MODEL };
	}
	return WORKERS_AI_CLIENT;
}
