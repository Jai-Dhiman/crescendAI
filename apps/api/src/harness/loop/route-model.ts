// apps/api/src/harness/loop/route-model.ts
import type { Bindings } from "../../lib/types";

export type TaskKind = "phase1_analysis" | "phase2_voice";

export interface ModelClient {
  provider: "anthropic" | "workers-ai";
  model: string;
}

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
  return WORKERS_AI_CLIENT;
}
