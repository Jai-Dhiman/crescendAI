export type TaskKind = "phase1_analysis" | "phase2_voice";

export interface GatewayClient {
	model: string;
}

// All teacher traffic routes through the single authenticated AI Gateway
// (env.AI_GATEWAY_ENDPOINT); the provider is encoded by the request path.
const SONNET_TEACHER: GatewayClient = {
	model: "claude-sonnet-4-20250514",
};

export function routeModel(_kind: TaskKind): GatewayClient {
	return SONNET_TEACHER;
}
