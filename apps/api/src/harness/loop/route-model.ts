export type TaskKind = "phase1_analysis" | "phase2_voice";

export interface GatewayClient {
	model: string;
	gatewayUrlVar: "AI_GATEWAY_TEACHER" | "AI_GATEWAY_BACKGROUND";
}

const SONNET_TEACHER: GatewayClient = {
	model: "claude-sonnet-4-20250514",
	gatewayUrlVar: "AI_GATEWAY_TEACHER",
};

export function routeModel(_kind: TaskKind): GatewayClient {
	return SONNET_TEACHER;
}
