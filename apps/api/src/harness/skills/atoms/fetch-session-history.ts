import type { ToolDefinition } from "../../loop/types";

export type SessionHistory = {
	sessions: {
		session_id: string;
		created_at: number;
		synthesis: unknown;
		diagnoses: unknown[];
	}[];
};

export const fetchSessionHistory: ToolDefinition = {
	name: "fetch-session-history",
	description:
		"Filters and orders pre-materialized session records within a date window. Input sessions are from the digest (already fetched). Returns sessions in descending created_at order (most recent first).",
	input_schema: {
		type: "object",
		properties: {
			sessions: {
				type: "array",
				items: {
					type: "object",
					properties: {
						session_id: { type: "string" },
						created_at: {
							type: "number",
							description: "Unix epoch milliseconds",
						},
						synthesis: {},
						diagnoses: { type: "array" },
					},
					required: ["session_id", "created_at", "synthesis", "diagnoses"],
				},
				description:
					"Pre-materialized session records from digest. Caller provides all candidate sessions.",
			},
			window_days: {
				type: "number",
				minimum: 1,
				description: "Number of days to look back from now_ms",
			},
			now_ms: {
				type: "number",
				description: "Current time in Unix epoch milliseconds",
			},
		},
		required: ["sessions", "window_days", "now_ms"],
	},
	invoke: async (input: unknown): Promise<SessionHistory> => {
		const { sessions, window_days, now_ms } = input as {
			sessions: {
				session_id: string;
				created_at: number;
				synthesis: unknown;
				diagnoses: unknown[];
			}[];
			window_days: number;
			now_ms: number;
		};
		const cutoff = now_ms - window_days * 86_400_000;
		const filtered = sessions
			.filter((s) => s.created_at >= cutoff)
			.sort((a, b) => b.created_at - a.created_at);
		return { sessions: filtered };
	},
};
