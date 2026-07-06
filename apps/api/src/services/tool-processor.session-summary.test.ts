import { describe, expect, it } from "vitest";
import type { ServiceContext } from "../lib/types";
import {
	type InlineComponent,
	processToolUse,
	summarizeSessionData,
	type ToolResult,
	toolResultModelContent,
} from "./tool-processor";

// ---------------------------------------------------------------------------
// Context-hygiene guard: the model-facing summary must never carry a raw score
// ---------------------------------------------------------------------------

// Distinctive score values that would be trivial to spot if they leaked verbatim.
const DISTINCT_A = 0.7314159;
const DISTINCT_B = 0.2718281;

function sessionDataComponent(
	queryType: string,
	data: unknown,
): InlineComponent[] {
	return [
		{ type: "session_data", config: { queryType, studentId: "s1", data } },
	];
}

describe("summarizeSessionData — dimension_history", () => {
	const components = sessionDataComponent("dimension_history", [
		{
			id: "o1",
			dimension: "dynamics",
			dimensionScore: 0.42,
			observationText: "Dynamics were flat through the development.",
			framing: "Earlier: dynamics were flat.",
			createdAt: new Date("2026-06-01T00:00:00Z"),
			sessionId: "sess-1",
		},
		{
			id: "o2",
			dimension: "dynamics",
			dimensionScore: DISTINCT_A,
			observationText: "Much more shape in the phrasing now.",
			framing: "Your dynamic shaping is opening up nicely.",
			createdAt: new Date("2026-07-01T00:00:00Z"),
			sessionId: "sess-2",
		},
	]);

	it("names the dimension and reports an improving trend in prose", () => {
		const summary = summarizeSessionData(components);
		expect(summary).toContain("dynamics");
		expect(summary.toLowerCase()).toContain("improv");
	});

	it("surfaces the most recent qualitative framing", () => {
		const summary = summarizeSessionData(components);
		expect(summary).toContain("Your dynamic shaping is opening up nicely.");
	});

	it("never emits a raw dimension score value", () => {
		const summary = summarizeSessionData(components);
		expect(summary).not.toContain(String(DISTINCT_A));
		expect(summary).not.toContain("0.42");
	});
});

describe("summarizeSessionData — recent_sessions", () => {
	const components = sessionDataComponent("recent_sessions", [
		{
			id: "sess-old",
			startedAt: new Date("2026-06-01T00:00:00Z"),
			endedAt: new Date("2026-06-01T01:00:00Z"),
			avgDynamics: 0.3,
			avgTiming: 0.9,
			avgPedaling: 0.2,
			avgArticulation: 0.5,
			avgPhrasing: 0.6,
			avgInterpretation: 0.55,
		},
		{
			id: "sess-new",
			startedAt: new Date("2026-07-01T00:00:00Z"),
			endedAt: new Date("2026-07-01T01:00:00Z"),
			avgDynamics: DISTINCT_A, // now the strongest
			avgTiming: 0.91,
			avgPedaling: DISTINCT_B, // still the weakest
			avgArticulation: 0.5,
			avgPhrasing: 0.6,
			avgInterpretation: 0.55,
		},
	]);

	it("ranks a relatively strong and a relatively weak dimension without numbers", () => {
		const summary = summarizeSessionData(components);
		// timing (0.91) is strongest, pedaling (0.27) is weakest in the latest session
		expect(summary).toContain("timing");
		expect(summary).toContain("pedaling");
		expect(summary).not.toContain(String(DISTINCT_A));
		expect(summary).not.toContain(String(DISTINCT_B));
		expect(summary).not.toContain("0.91");
	});
});

describe("summarizeSessionData — session_detail", () => {
	const components = sessionDataComponent("session_detail", {
		id: "sess-1",
		startedAt: new Date("2026-07-01T00:00:00Z"),
		endedAt: new Date("2026-07-01T01:00:00Z"),
		avgDynamics: DISTINCT_A, // strongest
		avgTiming: 0.6,
		avgPedaling: DISTINCT_B, // weakest
		avgArticulation: 0.5,
		avgPhrasing: 0.55,
		avgInterpretation: 0.58,
	});

	it("identifies strongest and weakest areas as prose, no raw scores", () => {
		const summary = summarizeSessionData(components);
		expect(summary).toContain("dynamics"); // strongest
		expect(summary).toContain("pedaling"); // weakest
		expect(summary).not.toContain(String(DISTINCT_A));
		expect(summary).not.toContain(String(DISTINCT_B));
	});
});

describe("summarizeSessionData — degenerate input", () => {
	it("handles null data (no matching session) gracefully", () => {
		const summary = summarizeSessionData(
			sessionDataComponent("session_detail", null),
		);
		expect(summary.length).toBeGreaterThan(0);
		expect(summary.toLowerCase()).toContain("no session data");
	});

	it("handles empty rows gracefully", () => {
		const summary = summarizeSessionData(
			sessionDataComponent("dimension_history", []),
		);
		expect(summary.length).toBeGreaterThan(0);
	});
});

// ---------------------------------------------------------------------------
// toolResultModelContent — the model-facing content selector
// ---------------------------------------------------------------------------

describe("toolResultModelContent", () => {
	it("uses modelSummary when present (never the raw component JSON)", () => {
		const result: ToolResult = {
			name: "show_session_data",
			componentsJson: [
				{
					type: "session_data",
					config: { data: [{ dimensionScore: DISTINCT_A }] },
				},
			],
			isError: false,
			modelSummary: "Your dynamics are improving.",
		};
		const content = toolResultModelContent(result);
		expect(content).toBe("Your dynamics are improving.");
		expect(content).not.toContain(String(DISTINCT_A));
	});

	it("falls back to component JSON for tools without a summary", () => {
		const result: ToolResult = {
			name: "search_catalog",
			componentsJson: [
				{ type: "catalog", config: { pieceId: "chopin.ballades.1" } },
			],
			isError: false,
		};
		const content = toolResultModelContent(result);
		expect(content).toContain("chopin.ballades.1");
	});
});

// ---------------------------------------------------------------------------
// processToolUse wiring: show_session_data attaches modelSummary,
// keeps raw numbers in componentsJson for the client chart
// ---------------------------------------------------------------------------

function mockDbReturning(rows: unknown): ServiceContext {
	const chain: Record<string, unknown> = {};
	for (const m of ["select", "from", "where", "orderBy"]) {
		chain[m] = () => chain;
	}
	chain.limit = () => Promise.resolve(rows);
	return { db: chain, env: {} } as unknown as ServiceContext;
}

describe("processToolUse — show_session_data hygiene wiring", () => {
	it("returns a distilled modelSummary while componentsJson keeps raw scores", async () => {
		const rows = [
			{
				id: "o1",
				dimension: "dynamics",
				dimensionScore: DISTINCT_A,
				observationText: "Nice shaping.",
				framing: "Your dynamic shaping is opening up.",
				createdAt: new Date("2026-07-01T00:00:00Z"),
				sessionId: "sess-2",
			},
		];
		const ctx = mockDbReturning(rows);

		const result = await processToolUse(ctx, "s1", "show_session_data", {
			query_type: "dimension_history",
			dimension: "dynamics",
			limit: 20,
		});

		expect(result.isError).toBe(false);
		// Model sees distilled prose, not the raw score.
		expect(result.modelSummary).toBeDefined();
		expect(result.modelSummary).not.toContain(String(DISTINCT_A));
		// The model-facing content selector must resolve to the summary.
		expect(toolResultModelContent(result)).not.toContain(String(DISTINCT_A));
		// Client still receives the raw numeric payload for chart rendering.
		expect(JSON.stringify(result.componentsJson)).toContain(String(DISTINCT_A));
	});
});
