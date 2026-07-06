import { describe, expect, it } from "vitest";
import type { ServiceContext } from "../lib/types";
import {
	type InlineComponent,
	processToolUse,
	type ToolResult,
	toolResultModelContent,
} from "./tool-processor";

// A distinctive raw score that would be trivial to spot if it leaked verbatim.
const DISTINCT_A = 0.7314159;

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
// processToolUse wiring: show_session_data embeds a distilled modelSummary and
// keeps raw numbers in componentsJson for the client chart.
// ---------------------------------------------------------------------------

function mockDbReturning(rows: unknown): ServiceContext {
	const chain: Record<string, unknown> = {};
	for (const m of ["select", "from", "where", "orderBy"])
		chain[m] = () => chain;
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
		expect(toolResultModelContent(result)).not.toContain(String(DISTINCT_A));
		// The distilled prose is embedded on the component too (single source).
		const config = (result.componentsJson[0] as InlineComponent).config as {
			modelSummary?: string;
		};
		expect(config.modelSummary).toBe(result.modelSummary);
		// Client still receives the raw numeric payload for chart rendering.
		expect(JSON.stringify(result.componentsJson)).toContain(String(DISTINCT_A));
	});
});
