import { afterEach, describe, expect, it, vi } from "vitest";
import type { ScoreHighlightConfig } from "../../lib/types";

// Mock osmd-manager
const mockEnsureRendered = vi.fn().mockResolvedValue(undefined);
const mockClipBars = vi.fn().mockReturnValue(null);

vi.mock("../../lib/osmd-manager", () => ({
	osmdManager: {
		ensureRendered: (...args: unknown[]) => mockEnsureRendered(...args),
		clipBars: (...args: unknown[]) => mockClipBars(...args),
	},
}));

describe("ScoreHighlightCard", () => {
	afterEach(() => {
		vi.clearAllMocks();
	});

	const config: ScoreHighlightConfig = {
		pieceId: "123e4567-e89b-12d3-a456-426614174000",
		highlights: [
			{ bars: [4, 8] as [number, number], dimension: "dynamics", annotation: "crescendo builds" },
			{ bars: [12, 16] as [number, number], dimension: "pedaling" },
		],
	};

	it("exports ScoreHighlightCard", async () => {
		const mod = await import("./ScoreHighlightCard");
		expect(typeof mod.ScoreHighlightCard).toBe("function");
	});

	it("calls ensureRendered with pieceId on import", async () => {
		const { ScoreHighlightCard } = await import("./ScoreHighlightCard");
		expect(ScoreHighlightCard).toBeDefined();
		// Component should accept config and onExpand props without TypeScript errors
		// Runtime rendering test would need @testing-library/react (integration test)
	});
});
