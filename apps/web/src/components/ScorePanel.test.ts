import { describe, expect, it, vi } from "vitest";
import { useScorePanelStore } from "../stores/score-panel";

// Mock osmd-manager
vi.mock("../lib/osmd-manager", () => ({
	osmdManager: {
		ensureRendered: vi.fn().mockResolvedValue(undefined),
		getOsmdInstance: vi.fn().mockReturnValue(null),
		clipBars: vi.fn().mockReturnValue(null),
		reset: vi.fn(),
	},
}));

describe("ScorePanel highlight integration", () => {
	it("reads highlightData from store when opened via openHighlight", () => {
		const store = useScorePanelStore.getState();
		store.openHighlight({
			pieceId: "piece-abc",
			highlights: [
				{ bars: [4, 8] as [number, number], dimension: "dynamics", annotation: "forte" },
			],
		});

		const state = useScorePanelStore.getState();
		expect(state.isOpen).toBe(true);
		expect(state.highlightData).not.toBeNull();
		expect(state.highlightData!.pieceId).toBe("piece-abc");
		expect(state.sessionData).toBeNull();

		store.clear();
	});

	it("ScorePanel module exports ScorePanel component", async () => {
		const mod = await import("./ScorePanel");
		expect(typeof mod.ScorePanel).toBe("function");
	});
});
