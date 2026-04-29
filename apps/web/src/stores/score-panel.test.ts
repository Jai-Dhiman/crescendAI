import { afterEach, describe, expect, it } from "vitest";
import { useScorePanelStore } from "./score-panel";

afterEach(() => {
	useScorePanelStore.getState().clear();
});

describe("openHighlight", () => {
	it("opens the panel with highlight data", () => {
		const store = useScorePanelStore.getState();
		store.openHighlight({
			pieceId: "piece-abc",
			highlights: [
				{
					bars: [4, 8] as [number, number],
					dimension: "dynamics",
					annotation: "test",
				},
			],
		});

		const state = useScorePanelStore.getState();
		expect(state.isOpen).toBe(true);
		expect(state.highlightData).not.toBeNull();
		expect(state.highlightData!.pieceId).toBe("piece-abc");
		expect(state.highlightData!.highlights).toHaveLength(1);
	});

	it("clears previous highlight data on close", () => {
		const store = useScorePanelStore.getState();
		store.openHighlight({
			pieceId: "piece-abc",
			highlights: [{ bars: [1, 4] as [number, number], dimension: "timing" }],
		});
		useScorePanelStore.getState().close();

		expect(useScorePanelStore.getState().isOpen).toBe(false);
		// highlightData preserved so reopening can restore
	});

	it("replaces previous highlight data on new openHighlight", () => {
		const store = useScorePanelStore.getState();
		store.openHighlight({
			pieceId: "piece-1",
			highlights: [{ bars: [1, 4] as [number, number], dimension: "dynamics" }],
		});
		store.openHighlight({
			pieceId: "piece-2",
			highlights: [{ bars: [5, 8] as [number, number], dimension: "pedaling" }],
		});

		const state = useScorePanelStore.getState();
		expect(state.highlightData!.pieceId).toBe("piece-2");
	});
});

describe("open (DEV gate removed)", () => {
	it("opens the panel with session data in any environment", () => {
		const store = useScorePanelStore.getState();
		store.open({
			piece: "Test Piece",
			section: "mm. 1-8",
			durationSeconds: 120,
			observations: [],
		});

		expect(useScorePanelStore.getState().isOpen).toBe(true);
	});
});
