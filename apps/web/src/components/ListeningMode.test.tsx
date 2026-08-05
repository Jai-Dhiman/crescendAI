// apps/web/src/components/ListeningMode.test.tsx
import { render } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./AudioWaveformRing", () => ({
	AudioWaveformRing: () => null,
}));

describe("ListeningMode edge-ring styling", () => {
	it("uses color-mix on the accent token for the edge-ring border and glow", async () => {
		const { ListeningMode } = await import("./ListeningMode");

		render(
			React.createElement(ListeningMode, {
				state: "recording",
				energy: 0,
				isPlaying: true,
				error: null,
				wsStatus: "connected",
				onStop: () => {},
				originRect: null,
				onExit: () => {},
				pieceContext: null,
				sessionNotes: "",
				onNotesChange: () => {},
				observations: [],
				analyserNode: null,
			}),
		);

		const edgeRing = document.querySelector(".z-49");
		expect(edgeRing).not.toBeNull();
		expect((edgeRing as HTMLElement).style.border).toBe(
			"2px solid color-mix(in srgb, var(--color-accent) 70%, transparent)",
		);
	});
});
