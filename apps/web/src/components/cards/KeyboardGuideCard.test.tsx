// src/components/cards/KeyboardGuideCard.test.tsx
import { cleanup, render, screen } from "@testing-library/react";
import * as React from "react";
import { afterEach, describe, expect, it } from "vitest";
import type { KeyboardGuideConfig } from "../../lib/types";

afterEach(() => {
	cleanup();
});

describe("KeyboardGuideCard", () => {
	it("renders title, description, and hands label", async () => {
		const config: KeyboardGuideConfig = {
			title: "Thumb-under crossing",
			description: "Rotate the wrist as the thumb passes under.",
			hands: "right",
		};
		const { KeyboardGuideCard } = await import("./KeyboardGuideCard");
		render(React.createElement(KeyboardGuideCard, { config }));

		expect(screen.getByText("Thumb-under crossing")).toBeInTheDocument();
		expect(
			screen.getByText("Rotate the wrist as the thumb passes under."),
		).toBeInTheDocument();
		expect(screen.getByText("Right hand")).toBeInTheDocument();
	});

	it("renders fingering when present and omits it when absent", async () => {
		const { KeyboardGuideCard } = await import("./KeyboardGuideCard");

		const withFingering: KeyboardGuideConfig = {
			title: "Scale run",
			description: "Even C major scale.",
			hands: "both",
			fingering: "1 2 3 1 2 3 4 5",
		};
		const { rerender } = render(
			React.createElement(KeyboardGuideCard, { config: withFingering }),
		);
		expect(screen.getByText("Fingering")).toBeInTheDocument();
		expect(screen.getByText("1 2 3 1 2 3 4 5")).toBeInTheDocument();

		const withoutFingering: KeyboardGuideConfig = {
			title: "Scale run",
			description: "Even C major scale.",
			hands: "both",
		};
		rerender(
			React.createElement(KeyboardGuideCard, { config: withoutFingering }),
		);
		expect(screen.queryByText("Fingering")).not.toBeInTheDocument();
	});
});
