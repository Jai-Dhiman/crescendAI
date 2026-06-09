import { cleanup, render } from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it } from "vitest";
import type { KeyboardGuideConfig } from "../../lib/types";

beforeEach(() => {
	cleanup();
});

describe("KeyboardGuideCard", () => {
	it("renders title, description, and a human-readable hands label", async () => {
		const config: KeyboardGuideConfig = {
			title: "Thumb-under in C major scale",
			description: "Pass the thumb under after the third finger.",
			hands: "right",
		};
		const { KeyboardGuideCard } = await import("./KeyboardGuideCard");
		const { container } = render(
			React.createElement(KeyboardGuideCard, { config }),
		);

		expect(container.textContent).toContain("Thumb-under in C major scale");
		expect(container.textContent).toContain(
			"Pass the thumb under after the third finger.",
		);
		expect(container.textContent).toContain("Right hand");
	});

	it("renders the fingering block when fingering is present", async () => {
		const config: KeyboardGuideConfig = {
			title: "Octave leaps",
			description: "Keep the wrist loose.",
			hands: "both",
			fingering: "1-5, 1-5",
		};
		const { KeyboardGuideCard } = await import("./KeyboardGuideCard");
		const { container } = render(
			React.createElement(KeyboardGuideCard, { config }),
		);

		expect(container.textContent).toContain("Fingering");
		expect(container.textContent).toContain("1-5, 1-5");
		expect(container.textContent).toContain("Both hands");
	});

	it("omits the fingering block when fingering is absent", async () => {
		const config: KeyboardGuideConfig = {
			title: "Legato pedaling",
			description: "Change pedal on each new harmony.",
			hands: "left",
		};
		const { KeyboardGuideCard } = await import("./KeyboardGuideCard");
		const { container } = render(
			React.createElement(KeyboardGuideCard, { config }),
		);

		expect(container.textContent).not.toContain("Fingering");
		expect(container.textContent).toContain("Left hand");
	});
});
