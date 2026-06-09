import { cleanup, render } from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { InlineComponent } from "../lib/types";

// Sibling cards routed by InlineCard pull in these modules at import time; stub
// the heavy ones so the switch can be exercised in jsdom.
vi.mock("../lib/score-renderer", () => ({
	scoreRenderer: { getClip: vi.fn(), load: vi.fn(), getPage: vi.fn() },
}));
vi.mock("../lib/api", () => ({
	api: { exercises: { assign: vi.fn() } },
	acceptSegmentLoop: vi.fn(),
	declineSegmentLoop: vi.fn(),
	dismissSegmentLoop: vi.fn(),
}));
vi.mock("../lib/passage-player", () => ({ PassagePlayer: class {} }));

beforeEach(() => {
	vi.clearAllMocks();
	cleanup();
});

describe("InlineCard routing", () => {
	it("routes keyboard_guide to a real card, not the placeholder", async () => {
		const component: InlineComponent = {
			type: "keyboard_guide",
			config: {
				title: "Thumb-under",
				description: "Pass the thumb under cleanly.",
				hands: "right",
			},
		};
		const { InlineCard } = await import("./InlineCard");
		const { container } = render(
			React.createElement(InlineCard, { component }),
		);

		expect(container.textContent).toContain("Thumb-under");
		expect(container.textContent).not.toContain("coming soon");
	});

	it("routes session_data to a real card, not the placeholder", async () => {
		const component: InlineComponent = {
			type: "session_data",
			config: {
				queryType: "recent_sessions",
				studentId: "student-1",
				data: [],
			},
		};
		const { InlineCard } = await import("./InlineCard");
		const { container } = render(
			React.createElement(InlineCard, { component }),
		);

		expect(container.textContent).toContain("Recent sessions");
		expect(container.textContent).not.toContain("coming soon");
	});
});
