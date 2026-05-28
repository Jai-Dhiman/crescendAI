// apps/web/src/routes/index.test.tsx
import { render, screen } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("../lib/landing-analytics", () => ({
	trackLandingEvent: vi.fn(),
}));

// Score rendering hits the network; in jsdom we let it reject so the score
// degrades gracefully (no throw).
vi.mock("../lib/score-renderer", () => ({
	scoreRenderer: {
		load: vi.fn().mockRejectedValue(new Error("no network in test")),
		getClip: vi.fn().mockRejectedValue(new Error("no network in test")),
	},
}));

vi.mock("../lib/api", () => ({
	api: { waitlist: { join: vi.fn().mockResolvedValue({ ok: true }) } },
}));

async function renderLanding() {
	const { Route } = await import("./index");
	const LandingPage = (
		Route as unknown as { options: { component: React.ComponentType } }
	).options.component;
	if (!LandingPage) throw new Error("LandingPage component not found on route");
	return render(React.createElement(LandingPage));
}

describe("LandingPage structure", () => {
	it("keeps the hero headline and Start Practicing CTA", async () => {
		await renderLanding();
		expect(screen.getByText(/A teacher for every pianist/)).not.toBeNull();
		expect(
			screen.getByRole("button", { name: /Start Practicing/i }),
		).not.toBeNull();
	});

	it("shows the glimpse with the Nocturne and an app frame", async () => {
		await renderLanding();
		expect(
			screen.getByText(/See what your playing sounds like/),
		).not.toBeNull();
		expect(screen.getByText(/Nocturne Op\. 9 No\. 2/)).not.toBeNull();
		expect(screen.getByAltText(/iOS app/i)).not.toBeNull();
	});

	it("renders the waitlist capture with an email field and submit", async () => {
		await renderLanding();
		expect(
			screen.getByText(/Every pianist deserves a great teacher/),
		).not.toBeNull();
		expect(screen.getByPlaceholderText(/you@example\.com/)).not.toBeNull();
		expect(
			screen.getByRole("button", { name: /Join the waitlist/i }),
		).not.toBeNull();
	});

	it("no longer shows the removed mock-video or Ballade copy", async () => {
		await renderLanding();
		expect(screen.queryByText(/Record yourself playing/)).toBeNull();
		expect(screen.queryByText(/Ballade No\. 1/)).toBeNull();
		expect(screen.queryByText(/Be first to play/)).toBeNull();
	});
});
