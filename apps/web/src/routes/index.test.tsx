// apps/web/src/routes/index.test.tsx
import { render, screen } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("../lib/landing-analytics", () => ({
	trackLandingEvent: vi.fn(),
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

	it("shows the feature animation cards", async () => {
		const { container } = await renderLanding();
		expect(screen.getByText(/Record yourself playing/)).not.toBeNull();
		expect(screen.getByText(/Exercises built for you/)).not.toBeNull();
		expect(screen.getByText(/A teacher who knows your playing/)).not.toBeNull();
		expect(container.querySelectorAll("video").length).toBe(3);
	});

	it("shows both device frames (desktop and mobile)", async () => {
		await renderLanding();
		expect(screen.getByAltText(/desktop app/i)).not.toBeNull();
		expect(screen.getByAltText(/mobile app/i)).not.toBeNull();
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

	it("no longer shows the removed sample-analysis copy", async () => {
		await renderLanding();
		expect(screen.queryByText(/Sample analysis/i)).toBeNull();
		expect(screen.queryByText(/Be first to play/)).toBeNull();
	});
});
