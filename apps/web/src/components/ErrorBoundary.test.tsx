// apps/web/src/components/ErrorBoundary.test.tsx
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "./ErrorBoundary";

function Thrower(): never {
	throw new Error("boom");
}

describe("ErrorBoundary retry button styling", () => {
	beforeEach(() => {
		vi.spyOn(console, "error").mockImplementation(() => {});
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it("renders the retry button with on-accent text, not the deleted espresso token", () => {
		render(
			<ErrorBoundary>
				<Thrower />
			</ErrorBoundary>,
		);

		const button = screen.getByRole("button", { name: /reload/i });
		expect(button.className).toContain("text-on-accent");
		expect(button.className).not.toContain("text-espresso");
	});

	it("uses accent/80 hover, not the deleted accent-lighter token", () => {
		render(
			<ErrorBoundary>
				<Thrower />
			</ErrorBoundary>,
		);

		const button = screen.getByRole("button", { name: /reload/i });
		expect(button.className).toContain("hover:bg-accent/80");
		expect(button.className).not.toContain("accent-lighter");
	});
});
