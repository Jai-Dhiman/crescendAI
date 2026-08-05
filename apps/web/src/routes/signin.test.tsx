// apps/web/src/routes/signin.test.tsx
import { QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it } from "vitest";
import { AuthProvider } from "../lib/auth";
import { queryClient } from "../lib/query-client";

async function renderSignIn() {
	const { Route } = await import("./signin");
	const SignInPage = (
		Route as unknown as { options: { component: React.ComponentType } }
	).options.component;
	if (!SignInPage) throw new Error("SignInPage component not found on route");
	return render(
		React.createElement(
			QueryClientProvider,
			{ client: queryClient },
			React.createElement(AuthProvider, null, React.createElement(SignInPage)),
		),
	);
}

describe("SignInPage structure", () => {
	it("uses the surface-page token for the hero gradient overlay, not a hard-coded rgba", async () => {
		const { container } = await renderSignIn();
		const gradientOverlay = container.querySelector("div.absolute.inset-0");
		expect(gradientOverlay).not.toBeNull();
		const background = (gradientOverlay as HTMLElement).getAttribute("style");
		expect(background).toContain("var(--color-surface-page)");
		expect(background).not.toContain("rgba(45,41,38");
	});
});
