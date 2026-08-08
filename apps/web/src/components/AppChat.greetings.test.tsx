import { QueryClientProvider } from "@tanstack/react-query";
import {
	createMemoryHistory,
	createRootRoute,
	createRouter,
	RouterProvider,
} from "@tanstack/react-router";
import { render, screen, waitFor } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../lib/auth";
import { queryClient } from "../lib/query-client";
import AppChat from "./AppChat";

// AppChat pulls in auth/conversation queries; the existing test-setup mocks
// (matchMedia, ResizeObserver, IntersectionObserver) cover its render-time
// needs. This test only asserts on text content, not on network state.
vi.mock("../hooks/useAuth", () => ({
	authQueryOptions: { queryKey: ["auth"], queryFn: () => null },
	useAuth: () => ({ data: null, isLoading: false }),
}));

// AppChat reads conversationIdFromUrl via useRouterState, which throws
// outside a RouterProvider (unlike useNavigate, which only warns). Build a
// minimal single-route memory router so AppChat can render standalone.
function renderAppChat() {
	const rootRoute = createRootRoute({ component: AppChat });
	const routeTree = rootRoute;
	const router = createRouter({
		routeTree,
		history: createMemoryHistory({ initialEntries: ["/"] }),
	});
	return render(
		React.createElement(
			QueryClientProvider,
			{ client: queryClient },
			React.createElement(
				AuthProvider,
				null,
				React.createElement(RouterProvider, { router }),
			),
		),
	);
}

describe("AppChat empty state", () => {
	it("renders no GREETINGS headline", async () => {
		renderAppChat();
		// The router resolves the initial match asynchronously; wait for the
		// empty-state chat input placeholder to confirm AppChat has settled.
		await waitFor(() => {
			expect(
				screen.getByPlaceholderText("What are you practicing today?"),
			).toBeInTheDocument();
		});
		// None of the retired lines should appear anywhere in the document.
		expect(
			screen.queryByText("Let's make some music."),
		).not.toBeInTheDocument();
		expect(
			screen.queryByText("Your piano misses you."),
		).not.toBeInTheDocument();
	});
});
