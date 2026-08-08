import { QueryClientProvider } from "@tanstack/react-query";
import {
	createMemoryHistory,
	createRootRoute,
	createRouter,
	RouterProvider,
} from "@tanstack/react-router";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../lib/auth";
import { queryClient } from "../lib/query-client";
import AppChat from "./AppChat";

vi.mock("../hooks/useAuth", () => ({
	authQueryOptions: { queryKey: ["auth"], queryFn: () => null },
	useAuth: () => ({ data: null, isLoading: false }),
}));

// Module-scoped so the test can assert on calls after AppChat renders --
// vi.mock's factory below closes over this same reference.
const stop = vi.fn();

// Force usePracticeSession into a recording state so AppChat's branch that
// mounts the full-screen practice surface is reachable without a real mic,
// AudioContext, or WebSocket.
vi.mock("../hooks/usePracticeSession", () => ({
	usePracticeSession: () => ({
		state: "recording",
		elapsedSeconds: 12,
		observations: [],
		latestScores: null,
		summary: null,
		error: null,
		chunksProcessed: 0,
		chunkStates: [],
		wsStatus: "connected",
		isOnline: true,
		isPlaying: true,
		energy: 0,
		analyserNode: null,
		practiceMode: null,
		marks: [],
		start: vi.fn(),
		stop,
		setPiece: vi.fn(),
		observationMessages: [],
		conversationId: null,
		activeLoop: null,
	}),
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

describe("AppChat practice mode", () => {
	it("mounts PracticeMode's pieceless surface while recording, not the waveform ring", async () => {
		renderAppChat();
		await waitFor(() => {
			expect(screen.getByTestId("session-timeline")).toBeInTheDocument();
		});
		expect(
			screen.queryByLabelText(/toggle metronome/i),
		).not.toBeInTheDocument();
	});

	it("stopping recording calls practice.stop and exits the practice surface", async () => {
		renderAppChat();
		await waitFor(() => {
			expect(screen.getByTestId("session-timeline")).toBeInTheDocument();
		});

		fireEvent.click(screen.getByRole("button", { name: /stop recording/i }));

		expect(stop).toHaveBeenCalledTimes(1);
		expect(screen.queryByTestId("session-timeline")).not.toBeInTheDocument();
	});
});
