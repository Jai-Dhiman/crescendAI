// src/components/cards/SessionDataCard.test.tsx
import { cleanup, render, screen } from "@testing-library/react";
import * as React from "react";
import { afterEach, describe, expect, it } from "vitest";
import type { SessionDataConfig } from "../../lib/types";

afterEach(() => {
	cleanup();
});

describe("SessionDataCard", () => {
	it("renders dimension_history observation rows", async () => {
		const config: SessionDataConfig = {
			queryType: "dimension_history",
			studentId: "stu-1",
			data: [
				{
					id: "obs-1",
					dimension: "pedaling",
					dimensionScore: 3.4,
					observationText: "Pedal changes lag the harmony.",
					framing: "correction",
					createdAt: "2026-06-20T10:00:00Z",
					sessionId: "sess-1",
				},
			],
		};
		const { SessionDataCard } = await import("./SessionDataCard");
		render(React.createElement(SessionDataCard, { config }));

		expect(screen.getByText("Dimension history")).toBeInTheDocument();
		expect(screen.getByText("pedaling")).toBeInTheDocument();
		expect(
			screen.getByText("Pedal changes lag the harmony."),
		).toBeInTheDocument();
	});

	it("renders recent_sessions with per-dimension averages", async () => {
		const config: SessionDataConfig = {
			queryType: "recent_sessions",
			studentId: "stu-1",
			data: [
				{
					id: "sess-1",
					startedAt: "2026-06-20T10:00:00Z",
					endedAt: "2026-06-20T10:30:00Z",
					avgDynamics: 3.2,
					avgTiming: 4.1,
					avgPedaling: 2.8,
					avgArticulation: 3.9,
					avgPhrasing: 3.5,
					avgInterpretation: 4.0,
				},
			],
		};
		const { SessionDataCard } = await import("./SessionDataCard");
		render(React.createElement(SessionDataCard, { config }));

		expect(screen.getByText("Recent sessions")).toBeInTheDocument();
		// one chip per dimension (6) — assert a couple of formatted scores
		expect(screen.getByText("3.2")).toBeInTheDocument();
		expect(screen.getByText("4.1")).toBeInTheDocument();
		expect(screen.getByText("dyn")).toBeInTheDocument();
	});

	it("renders an empty state when data is null or empty", async () => {
		const { SessionDataCard } = await import("./SessionDataCard");

		const emptyHistory: SessionDataConfig = {
			queryType: "dimension_history",
			studentId: "stu-1",
			data: [],
		};
		render(React.createElement(SessionDataCard, { config: emptyHistory }));
		expect(screen.getByText("No session data yet.")).toBeInTheDocument();
		cleanup();

		const nullDetail: SessionDataConfig = {
			queryType: "session_detail",
			studentId: "stu-1",
			data: null,
		};
		render(React.createElement(SessionDataCard, { config: nullDetail }));
		expect(screen.getByText("No session data yet.")).toBeInTheDocument();
	});

	it("renders session_detail averages for a single session row", async () => {
		const config: SessionDataConfig = {
			queryType: "session_detail",
			studentId: "stu-1",
			data: {
				id: "sess-9",
				startedAt: "2026-06-21T09:00:00Z",
				endedAt: "2026-06-21T09:25:00Z",
				avgDynamics: 2.5,
				avgTiming: 2.5,
				avgPedaling: 2.5,
				avgArticulation: 2.5,
				avgPhrasing: 2.5,
				avgInterpretation: 2.5,
			},
		};
		const { SessionDataCard } = await import("./SessionDataCard");
		render(React.createElement(SessionDataCard, { config }));

		expect(screen.getByText("Session detail")).toBeInTheDocument();
		// all six dims share the same score; assert all six chips rendered it
		expect(screen.getAllByText("2.5")).toHaveLength(6);
	});
});
