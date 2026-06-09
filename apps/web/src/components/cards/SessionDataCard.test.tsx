import { cleanup, render } from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it } from "vitest";
import type {
	SessionDataConfig,
	SessionDataObservationRow,
	SessionDataSessionRow,
} from "../../lib/types";

beforeEach(() => {
	cleanup();
});

function makeSession(
	overrides: Partial<SessionDataSessionRow> = {},
): SessionDataSessionRow {
	return {
		id: "session-1",
		startedAt: "2026-06-01T10:00:00.000Z",
		endedAt: "2026-06-01T10:30:00.000Z",
		avgDynamics: 3.2,
		avgTiming: 4.1,
		avgPedaling: 2.8,
		avgArticulation: 3.5,
		avgPhrasing: 4.0,
		avgInterpretation: 3.9,
		...overrides,
	};
}

describe("SessionDataCard", () => {
	it("renders observation rows for dimension_history", async () => {
		const obs: SessionDataObservationRow = {
			id: "obs-1",
			dimension: "dynamics",
			dimensionScore: 3.4,
			observationText: "Crescendo arrived too early in bar 12.",
			framing: "correction",
			createdAt: "2026-06-02T09:00:00.000Z",
			sessionId: "session-1",
		};
		const config: SessionDataConfig = {
			queryType: "dimension_history",
			studentId: "student-1",
			data: [obs],
		};
		const { SessionDataCard } = await import("./SessionDataCard");
		const { container } = render(
			React.createElement(SessionDataCard, { config }),
		);

		expect(container.textContent).toContain("Dimension history");
		expect(container.textContent).toContain("dynamics");
		expect(container.textContent).toContain(
			"Crescendo arrived too early in bar 12.",
		);
		expect(container.textContent).toContain("3.4");
	});

	it("renders one row per session for recent_sessions", async () => {
		const config: SessionDataConfig = {
			queryType: "recent_sessions",
			studentId: "student-1",
			data: [
				makeSession({ id: "s1" }),
				makeSession({ id: "s2", avgTiming: 1.1 }),
			],
		};
		const { SessionDataCard } = await import("./SessionDataCard");
		const { container } = render(
			React.createElement(SessionDataCard, { config }),
		);

		expect(container.textContent).toContain("Recent sessions");
		expect(container.textContent).toContain("Dynamics");
		// both sessions' distinct timing values present
		expect(container.textContent).toContain("4.1");
		expect(container.textContent).toContain("1.1");
	});

	it("renders the full dimension grid for session_detail", async () => {
		const config: SessionDataConfig = {
			queryType: "session_detail",
			studentId: "student-1",
			data: makeSession(),
		};
		const { SessionDataCard } = await import("./SessionDataCard");
		const { container } = render(
			React.createElement(SessionDataCard, { config }),
		);

		expect(container.textContent).toContain("Session detail");
		for (const label of [
			"Dynamics",
			"Timing",
			"Pedaling",
			"Articulation",
			"Phrasing",
			"Interpretation",
		]) {
			expect(container.textContent).toContain(label);
		}
		expect(container.textContent).toContain("2.8");
	});

	it("renders a dash for null dimension scores", async () => {
		const config: SessionDataConfig = {
			queryType: "session_detail",
			studentId: "student-1",
			data: makeSession({ avgPedaling: null }),
		};
		const { SessionDataCard } = await import("./SessionDataCard");
		const { container } = render(
			React.createElement(SessionDataCard, { config }),
		);

		expect(container.textContent).toContain("—");
	});

	it("renders an empty state when data is null", async () => {
		const config: SessionDataConfig = {
			queryType: "session_detail",
			studentId: "student-1",
			data: null,
		};
		const { SessionDataCard } = await import("./SessionDataCard");
		const { container } = render(
			React.createElement(SessionDataCard, { config }),
		);

		expect(container.textContent).toContain("Session not found.");
	});

	it("renders an empty state for an empty dimension_history list", async () => {
		const config: SessionDataConfig = {
			queryType: "dimension_history",
			studentId: "student-1",
			data: [],
		};
		const { SessionDataCard } = await import("./SessionDataCard");
		const { container } = render(
			React.createElement(SessionDataCard, { config }),
		);

		expect(container.textContent).toContain("No observations recorded yet.");
	});
});
