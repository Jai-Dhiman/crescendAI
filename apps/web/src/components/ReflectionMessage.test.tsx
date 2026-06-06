import {
	cleanup,
	fireEvent,
	render,
	screen,
	waitFor,
} from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockAssignPending = vi.fn();
vi.mock("../lib/api", () => ({
	api: {
		exercises: {
			assignPending: (...args: unknown[]) => mockAssignPending(...args),
		},
	},
}));

const mockArtifact = vi.fn(() => null);
vi.mock("./Artifact", () => ({
	Artifact: (props: {
		component: { type: string; config: unknown };
		artifactId: string;
	}) => {
		mockArtifact(props.component);
		return React.createElement(
			"div",
			{ "data-testid": "artifact", "data-type": props.component.type },
			null,
		);
	},
}));

beforeEach(() => {
	vi.clearAllMocks();
	cleanup();
});

const pendingConfig = {
	exerciseId: "ex-123",
	focusDimension: "pedaling",
	previewTitle: "Pedaling clarity drill",
};
const resolvedPayload = {
	sourcePassage: "bars 5-8",
	targetSkill: "Pedaling clarity",
	exercises: [
		{
			title: "Legato run",
			instruction: "Half tempo.",
			focusDimension: "pedaling",
			exerciseId: "ex-123",
		},
	],
};

describe("ReflectionMessage", () => {
	it("renders reflection text, Confirm button, and Not-now button", async () => {
		const { ReflectionMessage } = await import("./ReflectionMessage");
		render(
			React.createElement(ReflectionMessage, {
				sessionId: "sess-1",
				reflectionText:
					"Your pedaling smeared the line in the running passage.",
				pendingConfig,
				onDecline: vi.fn(),
			}),
		);
		expect(
			screen.getByText(
				"Your pedaling smeared the line in the running passage.",
			),
		).toBeTruthy();
		expect(screen.getByRole("button", { name: /confirm/i })).toBeTruthy();
		expect(screen.getByRole("button", { name: /not now/i })).toBeTruthy();
	});

	it("clicking Confirm calls assignPending and reveals the exercise via Artifact", async () => {
		mockAssignPending.mockResolvedValueOnce(resolvedPayload);
		const { ReflectionMessage } = await import("./ReflectionMessage");
		render(
			React.createElement(ReflectionMessage, {
				sessionId: "sess-1",
				reflectionText: "Your pedaling smeared the line.",
				pendingConfig,
				onDecline: vi.fn(),
			}),
		);
		fireEvent.click(screen.getByRole("button", { name: /confirm/i }));
		await waitFor(() =>
			expect(mockAssignPending).toHaveBeenCalledWith({
				sessionId: "sess-1",
				exerciseId: "ex-123",
			}),
		);
		await waitFor(() =>
			expect(mockArtifact).toHaveBeenCalledWith(
				expect.objectContaining({ type: "exercise_set" }),
			),
		);
	});

	it("clicking Not-now calls onDecline with focusDimension and does NOT call assignPending", async () => {
		const onDecline = vi.fn();
		const { ReflectionMessage } = await import("./ReflectionMessage");
		render(
			React.createElement(ReflectionMessage, {
				sessionId: "sess-1",
				reflectionText: "Your pedaling smeared the line.",
				pendingConfig,
				onDecline,
			}),
		);
		fireEvent.click(screen.getByRole("button", { name: /not now/i }));
		expect(onDecline).toHaveBeenCalledWith("pedaling");
		expect(mockAssignPending).not.toHaveBeenCalled();
	});

	it("Confirm shows loading state while assignPending is in flight, then reveals", async () => {
		let resolveAssign!: (v: typeof resolvedPayload) => void;
		mockAssignPending.mockReturnValueOnce(
			new Promise<typeof resolvedPayload>((res) => {
				resolveAssign = res;
			}),
		);
		const { ReflectionMessage } = await import("./ReflectionMessage");
		render(
			React.createElement(ReflectionMessage, {
				sessionId: "sess-1",
				reflectionText: "Your pedaling smeared the line.",
				pendingConfig,
				onDecline: vi.fn(),
			}),
		);
		fireEvent.click(screen.getByRole("button", { name: /confirm/i }));
		await waitFor(() =>
			expect(
				screen.getByRole("button", { name: /confirm|adding/i }),
			).toBeTruthy(),
		);
		resolveAssign(resolvedPayload);
		await waitFor(() =>
			expect(mockArtifact).toHaveBeenCalledWith(
				expect.objectContaining({ type: "exercise_set" }),
			),
		);
	});
});
