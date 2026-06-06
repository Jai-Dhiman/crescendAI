import { cleanup, render } from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockArtifact = vi.fn(() => null);
vi.mock("./Artifact", () => ({
	Artifact: (props: { component: { type: string }; artifactId: string }) => {
		mockArtifact(props.component);
		return null;
	},
}));
vi.mock("./MessageContent", () => ({
	MessageContent: ({ content }: { content: string }) =>
		React.createElement("div", { "data-testid": "message-content" }, content),
}));
vi.mock("./ToolCallBar", () => ({ ToolCallBar: () => null }));

beforeEach(() => {
	vi.clearAllMocks();
	cleanup();
});

describe("ChatMessages — pending_exercise filter", () => {
	it("does NOT render Artifact for a pending_exercise component on a synthesis message", async () => {
		const { ChatMessages } = await import("./ChatMessages");
		const message = {
			id: "msg-1",
			role: "assistant" as const,
			content: "Your pedaling smeared the line. Want a drill for that?",
			createdAt: new Date().toISOString(),
			messageType: "synthesis" as const,
			sessionId: "session-abc",
			components: [
				{
					type: "pending_exercise" as const,
					config: {
						exerciseId: "ex-123",
						focusDimension: "pedaling",
						previewTitle: "Pedaling clarity drill",
					},
				},
			],
		};
		render(React.createElement(ChatMessages, { messages: [message] }));
		const pendingCalls = mockArtifact.mock.calls.filter(
			(args) => (args[0] as { type?: string })?.type === "pending_exercise",
		);
		expect(pendingCalls).toHaveLength(0);
	});

	it("still renders Artifact for a non-pending_exercise component on the same message", async () => {
		const { ChatMessages } = await import("./ChatMessages");
		const message = {
			id: "msg-2",
			role: "assistant" as const,
			content: "Here are exercises.",
			createdAt: new Date().toISOString(),
			components: [
				{
					type: "exercise_set" as const,
					config: {
						sourcePassage: "bars 1-4",
						targetSkill: "pedaling",
						exercises: [
							{
								title: "Legato drill",
								instruction: "Half tempo.",
								focusDimension: "pedaling",
							},
						],
					},
				},
			],
		};
		render(React.createElement(ChatMessages, { messages: [message] }));
		expect(mockArtifact).toHaveBeenCalledWith(
			expect.objectContaining({ type: "exercise_set" }),
		);
	});
});
