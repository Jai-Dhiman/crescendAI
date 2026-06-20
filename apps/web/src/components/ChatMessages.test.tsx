import { cleanup, render, screen, waitFor } from "@testing-library/react";
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

describe("ChatMessages — ReflectionMessage routing", () => {
	it("routes a synthesis message with a pending_exercise component to ReflectionMessage", async () => {
		vi.resetModules();
		vi.doMock("./ReflectionMessage", () => ({
			ReflectionMessage: () =>
				React.createElement(
					"div",
					{ "data-testid": "reflection-message" },
					null,
				),
		}));
		// re-mock the siblings that the fresh module graph will pull in
		vi.doMock("./Artifact", () => ({ Artifact: () => null }));
		vi.doMock("./MessageContent", () => ({
			MessageContent: ({ content }: { content: string }) =>
				React.createElement(
					"div",
					{ "data-testid": "message-content" },
					content,
				),
		}));
		vi.doMock("./ToolCallBar", () => ({ ToolCallBar: () => null }));

		const { ChatMessages } = await import("./ChatMessages");
		const message = {
			id: "msg-synth",
			role: "assistant" as const,
			content: "Your pedaling smeared the line. Want a drill?",
			createdAt: new Date().toISOString(),
			messageType: "synthesis" as const,
			sessionId: "sess-abc",
			components: [
				{
					type: "pending_exercise" as const,
					config: {
						exerciseId: "ex-789",
						focusDimension: "pedaling",
						previewTitle: "Pedaling clarity drill",
					},
				},
			],
		};
		render(
			React.createElement(ChatMessages, {
				messages: [message],
				onDecline: vi.fn(),
			}),
		);
		await waitFor(() =>
			expect(screen.getByTestId("reflection-message")).toBeTruthy(),
		);
		expect(
			document.querySelectorAll("[data-testid='message-content']"),
		).toHaveLength(0);
	});
});

describe("ChatMessages — assistant-message testid", () => {
	it("renders data-testid=assistant-message on a plain assistant reply", async () => {
		vi.resetModules();
		vi.doMock("./Artifact", () => ({ Artifact: () => null }));
		vi.doMock("./MessageContent", () => ({
			MessageContent: ({ content }: { content: string }) =>
				React.createElement("div", { "data-testid": "message-content" }, content),
		}));
		vi.doMock("./ToolCallBar", () => ({ ToolCallBar: () => null }));

		const { ChatMessages } = await import("./ChatMessages");
		const message = {
			id: "msg-assistant",
			role: "assistant" as const,
			content: "Here is my feedback on your playing.",
			createdAt: new Date().toISOString(),
		};
		render(React.createElement(ChatMessages, { messages: [message] }));
		expect(document.querySelector("[data-testid='assistant-message']")).not.toBeNull();
	});

	it("does NOT add assistant-message testid to a synthesis message", async () => {
		vi.resetModules();
		vi.doMock("./Artifact", () => ({ Artifact: () => null }));
		vi.doMock("./MessageContent", () => ({
			MessageContent: ({ content }: { content: string }) =>
				React.createElement("div", { "data-testid": "message-content" }, content),
		}));
		vi.doMock("./ToolCallBar", () => ({ ToolCallBar: () => null }));

		const { ChatMessages } = await import("./ChatMessages");
		const message = {
			id: "msg-synth",
			role: "assistant" as const,
			content: "Your phrasing needs work.",
			createdAt: new Date().toISOString(),
			messageType: "synthesis" as const,
		};
		render(React.createElement(ChatMessages, { messages: [message] }));
		expect(document.querySelector("[data-testid='synthesis-message']")).not.toBeNull();
		expect(document.querySelector("[data-testid='assistant-message']")).toBeNull();
	});

	it("does NOT add assistant-message testid to a user message", async () => {
		vi.resetModules();
		vi.doMock("./Artifact", () => ({ Artifact: () => null }));
		vi.doMock("./MessageContent", () => ({
			MessageContent: ({ content }: { content: string }) =>
				React.createElement("div", { "data-testid": "message-content" }, content),
		}));
		vi.doMock("./ToolCallBar", () => ({ ToolCallBar: () => null }));

		const { ChatMessages } = await import("./ChatMessages");
		const message = {
			id: "msg-user",
			role: "user" as const,
			content: "How was my playing?",
			createdAt: new Date().toISOString(),
		};
		render(React.createElement(ChatMessages, { messages: [message] }));
		expect(document.querySelector("[data-testid='assistant-message']")).toBeNull();
	});
});
