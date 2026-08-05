// apps/web/src/routes/app.chats.test.tsx
import { QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../lib/auth";
import { queryClient } from "../lib/query-client";

const mockDeleteMutate = vi.fn();
vi.mock("../hooks/useConversations", () => ({
	useConversations: () => ({
		data: [
			{
				id: "conv-1",
				title: "Sample conversation",
				updatedAt: new Date().toISOString(),
			},
		],
		isPending: false,
	}),
	useDeleteConversations: () => ({
		mutate: mockDeleteMutate,
		isPending: false,
	}),
}));

async function renderAllChats() {
	const { Route } = await import("./app.chats");
	const AllChatsPage = (
		Route as unknown as { options: { component: React.ComponentType } }
	).options.component;
	if (!AllChatsPage)
		throw new Error("AllChatsPage component not found on route");
	return render(
		React.createElement(
			QueryClientProvider,
			{ client: queryClient },
			React.createElement(
				AuthProvider,
				null,
				React.createElement(AllChatsPage),
			),
		),
	);
}

describe("AllChats structure", () => {
	it("uses the danger token for the delete-selection action", async () => {
		await renderAllChats();
		fireEvent.click(screen.getByLabelText("Select"));
		const action = screen.getByRole("button", { name: /delete/i });
		expect(action.className).toContain("text-danger");
	});
});
