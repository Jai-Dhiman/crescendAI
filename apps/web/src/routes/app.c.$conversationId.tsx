import { createFileRoute } from "@tanstack/react-router";

// AppChat is rendered by the /app layout route (app.tsx).
// This route exists for route matching and $conversationId param extraction.
export const Route = createFileRoute("/app/c/$conversationId")({
	component: () => null,
});
