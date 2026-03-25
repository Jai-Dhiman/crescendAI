import { createFileRoute } from "@tanstack/react-router";

// AppChat is rendered by the /app layout route (app.tsx).
// This route exists only for TanStack Router's route matching.
export const Route = createFileRoute("/app/")({
	component: () => null,
});
