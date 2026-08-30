import { createFileRoute } from "@tanstack/react-router";

// Home renders here (#160, Task 16). app.tsx is the /app layout.
export const Route = createFileRoute("/app/")({
	component: () => null,
});
