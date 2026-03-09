import { createFileRoute } from "@tanstack/react-router";
import AppChat from "../components/AppChat";

export const Route = createFileRoute("/app/")({
	component: () => <AppChat />,
});
