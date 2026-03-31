import { createAuthClient } from "better-auth/react";

const API_BASE = import.meta.env.PROD
	? "https://api.crescend.ai"
	: "http://localhost:8787";

export const authClient = createAuthClient({
	baseURL: API_BASE,
});
