import { hc } from "hono/client";
import type { AppType } from "../../../api/src/index";

const API_BASE = import.meta.env.PROD
	? "https://api.crescend.ai"
	: "http://localhost:8787";

// Typed RPC client with cookie credentials
export const client = hc<AppType>(API_BASE, {
	init: {
		credentials: "include",
	},
});
