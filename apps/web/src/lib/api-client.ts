import { hc } from "hono/client";
import type { AppType } from "../../../api/src/index";
import { API_BASE } from "./config";

// Typed RPC client with cookie credentials
export const client = hc<AppType>(API_BASE, {
	init: {
		credentials: "include",
	},
});
