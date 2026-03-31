import { Hono } from "hono";
import type { Bindings, Variables } from "../lib/types";

const health = new Hono<{ Bindings: Bindings; Variables: Variables }>()
	.get("/", (c) => {
		return c.json({
			status: "ok",
			version: "2.0.0",
			stack: "hono",
		});
	});

export { health as healthRoutes };
