import { Hono } from "hono";
import { describe, expect, it } from "vitest";
import { app } from "../index";
import { segmentLoopsRoutes } from "./segment-loops";

const testApp = new Hono().route("/api/segment-loops", segmentLoopsRoutes);

describe("segment-loops routes — auth guard", () => {
  it("POST /api/segment-loops/:id/accept returns 401 without auth", async () => {
    const res = await testApp.request(
      "/api/segment-loops/00000000-0000-0000-0000-000000000001/accept",
      { method: "POST" },
    );
    expect(res.status).toBe(401);
  });

  it("POST /api/segment-loops/:id/decline returns 401 without auth", async () => {
    const res = await testApp.request(
      "/api/segment-loops/00000000-0000-0000-0000-000000000001/decline",
      { method: "POST" },
    );
    expect(res.status).toBe(401);
  });

  it("POST /api/segment-loops/:id/dismiss returns 401 without auth", async () => {
    const res = await testApp.request(
      "/api/segment-loops/00000000-0000-0000-0000-000000000001/dismiss",
      { method: "POST" },
    );
    expect(res.status).toBe(401);
  });
});

const mockEnv = {
  SENTRY_DSN: "",
  HYPERDRIVE: { connectionString: "postgresql://fake:fake@localhost:5432/fake" },
  ALLOWED_ORIGIN: "http://localhost:3000",
  BETTER_AUTH_SECRET: "test-secret",
} as unknown as Record<string, unknown>;

describe("segment-loops routes — mounted", () => {
  it("POST /api/segment-loops/:id/accept returns 401 (auth guard active via main app)", async () => {
    const res = await app.request(
      "/api/segment-loops/00000000-0000-0000-0000-000000000001/accept",
      { method: "POST" },
      mockEnv,
    );
    expect(res.status).toBe(401);
  });
});
