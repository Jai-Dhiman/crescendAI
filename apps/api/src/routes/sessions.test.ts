// apps/api/src/routes/sessions.test.ts
import { Hono } from "hono";
import { describe, expect, it } from "vitest";
import type { PassageManifest } from "../services/passage-manifest";
import { sessionsRoutes } from "./sessions";

const testApp = new Hono().route("/api/sessions", sessionsRoutes);

describe("sessions routes", () => {
  it("GET /api/sessions/:id/passage returns 401 without auth", async () => {
    const res = await testApp.request(
      "/api/sessions/00000000-0000-0000-0000-000000000001/passage?bars=5-8",
    );
    expect(res.status).toBe(401);
  });
});
