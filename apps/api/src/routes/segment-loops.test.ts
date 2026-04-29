import { Hono } from "hono";
import { describe, expect, it } from "vitest";
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
