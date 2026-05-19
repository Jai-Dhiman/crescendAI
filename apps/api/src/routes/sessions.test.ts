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

  it("GET /:id/passage returns manifest when DO returns 200", async () => {
    const cannedManifest: PassageManifest = {
      source: { kind: "session", sessionId: "00000000-0000-0000-0000-000000000001" },
      pieceId: "chopin.ballades.1",
      bars: [5, 8],
      chunks: [
        { url: "https://api/c1", chunkIndex: 1, durationSec: 15 },
      ],
      startOffsetSec: 1.0,
      endOffsetSec: 13.0,
      barTimeline: [
        { bar: 5, tSec: 0 },
        { bar: 6, tSec: 4 },
        { bar: 7, tSec: 8 },
        { bar: 8, tSec: 12 },
      ],
    };

    const stubFetch = async (_: Request) =>
      new Response(JSON.stringify(cannedManifest), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });

    const env = {
      SESSION_BRAIN: {
        idFromName: () => ({}),
        get: () => ({ fetch: stubFetch }),
      },
    } as unknown as Record<string, unknown>;

    const dbStub = {
      query: {
        sessions: {
          findFirst: async () => ({ id: "00000000-0000-0000-0000-000000000001", studentId: "student-1" }),
        },
      },
    };

    const testAppAuthed = new Hono()
      .use("*", async (c, next) => {
        (c as unknown as { set: (k: string, v: unknown) => void }).set("studentId", "student-1");
        (c as unknown as { set: (k: string, v: unknown) => void }).set("db", dbStub);
        await next();
      })
      .route("/api/sessions", sessionsRoutes);

    const res = await testAppAuthed.request(
      "/api/sessions/00000000-0000-0000-0000-000000000001/passage?bars=5-8",
      {},
      env,
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as PassageManifest;
    expect(body.pieceId).toBe("chopin.ballades.1");
    expect(body.barTimeline).toHaveLength(4);
  });

  it("GET /:id/passage propagates 409 from DO", async () => {
    const stubFetch = async () =>
      new Response(JSON.stringify({ error: "no_alignment" }), { status: 409 });
    const env = {
      SESSION_BRAIN: { idFromName: () => ({}), get: () => ({ fetch: stubFetch }) },
    } as unknown as Record<string, unknown>;

    const dbStub = {
      query: {
        sessions: {
          findFirst: async () => ({ id: "00000000-0000-0000-0000-000000000001", studentId: "student-1" }),
        },
      },
    };

    const testAppAuthed = new Hono()
      .use("*", async (c, next) => {
        (c as unknown as { set: (k: string, v: unknown) => void }).set("studentId", "student-1");
        (c as unknown as { set: (k: string, v: unknown) => void }).set("db", dbStub);
        await next();
      })
      .route("/api/sessions", sessionsRoutes);

    const res = await testAppAuthed.request(
      "/api/sessions/00000000-0000-0000-0000-000000000001/passage?bars=5-8",
      {},
      env,
    );
    expect(res.status).toBe(409);
  });
});
