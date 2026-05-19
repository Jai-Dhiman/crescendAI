// apps/api/src/routes/sessions.ts
import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import { NotFoundError } from "../lib/errors";

const barsRegex = /^(\d+)-(\d+)$/;

const passageQuerySchema = z.object({
  bars: z
    .string()
    .regex(barsRegex, { message: "bars must be in form 'N-M' with integers" })
    .transform((s) => {
      const m = barsRegex.exec(s);
      if (!m) throw new Error("unreachable after regex validation");
      return [Number(m[1]), Number(m[2])] as [number, number];
    })
    .refine(([s, e]) => s >= 1 && e >= s, {
      message: "bars start must be >= 1 and <= end",
    }),
});

const sessionIdParamSchema = z.object({ id: z.string().uuid() });

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>().get(
  "/:id/passage",
  validate("param", sessionIdParamSchema),
  validate("query", passageQuerySchema),
  async (c) => {
    requireAuth(c.var.studentId);
    const { id: sessionId } = c.req.valid("param");
    const { bars } = c.req.valid("query");

    const session = await c.var.db.query.sessions.findFirst({
      where: (s, { eq: e, and: a }) =>
        a(e(s.id, sessionId), e(s.studentId, c.var.studentId!)),
    });
    if (!session) throw new NotFoundError("session", sessionId);

    const doId = c.env.SESSION_BRAIN.idFromName(sessionId);
    const stub = c.env.SESSION_BRAIN.get(doId);
    const doRes = await stub.fetch(
      new Request(
        `https://do/passage?bars=${bars[0]}-${bars[1]}&sessionId=${sessionId}`,
        { method: "GET" },
      ),
    );
    if (doRes.status === 409) {
      return c.json({ error: "passage_unavailable" }, 409);
    }
    if (!doRes.ok) {
      return c.json({ error: "internal" }, 500);
    }
    const manifest = (await doRes.json()) as unknown;
    return c.json(manifest, 200);
  },
);

export { app as sessionsRoutes };
