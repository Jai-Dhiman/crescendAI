# Project Stage — Pre-Beta, Local-First

**Status (2026-08-04): PRE-BETA. Zero real users.** Nothing here is serving customers yet. Read this before reasoning about "production", deploys, or databases.

## What "production" and "shipped" mean right now

- **"Shipped to main" = a local merge to the `main` branch. It does NOT mean deployed.** New work lands on `main` locally and is verified locally. The live Worker at `api.crescend.ai` / `crescend.ai` is **not** auto-updated.
- **Production deploy is a deliberate, manual, gated step** (`just deploy-api`, web deploy). We deploy only after local verification passes, and only when we choose to. Because there are no users, there is no pressure to deploy continuously.
- Practical rule: **do not deploy as part of normal feature work.** Build → verify locally → stop. Deploy is its own decision.

## Databases — there are two, know which one you're touching

| DB | Where | Used by | When to migrate |
|---|---|---|---|
| `crescendai_dev` | local Postgres (`postgresql://…@localhost:5432/crescendai_dev`) | **local dev** — `wrangler dev` reaches it via Hyperdrive `localConnectionString` (`apps/api/wrangler.toml`) | every schema change, immediately, for local testing |
| PlanetScale Postgres | `…psdb.cloud` (the `DATABASE_URL` in `apps/api/.dev.vars`) | the eventual **production** Worker | only at deploy time, deliberately |

**Apply migrations locally** with the local URL, not the hosted one:

```bash
cd apps/api
DATABASE_URL="postgresql://jdhiman:postgres@localhost:5432/crescendai_dev" bun run migrate
```

`bun run migrate` (drizzle-kit) reads `process.env.DATABASE_URL`. With no override it would target the **hosted PlanetScale prod DB** — so always set the local URL when migrating for local work. The hosted DB is migrated only when deploying.

## Verification posture (pre-beta)

"Ready" = **local green**, not "deployed and healthy":
- Unit/integration test suites pass (`bunx vitest run` in `apps/api` and `apps/web`).
- The Worker bundles (`bunx wrangler deploy --dry-run`) and boots (`just dev` / `just dev-light`).
- The relevant flow is manually exercised in the local app (record a session, watch the behavior).

If local is green, we are **deploy-ready**. Pulling the trigger is a separate, explicit choice.

## Measurement posture

**Eval and production run the same code path (#28).** Synthesis always goes through `synthesizeV6` → the hook-driven two-phase compound loop in `apps/api/src/harness/`. The legacy single-prompt `synthesize()` path and the `HARNESS_V6_ENABLED` flag are deleted, so there is no longer an eval-only branch that can diverge from what users hit. The judge grades the full `SynthesisArtifact` (strengths, focus areas with severity, next-session focus), not just the delivered headline.

Two caveats remain:

- **The locked `_SONNET_BASELINE` (#22) is stale.** It was measured on the now-deleted legacy path, so it is not comparable to anything the current system produces. A fresh glm baseline over the 98-holdout has **not** been locked yet — that is #28 item 4, blocked on Workers AI daily quota.
- **A failed synthesis is silent, not flagged.** When V6 fails validation or produces no artifact, the DO returns without sending a `synthesis` message at all. The eval detects this as `SessionResult.synthesis is None` plus a "No synthesis received" entry in `SessionResult.errors`. There is deliberately no `is_fallback` boolean — the only emitter hardcoded it to `false`, so it could never report a failure.

See `docs/harness.md` for the V6 architecture and `docs/architecture.md` for the full system.
