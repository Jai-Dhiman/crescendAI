# Project Stage — Pre-Beta, Local-First

**Status (2026-06-05): PRE-BETA. Zero real users.** Nothing here is serving customers yet. Read this before reasoning about "production", deploys, or databases.

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

## Synthesis path (V6-only as of #28)

The **V6 harness** is now the single synthesis path (`synthesizeV6` → the hook-driven two-phase compound loop in `apps/api/src/harness/`). The `HARNESS_V6_ENABLED` flag and the legacy single-prompt `synthesize()`/`buildSynthesisFraming()` path are **deleted**; the eval now replays through the same V6 DO path users hit (`docs/harness.md` principle #16).

History worth keeping: before #28, V6 Phase 2 was silently broken for **every** session — its forced-tool schema (`zodToJsonSchema(target:"openApi3")`) was rejected by the Anthropic API (HTTP 400, JSON Schema draft 2020-12), so V6 emitted no artifact and the eval was wired around it to the legacy path. Unnoticed only because pre-beta = 0 users. Fixed in #28 (draft-07 inlined schema + a Phase-2 validation-repair loop).

Open measurement item (NOT yet done):

- The locked `_SONNET_BASELINE` (#22) still reflects the **retired legacy path**. Re-locking it on V6 requires a live `wrangler dev` run over the 98-recording holdout (`run_eval.py --do-path`), which needs `ALLOW_EVAL_STUDENT_OVERRIDE`+`EVAL_SHARED_SECRET` in `.dev.vars` **and Anthropic credits**. Until that run, V6 synthesis is verified by unit tests + the live sparse-cold-start repro (3/3 valid artifacts), not the baseline.

See `docs/harness.md` for the V6 architecture and `docs/architecture.md` for the full system.
