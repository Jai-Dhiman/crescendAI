# Apps scope

Read the repository-root `AGENTS.md` first.

## Boundaries

- `api/` is the Cloudflare Worker API. Follow `api/TS_STYLE.md` for every
  TypeScript change; it is authoritative for Hono routing, dependency
  injection, domain errors, validation, Durable Object state, and logging.
- `web/` is the browser client. Treat its package manifest and source as
  authoritative; do not copy endpoint or component inventories into context
  files.
- `ios/` has additional constraints in `ios/AGENTS.md`.
- Use `bun` for API and web dependencies and scripts.

## Verification and safety

- Common local commands: `just api`, `just web`, `just test-api`,
  `just check-api`.
- Test against local Postgres `crescendai_dev`. Bare migrations and all deploy
  commands are production mutations and remain human-lit.
- Preserve explicit errors across service boundaries. Do not hide API,
  WebSocket, inference, or persistence failures with client-side fallbacks.
- Capture errors using the surface’s existing Sentry integration; follow nearby
  code rather than duplicating SDK recipes here.

## Routing

- Current product state: `docs/apps/00-status.md`.
- Cross-surface pipeline and ownership: `docs/apps/02-pipeline.md`.
- UI system: `docs/apps/05-ui-system.md`.
- Capabilities and evaluation: `docs/apps/06-capabilities.md`,
  `docs/apps/07-evaluation.md`.
- System-wide architecture: `docs/architecture.md`.

Architecture choices, API contracts, deploys, and manual click-through verdicts
are human-lit. Formatting, typechecks, and deterministic tests may run dark.
