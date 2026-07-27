# CrescendAI

`CLAUDE.md` is the canonical Claude Code context. Keep the sibling `AGENTS.md`
byte-identical as a compatibility mirror: edit `CLAUDE.md` first, then mirror
the exact content.

**"A teacher for every pianist."** CrescendAI is a web and iOS practice
companion that evaluates musical expression from piano audio, not only note
accuracy.

## Stage and safety

- **PRE-BETA, local-first, zero real users.** See `docs/project-stage.md`.
- A local merge to `main` is not a deployment. Never run `just deploy-api`,
  publish the web app, or change production resources unless the user requests
  that exact action.
- Local API development uses Postgres `crescendai_dev`. Apply local migrations
  with:

  ```bash
  cd apps/api
  DATABASE_URL="postgresql://jdhiman:postgres@localhost:5432/crescendai_dev" bun run migrate
  ```

  From `apps/api`, bare `bun run migrate` targets the hosted production
  database.
- "Ready" means local tests/checks, local services, and the relevant manual
  click-through are green.

## Tooling

- Python packages and scripts: `uv`, never `pip`.
- JavaScript/TypeScript packages and scripts: `bun`, never `npm`.
- Repository workflows use `just` (`just dev`, `just test-model`,
  `just test-api`, `just check-api`).
- ML experiments use Trackio.
- Prefer Serena symbol tools for code navigation; use sourcekit-lsp-backed
  Serena for Swift.

## Work tracking and isolation

- GitHub Issues in `Jai-Dhiman/crescendAI` are the canonical backlog.
- The WIP board is
  [CrescendAI — Now, Ready, Parked](https://github.com/users/Jai-Dhiman/projects/9):
  `Now <= 2`, `Ready <= 3`. Prioritization is human-lit; Claude Code performs
  all board bookkeeping with `gh project` or `gh api`.
- Do not ask the user to add, move, field, or archive Project items manually.
  Inspect the current counts before promotion, enforce the WIP limits, and
  update the Project in the same session when an issue is created, resumed,
  parked, completed, or closed.
- At session start, run:

  ```bash
  gh issue list --assignee @me --state open --json number,title,labels,updatedAt
  ```

  If the request matches an issue, read its body, comments, and latest `STATE:`.
  Otherwise ask before creating a new issue.
- One issue maps to one `issue-NNN-slug` branch and one isolated worktree.
  Reserve the primary checkout for orchestration and local merges; make edits
  inside `.worktrees/issue-NNN-slug`.
- If Claude Code's local primary-tree guard blocks an edit on `main` or
  `master`, enter the issue worktree; do not bypass the guard.
- Before ending work on an issue, post
  `STATE: <current state> Next: <concrete next step>`.
- Durable decisions belong in the authoritative docs below. Active state belongs
  in issues, not new summary/plan files.

## Automation boundary

- Dark-eligible: deterministic inventory, formatting, lint, typecheck, and test
  commands with fast, non-gameable pass/fail output.
- Human-lit: architecture, research-gate interpretation, issue closure, merges,
  deployments, production mutations, and destructive cleanup.
- Failed gates keep their original criteria and negative results. Do not turn a
  null result into a pass or an impossibility claim.

## Routing

- Product stage and architecture: `docs/project-stage.md`,
  `docs/architecture.md`.
- App status, pipeline, UI, capabilities, and evaluation:
  `docs/apps/00-status.md`, `02-pipeline.md`, `05-ui-system.md`,
  `06-capabilities.md`, `07-evaluation.md`.
- Model facts and research decisions: `docs/model/`; active measurement
  conventions live in `docs/model/claim-verifier-signed-d-conventions.md`.
- API TypeScript work must follow `apps/api/TS_STYLE.md`.
- Read the nearest scoped `CLAUDE.md` under `apps/`, `apps/ios/`, or `model/`
  before editing there.
- Claude Code feature workflow: `/brainstorm` -> `/plan` -> `/challenge` ->
  `/build` -> `/review` -> `/ship`. Bugs start with `/investigate`.
  Deployments are separate and require `/canary`.

Prefer explicit failures over silent fallbacks. Touch only lines required by the
task, and do not add documentation files unless requested.
