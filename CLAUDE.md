# CrescendAI

**"A teacher for every pianist."**

Multi-platform (iOS + web) practice companion for pianists.

## Project Stage — PRE-BETA, local-first (see `docs/project-stage.md`)

**Zero real users.** "Shipped to main" = a local merge, NOT a deploy. Production deploy (`just deploy-api`, web) is a deliberate, separate, manual step — do NOT deploy as part of normal feature work. Local dev runs against local Postgres `crescendai_dev` (Hyperdrive `localConnectionString` in `apps/api/wrangler.toml`); apply migrations locally with `DATABASE_URL="postgresql://jdhiman:postgres@localhost:5432/crescendai_dev" bun run migrate` (bare `bun run migrate` targets the hosted PlanetScale prod DB — only at deploy time). "Ready" = local green (tests + `wrangler dev` + manual click-through). Caveat: the eval harness measures the legacy `synthesize()` path, not the production V6 harness — eval numbers are a legacy proxy until the eval is made V6-aware.

## What CrescendAI Does

- Evaluates *how* a piano performance sounds (tone, dynamics, phrasing, pedaling) -- not just note accuracy like MIDI-based apps
- Uses dual encoders -- MuQ (audio, finetuned) and Aria (symbolic, 650M-param pretrained from EleutherAI) -- as parallel streams (not gated fusion); each emits its own 6-dim quality scores plus a deterministic MPM-style feature extraction from AMT. The teacher LLM is the cross-modal reasoner; disagreement between streams is signal, not noise.
- Target users: self-learners (B2C), music educators (B2B), institutions (B2B)
- Competitors (Simply Piano, Flowkey, Piano Marvel) check note accuracy via MIDI; CrescendAI evaluates musical expression from audio

## Architecture

See `docs/architecture.md` for the full system design. Key points:
**Apps docs:** `docs/apps/00-status.md` through `docs/apps/05-ui-system.md` (each has status header)

## Model Strategy

- **Model v2:** MuQ (audio, pretrained on 160K hrs) + Aria (symbolic, pretrained on 820K MIDIs) as parallel streams; both stream outputs + MPM-style extracted features go to the teacher LLM
- **Score conditioning:** delta = z_perf - z_score when score MIDI available
- **Training:** PercePiano anchor (20%) + ordinal-dominated (80%) with T2 competition + T5 YouTube Skill data

## Development

Uses `just` (justfile) for dev commands. Install: `brew install just`.

| Command | What it starts |
|---------|---------------|
| `just dev` | All 4 services: MuQ (8000) + AMT (8001) + API (8787) + Web (3000) |
| `just dev-muq` | MuQ + API + Web (no AMT, faster startup) |
| `just dev-light` | API + Web only (uses production HF endpoints) |
| `just muq` / `just amt` / `just api` / `just web` | Individual services |
| `just fingerprint` | Generate N-gram index + rerank features from score library |
| `just test-model` / `just test-api` / `just check-api` | Tests and checks |
| `just chroma-eval-verify-smoke` | Fast sanity check (--skip-dtw); no real audio required; use in pre-commit / CI |
| `just chroma-eval-verify` | Full chroma-DTW eval with real audio + AMT pseudo-truth; **fails loudly if audio or pseudo-truth data missing** (run `just amt-regen-pseudo-truth` first) |
| `just chroma-eval-ratchet` | Promote last_run.json to baseline.json after a deliberate metric improvement |
| `just piece-id-feasibility` | Chroma piece-ID feasibility harness (Issue #21); run `just piece-id-feasibility-acquire` first to populate audio cache |
| `just deploy-api` | Deploy API worker to production |
| `just migrate-generate` / `just migrate-prod` | Drizzle migrations (generate SQL / apply to prod) |

## Package Managers

- Python: `uv` (not pip)
- JavaScript: `bun` (not npm)

## Issue Tracking

**GitHub Issues is the canonical backlog.** Repo: `Jai-Dhiman/crescendAI`. No other tracker.

### Session rituals (required)

- **At session start, before any work:** run `gh issue list --assignee @me --state open --json number,title,labels,updatedAt`. If the user's request matches an open issue, resume it (read its comments for the last `STATE:` line). If not, ask whether to open a new one.
- **At session end with progress made:** post `gh issue comment N --body "STATE: <one-line current state> Next: <one-line concrete next step>"`. Future sessions grep for `STATE:` to resume.

### Doc bloat rules

- `docs/implementation/*.md` is for **ephemeral scratch only** — never permanent content. If `/ship` sees a new `docs/implementation/*.md` file in the diff without a linked issue number, fail and ask.
- `docs/specs/` and `docs/plans/` are deleted by `/ship`. Anything that needs to outlive the merge belongs in `docs/apps/`, `docs/model/`, or `docs/architecture.md`.
- `MEMORY.md` is for **durable facts only** (decisions, preferences, gotchas, architecture). Active-work entries belong in GitHub issues, not memory.

### Labels in use

- `epic:agentic-teacher` — agentic teacher redesign sub-issues
- `epic:doc-cleanup` — documentation audit work
- `epic:local-prototype` — work blocking a functional local prototype
- `verification` — integration / end-to-end verification issues
- `blocked` — cannot proceed; blocker in body
- `needs-triage` — newly created, awaiting triage

### Verified gh command suite

```bash
# Create (URL → number: ${URL##*/})
URL=$(gh issue create --title "..." --body "..." --label "epic:X,needs-triage")
N=${URL##*/}

# Comment
gh issue comment N --body "STATE: ..."

# List filtered
gh issue list --label epic:agentic-teacher --state open --json number,title

# Native sub-issue (note: -F for typed int, NOT -f)
gh api -X POST /repos/Jai-Dhiman/crescendAI/issues/PARENT/sub_issues \
  -F "sub_issue_id=$(gh api /repos/Jai-Dhiman/crescendAI/issues/CHILD -q .id)"

# List sub-issues
gh api /repos/Jai-Dhiman/crescendAI/issues/N/sub_issues -q '.[] | "#\(.number) \(.title)"'

# Close
gh issue close N --reason completed --comment "Closed by <PR# or commit>"
```

### Branch ↔ issue mapping

- One issue = one branch = one PR. Branch name: `issue-NNN-short-slug`.
- PR body must include `Closes #NNN` so merge auto-closes the issue.
- Local merges (no PR) require explicit `gh issue close NNN` in `/ship`.
- **One session = one worktree (anti-pollution).** Concurrent sessions share ONE primary checkout, which can only hold one branch — so two sessions working at the repo root inevitably commit onto the same branch and mix work. Before your first code edit, create/enter an isolated worktree tied to the issue and do ALL editing there:
  ```bash
  BRANCH=issue-NNN-slug
  git worktree add ".worktrees/$BRANCH" -b "$BRANCH"   # .worktrees/ is gitignored
  cd ".worktrees/$BRANCH"
  ```
  `/build` already does this automatically. For `/investigate`, `/autoresearch`, and manual fixes, do it yourself. The primary checkout (repo root) is reserved for orchestration and `/ship`'s merge. A `PreToolUse` hook (`.claude/hooks/guard-primary-tree-edits.py`) **enforces** this: it blocks edits made from the primary checkout while on `main`/`master`. If it blocks you, you forgot to enter a worktree.

## Coding Standards

- Explicit exception handling over silent fallbacks
- No backup files when making fixes
- No emojis unless explicitly requested
- **TypeScript (API):** Follow `apps/api/TS_STYLE.md` for all code in `apps/api/`. Key rules: never destructure `c.env`, ServiceContext for DI, domain errors in services (no HTTPException), chain `.route()` for Hono RPC types, Zod validation with JSON error hooks, state versioning in DOs across awaits, `console.log(JSON.stringify({...}))` for logging.

## Observability

- **Error tracking:** Sentry across all surfaces (iOS, web, API)
  - iOS: `sentry-cocoa` SPM -- crash reporting, error capture, breadcrumbs
  - Web: `@sentry/react` -- client-side errors, API errors, WebSocket errors
  - API: `@sentry/cloudflare` SDK -- traces, breadcrumbs, error capture
- Sentry org: `crescendai` with projects: `crescendai-api`, `crescendai-web`, `crescendai-ios`
- Error logging: `console_error!` (Rust), `Sentry.captureException` (web), `SentrySDK.capture(error:)` (iOS)


## Swift in `apps/ios/`

Use Serena (`mcp__serena__*`, backed by sourcekit-lsp) for Swift symbol lookups; fall back to Grep/Read for files under `apps/ios/` if the language server misbehaves.
