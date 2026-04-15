# CrescendAI

**"A teacher for every pianist."**

Multi-platform (iOS + web) practice companion for pianists.

## What CrescendAI Does

- Evaluates *how* a piano performance sounds (tone, dynamics, phrasing, pedaling) -- not just note accuracy like MIDI-based apps
- Uses dual encoders -- MuQ (audio, finetuned) and Aria (symbolic, 650M-param pretrained from EleutherAI) -- with gated fusion, outputting 6 teacher-grounded dimensions: dynamics, timing, pedaling, articulation, phrasing, interpretation
- Target users: self-learners (B2C), music educators (B2B), institutions (B2B)
- Competitors (Simply Piano, Flowkey, Piano Marvel) check note accuracy via MIDI; CrescendAI evaluates musical expression from audio

## Architecture

See `docs/architecture.md` for the full system design. Key points:
**Apps docs:** `docs/apps/00-status.md` through `docs/apps/05-ui-system.md` (each has status header)

## Model Strategy

- **Model v2:** MuQ (audio, pretrained on 160K hrs) + Aria (symbolic, pretrained on 820K MIDIs) with gated fusion
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
| `just deploy-api` | Deploy API worker to production |
| `just migrate-generate` / `just migrate-prod` | Drizzle migrations (generate SQL / apply to prod) |

## Package Managers

- Python: `uv` (not pip)
- JavaScript: `bun` (not npm)

## Coding Standards

- ALWAYS use model: "Sonnet 4.6" when creating and using subagents for search, reivew, or subagent driven development
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


# Code Intelligence Stack — Tool Routing

Three code-intelligence tools, each at a different layer. **Understanding first, then token efficiency, then speed.** Always pick the tool that matches the *question*, not the one you're most familiar with.

| Layer | Tool | Answers | When to reach for it |
|-------|------|---------|----------------------|
| Causal / architectural | **GitNexus** (MCP) | "What breaks if I change X?" "How does auth flow?" "Who calls this?" | Edits, refactors, debugging, exploring unfamiliar execution flows |
| Lexical / structural | **codedb** (MCP) | "Where is X defined?" "What symbols are in file Y?" "Find text Z across the repo" | Fast symbol lookup, file outlines, text search, reading files with line ranges |
| Terminal I/O | **rtk** (shell hook) | N/A — it's a transparent proxy | Automatic. Compresses git/ls/grep output. Zero agent decisions needed. |

## Routing rules (in priority order)

1. **About to edit a symbol?** GitNexus `impact` + `context` FIRST. This is mandatory (see GitNexus section below). codedb cannot answer "what breaks" — it has no call graph.
2. **Need to understand how something flows through the system?** GitNexus `query` + read the process resource. Do NOT start with grep.
3. **Need to find where a symbol is defined, or search for a string/pattern?** codedb `codedb_symbol` / `codedb_search` / `codedb_word`. Sub-ms, ~20 tokens per query vs thousands for ripgrep. Prefer over the Grep tool when the codebase is indexed.
4. **Need to read a file or see its symbols?** codedb `codedb_read` (supports line ranges) or `codedb_outline`. Prefer over Read for partial reads and symbol listings.
5. **Working in `apps/ios/` (Swift)?** codedb does NOT index Swift. Fall back to GitNexus (it covers Swift) or the Grep/Read tools.
6. **Running shell commands?** Just run them — rtk rewrites transparently via the hook. No agent action needed.

## Codedb MCP tools (16 total, most useful first)

| Tool | Use for |
|------|---------|
| `codedb_symbol` | Find definition of a symbol across the repo |
| `codedb_search` | Trigram-accelerated full-text search (supports regex + scoping) |
| `codedb_word` | O(1) exact word lookup via inverted index |
| `codedb_outline` | List functions/structs/imports in a file with line numbers |
| `codedb_read` | Read file content with line ranges (hash-cached) |
| `codedb_deps` | Reverse dependency graph (which files import this file) |
| `codedb_tree` | File tree with language + symbol counts |
| `codedb_bundle` | Batch up to 20 read-only queries in one call |
| `codedb_hot` | Recently modified files |
| `codedb_status` | Index status / sequence number |

## Tool-selection cheat sheet

- "Who calls `process_muq_result`?" → **GitNexus** `context`
- "Where is `SessionAccumulator` defined?" → **codedb** `codedb_symbol`
- "What breaks if I rename `try_generate_observation`?" → **GitNexus** `impact` (mandatory)
- "Find all uses of `AI_GATEWAY_BASE_URL`" → **codedb** `codedb_word` (faster/cheaper than Grep)
- "List functions in `apps/api/src/practice/piece_identify.rs`" → **codedb** `codedb_outline`
- "How does session synthesis get triggered?" → **GitNexus** `query` + process resource
- "Show me the last 200 lines of `handler.py`" → **codedb** `codedb_read` with line range
- "Safe rename across the repo" → **GitNexus** `rename` (dry_run first) — codedb cannot do this

## Codedb caveats

- **No Swift support** — use GitNexus or Grep for `apps/ios/`.
- Alpha software; snapshot format may change. `rm -rf ~/.codedb/` if index gets corrupted.
- `codedb_remote` hits a public third-party service (`codedb.codegraff.com`). Never point it at anything private — use it only for querying public GitHub deps.
- Telemetry on by default (aggregate counts only, no code/paths). To disable: `CODEDB_NO_TELEMETRY=1`.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **crescendai** (8821 symbols, 19903 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/crescendai/context` | Codebase overview, check index freshness |
| `gitnexus://repo/crescendai/clusters` | All functional areas |
| `gitnexus://repo/crescendai/processes` | All execution flows |
| `gitnexus://repo/crescendai/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

<!-- gitnexus:end -->
