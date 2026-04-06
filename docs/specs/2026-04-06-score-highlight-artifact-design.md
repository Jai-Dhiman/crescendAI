# Score Highlight Artifact Design

**Goal:** The teacher LLM can point at specific bars in the student's score with dimension-colored annotations, rendered as inline SVG snippets in the chat flow with expand-to-sidebar for full score view.

**Not in scope:**
- Custom MusicXML renderer / OSMD fork
- Server-side score image rendering
- Keyboard guide or reference browser artifacts (separate features)
- Score editing or annotation editing by the student
- Offline/cached score rendering
- Changes to observation pacing or filtering logic in the DO

## Problem

The `score_highlight` tool exists end-to-end at the API level (tool-processor.ts lines 158-217, registered in TOOL_REGISTRY) but has no frontend rendering -- it falls through to `PlaceholderCard` showing "score highlight (coming soon)". The teacher LLM can call the tool, but the student sees nothing useful.

The existing `ScorePanel.tsx` renders scores with OSMD but is hardcoded to a single Chopin Nocturne MXL (`/scores/chopin-nocturne-op9-no2.mxl`, line 270) and gated to DEV-only (`scorePanelStore.open()` returns immediately in production, line 39). It's driven by `MockSessionData`, not real tool output.

The `ScoreHighlightConfig` type in `types.ts:41` is an empty stub (`[key: string]: unknown`).

## Solution (from the user's perspective)

When the teacher generates a score highlight during a practice session or chat:
1. An **inline card** appears in the chat showing a cropped SVG snippet of the highlighted bars, with dimension-colored overlays and annotation text. Multiple highlight regions can appear in a single card.
2. Clicking **expand** opens the **right sidebar** (existing ScorePanel) showing the full score scrolled to the highlighted region with annotation markers.
3. If score data fails to load, the card degrades to text-only (dimension, bar range, annotation) -- the teaching content is never lost.

## Design

### Approach: Shared OSMD instance + SVG clipping

OSMD is expensive per instance (~200-300ms init + render, heavy SVG DOM). Instead of instantiating OSMD per inline card, a single **OSMD Manager** module renders each piece once in a hidden off-screen container and serves clipped SVG fragments to inline cards.

**Why this over alternatives:**
- Per-card OSMD instances: 5 highlights = 5x 200-300ms + 5x full SVG DOM. Rejected.
- Custom OSMD fork/lightweight renderer: Music notation layout is deeply complex (proportional spacing, voice alignment, beam grouping). 2-4 weeks of engineering for uncertain quality. Rejected.
- Static server-side images: Extra infrastructure, no interactivity. Rejected.

**Key trade-off chosen:** One-time 200-300ms render cost per piece per session (amortized across all highlights for that piece) in exchange for near-instant inline card rendering.

### Score data format: MXL in R2

Scores stored as compressed MusicXML (`.mxl`) in R2. OSMD consumes MXL natively. Smallest file size (~50-150KB). Industry standard.

**R2 key consolidation:** Currently two inconsistent paths exist (`scores/${piece_id}/score.json` in tool-processor.ts vs `scores/v1/${pieceId}.json` in scores.ts). Consolidate to `scores/v1/${pieceId}.mxl` via the existing `getPieceData` service function. The `GET /api/scores/:pieceId/data` endpoint switches Content-Type to `application/vnd.recordare.musicxml`.

### Multi-region highlights

A single tool call can declare multiple highlight regions (e.g., "compare bars 4-8 dynamics with bars 20-24 dynamics"). Each region has its own bar range, dimension, and annotation. The inline card renders all regions; the sidebar shows all annotation markers.

### Sidebar coexistence

The existing ScorePanel sidebar and the new inline artifact cards coexist and serve different purposes:
- **Inline card:** "The teacher is pointing at bars 12-16 right now" -- contextual, in chat flow
- **Sidebar:** "Look at the whole score" -- workspace, resizable, scrollable

Expanding an inline card drives the sidebar. The sidebar can also be opened independently (removing the DEV-only gate).

## Modules

### 1. OSMD Manager (`apps/web/src/lib/osmd-manager.ts`)
- **Interface:** `ensureRendered(pieceId: string): Promise<void>`, `clipBars(pieceId: string, startBar: number, endBar: number): SVGElement | null`, `getOsmdInstance(pieceId: string): OsmdInstance | null`
- **Hides:** OSMD dynamic import, lazy initialization, hidden container lifecycle, per-piece render caching, measureList traversal for bounding box computation, SVG node cloning and viewBox cropping
- **Tested through:** `ensureRendered` caching behavior (mock OSMD import), `clipBars` returns null for out-of-range bars
- **Depth verdict:** DEEP -- 3-method interface hides OSMD lifecycle, caching, SVG geometry, and DOM manipulation

### 2. ScoreHighlightCard (`apps/web/src/components/cards/ScoreHighlightCard.tsx`)
- **Interface:** `{ config: ScoreHighlightConfig, onExpand?: () => void, artifactId?: string }`
- **Hides:** OSMD Manager interaction, loading/error states, SVG fragment layout, dimension color mapping, text fallback rendering
- **Tested through:** Renders loading state, renders text fallback on error, expand button calls store
- **Depth verdict:** DEEP -- simple props interface, complex async rendering pipeline behind it

### 3. ScoreHighlightConfig (`apps/web/src/lib/types.ts`)
- **Interface:** `{ pieceId: string, highlights: Array<{ bars: [number, number], dimension: string, annotation?: string }> }`
- **Hides:** Nothing -- this is a data shape
- **Depth verdict:** N/A (type definition)

### 4. Score API client (`apps/web/src/lib/api.ts` -- new `api.scores` namespace)
- **Interface:** `api.scores.getData(pieceId: string): Promise<ArrayBuffer>`
- **Hides:** URL construction, auth headers, error handling, response type parsing
- **Depth verdict:** SHALLOW (justified: follows established `api.*` pattern, consistency > depth here)

### 5. ScorePanel adaptation (`apps/web/src/components/ScorePanel.tsx` + store)
- **Interface:** `scorePanelStore.openHighlight({ pieceId, highlights })` (new action alongside existing `open()`)
- **Hides:** OSMD Manager reuse, highlight-to-annotation mapping, scroll-to-bar logic
- **Tested through:** Store action correctly sets state
- **Depth verdict:** DEEP -- simple store action triggers complex OSMD + scroll + annotation pipeline

### 6. API tool schema update (`apps/api/src/services/tool-processor.ts`)
- **Interface:** Anthropic tool schema (unchanged tool name `score_highlight`, new input shape)
- **Hides:** Zod validation, catalog lookup, config assembly
- **Tested through:** Zod schema validation (positive/negative), processToolUse pass-through
- **Depth verdict:** DEEP -- simple tool call interface, validation + DB + R2 behind it

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/web/src/lib/types.ts` | Replace `ScoreHighlightConfig` stub with concrete interface | Modify |
| `apps/web/src/lib/osmd-manager.ts` | Shared OSMD instance manager with SVG clipping | New |
| `apps/web/src/lib/api.ts` | Add `api.scores.getData()` method | Modify |
| `apps/web/src/components/cards/ScoreHighlightCard.tsx` | Inline card with clipped SVG snippets | New |
| `apps/web/src/components/InlineCard.tsx` | Add `case "score_highlight"` routing | Modify |
| `apps/web/src/components/Artifact.tsx` | Add `score_highlight` branch in `getCollapsedProps` | Modify |
| `apps/web/src/stores/score-panel.ts` | Add `openHighlight()` action, remove DEV gate, accept real data shape | Modify |
| `apps/web/src/components/ScorePanel.tsx` | Remove hardcoded MXL, use OSMD Manager, accept highlight data | Modify |
| `apps/api/src/services/tool-processor.ts` | Update `score_highlight` schema to multi-region highlights, use `getPieceData` for R2 | Modify |
| `apps/api/src/services/tool-processor.test.ts` | Update existing tests for new schema shape | Modify |
| `apps/api/src/services/scores.ts` | Update R2 key to `.mxl`, update Content-Type | Modify |
| `apps/api/src/routes/scores.ts` | Update Content-Type header to `application/vnd.recordare.musicxml` | Modify |

## Open Questions

- Q: Should the OSMD Manager pre-render scores that appear in tool_result SSE events before the card mounts, or render lazily on card mount? Default: Lazy on mount (simpler, avoids wasted renders for cards the user never scrolls to).
- Q: What is the maximum number of highlight regions per tool call? Default: Cap at 5 (prevents LLM from generating absurdly large highlight cards).
- Q: Should the sidebar remember its last-opened highlight when toggled closed and reopened? Default: No -- closing clears state, reopening requires a new expand action.
