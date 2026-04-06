# Catalog Search Tool Design

**Goal:** The teacher LLM can resolve a student's natural-language piece reference ("Chopin Waltz Op. 64 No. 2") to a catalog `piece_id` UUID, enabling it to chain into tools like `score_highlight` and `reference_browser` without asking the student for a UUID.
**Not in scope:** Full-text search, external music database lookup, piece request tracking/demand signals, catalog CRUD.

## Problem

The `score_highlight` tool requires `piece_id` (UUID). When a student says "can you highlight bars 1-2 of my Chopin waltz?", the teacher LLM has no way to resolve that to a UUID. It asks the student for a piece_id -- a terrible UX since students never know UUIDs.

In a practice session, the piece gets identified via AMT fingerprinting (`SessionBrain.pieceIdentification`), but:
1. Fresh chat conversations (no practice session) have zero piece context
2. The student may reference a DIFFERENT piece than the one currently being practiced
3. Even in-session, the identified `pieceId` is only in the DO state -- it's never injected into the teacher's system prompt for the chat path

## Solution (from the student's perspective)

The student says "highlight the first two bars of my Chopin waltz" and the teacher:
1. Calls `search_catalog` with `{ query: "Chopin waltz" }` (or `{ composer: "Chopin", title: "waltz" }`)
2. Gets back matching pieces with their UUIDs
3. Picks the best match (or asks the student to disambiguate if multiple)
4. Calls `score_highlight` with the resolved `piece_id`

In a practice session where the piece is already identified, the teacher already knows the `piece_id` from the dynamic context and skips step 1-3.

## Design

### Part A: `search_catalog` tool

A new tool in the `TOOL_REGISTRY` that queries the `pieces` table by composer and/or title using case-insensitive `ILIKE` matching. Returns up to 5 matching pieces with their UUIDs and metadata.

**Why a tool (not embedded in score_highlight):**
- `reference_browser` also needs piece_id resolution -- a shared tool avoids duplication
- The LLM can disambiguate before committing to an action (show 3 matches, ask "which one?")
- Follows the existing pattern: tools have single responsibilities

**Why ILIKE (not fuzzy/trigram):**
- The catalog is small (~242 pieces from ASAP). ILIKE on `composer` and `title` with `%query%` is sufficient.
- No pg_trgm extension needed (simpler infra).
- The LLM is good at reformulating queries ("Chopin" -> "chopin", "Op. 64" -> "op. 64") so basic substring matching works.

**Schema:**
```
search_catalog({
  query?: string,      // Free text search across composer+title
  composer?: string,   // Filter by composer name (ILIKE)
  title?: string,      // Filter by title (ILIKE)
})
```
At least one of `query`, `composer`, or `title` is required. Returns up to 5 matches: `{ pieceId, composer, title, barCount }`.

**Security:** Read-only parameterized Drizzle query on the `pieces` table. The catalog is already publicly accessible via `GET /api/scores` with no auth. No SQL injection risk -- Drizzle parameterizes all queries. No student data exposed.

### Part B: Piece context in chat dynamic context

When `buildChatFraming` is called, if the student has an active practice session with an identified piece, include the `pieceId`, `composer`, and `title` in the dynamic context XML so the teacher already knows the piece without needing a tool call.

**Flow:** `prepareChatContext` (in `chat.ts`) queries the student's most recent session for piece identification data, then passes it to `buildChatFraming`, which emits it as `<current_piece>` XML in the dynamic context.

**Why this helps:** In-session chats skip the `search_catalog` round-trip entirely. The teacher sees `<current_piece>piece_id=abc, composer=Chopin, title=Waltz Op. 64 No. 2</current_piece>` and uses the `piece_id` directly.

## Modules

### search_catalog tool (in tool-processor.ts)
- **Interface:** `processSearchCatalog(ctx, studentId, input) -> InlineComponent[]` -- returns a text component with search results
- **Hides:** ILIKE query construction, result limiting, null-field handling, structured JSON response
- **Tested through:** Zod schema validation tests + processToolUse integration test with mock DB

### buildChatFraming extension (in prompts.ts + chat.ts)
- **Interface:** `buildChatFraming(studentLevel, studentGoals, memoryContext, currentPiece?) -> string`
- **Hides:** XML formatting of piece context, null handling
- **Tested through:** buildChatFraming unit test with piece metadata input

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/services/tool-processor.ts` | Add `search_catalog` tool (schema, process fn, Anthropic schema, registry entry) | Modify |
| `apps/api/src/services/tool-processor.test.ts` | Add schema validation + pass-through tests for `search_catalog` | Modify |
| `apps/api/src/services/prompts.ts` | Extend `buildChatFraming` to accept optional piece metadata | Modify |
| `apps/api/src/services/chat.ts` | Query recent session for piece identification, pass to `buildChatFraming` | Modify |
| `apps/api/src/services/prompts.test.ts` | Add test for `buildChatFraming` with piece context | Create |

## Open Questions

- Q: Should `search_catalog` return a visible UI component (like a card) or just text for the LLM to interpret? Default: Text-only (`InlineComponent` with type `"text"` or empty components array with result in the tool_result content). The LLM interprets results and decides what to show the student. No UI card needed.
- Q: Should `search_catalog` also check `piece_requests` for demand tracking? Default: No -- out of scope. Demand tracking is a separate feature.
