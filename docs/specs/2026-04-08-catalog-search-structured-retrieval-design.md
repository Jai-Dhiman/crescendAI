# Catalog Search: Structured Retrieval Design

**Goal:** The `search_catalog` tool reliably disambiguates between pieces that share a composer and genre (e.g., "Waltz Op. 64 No. 2" vs "Waltz Op. 64 No. 3") by using exact integer match on opus and piece numbers instead of substring search.
**Not in scope:** Vector/semantic search, pg_trgm fuzzy matching, external music database lookup, alias tables, piece request demand tracking, catalog CRUD, catalog expansion beyond ASAP.

## Problem

The current `search_catalog` tool uses `ILIKE '%query%'` substring matching. This fails in two concrete ways:

**Failure 1 — Cross-column substring mismatch** (`tool-processor.ts:440-445`):
Student says "Chopin Ballade No. 1". Teacher calls `{ query: "Chopin Ballade" }`. SQL becomes `composer ILIKE '%Chopin Ballade%' OR title ILIKE '%Chopin Ballade%'`. The DB stores `composer="Chopin"` and `title="Ballades No. 1"` in separate columns — neither contains "Chopin Ballade" as a substring. Zero results.

**Failure 2 — Within-collection disambiguation breaks with fuzzy search**:
"Waltz Op. 64 No. 2" and "Waltz Op. 64 No. 3" produce near-identical ILIKE results. Any fuzzy approach (vector embeddings, pg_trgm) compresses semantic meaning — both pieces are Chopin waltzes from Op. 64 and will have near-identical representations. The "2" vs "3" carries zero weight in semantic space but is the entire discriminating signal.

The root mismatch: ILIKE and fuzzy approaches are lexical/semantic tools asked to solve a structured-identifier problem. Opus and piece numbers are integers. The right tool is exact integer match, not approximate text matching.

## Solution (from the student's perspective)

Student says "can you highlight bars 5-10 of the Chopin Waltz Op. 64 No. 2". Teacher:
1. Calls `search_catalog({ composer: "Chopin", opus_number: 64, piece_number: 2 })` — Claude Sonnet extracts structured slots from the student's words; it knows "Op. 64 No. 2" maps to `{opus_number: 64, piece_number: 2}`.
2. Gets back exactly one match with its UUID.
3. Calls `score_highlight` with the resolved `piece_id`.

No disambiguation dialog needed. No retry required. The LLM is the semantic layer (handles "Valse" = "Waltz", "Moonlight Sonata" = Op. 27 No. 2, etc.) — and the DB is the structured lookup layer.

## Design

### LLM as the semantic layer

Claude Sonnet already has the entire Western piano repertoire in its training data. It knows:
- "Moonlight Sonata" → Beethoven Op. 27 No. 2
- "Revolutionary Etude" → Chopin Op. 10 No. 12
- "Valse" → Waltz (French synonym)
- "that Bach WTC prelude" → catalogue_type = "wtc"

The LLM extracts structured slots from natural language **before** calling the tool. This makes the search layer simple and exact. This is "LLM-before-retrieval" — the LLM parses the query into structured form, then the DB does exact lookup. The opposite of RAG (retrieve then generate).

### Tool schema: structured fields replace free-form query

Old schema: `{ query?, composer?, title? }` — all ILIKE, forces LLM to choose which substring to pass.

New schema: `{ composer?, opus_number?, piece_number?, title_keywords?, query? }` — structured fields are the primary path; `query` is a last-resort fallback.

### Search logic: exact on integers, fuzzy on text

```
composer     → ILIKE '%Chopin%'           (fuzzy: composer names are short and well-known)
opus_number  → = 64                       (exact: integer, the discriminating feature)
piece_number → = 2                        (exact: integer, the discriminating feature)
title_keywords → ILIKE '%Ballade%' tokens (token-split fuzzy for genre words)
query        → token-split ILIKE across both fields (fallback only when no structured fields)
```

All provided fields are ANDed. Opus and piece numbers use exact Postgres `=` — not ILIKE. After filtering by composer (~20-40 candidates) and exact opus/piece number, at most 1-2 results remain.

### Adding structured columns to `pieces`

Three new nullable integer/text columns on the `pieces` table:
- `opus_number INTEGER` — e.g., 64 for "Op. 64", null when title has no opus
- `piece_number INTEGER` — e.g., 2 for "No. 2" or trailing dash-number in WTC titles
- `catalogue_type TEXT` — "op" | "wtc" | "bwv" | null — distinguishes opus systems

A one-time backfill script parses the existing 242 ASAP titles (which follow a regular naming convention) and writes these fields. New pieces inserted in the future should set these fields at insert time.

### ASAP title parsing

ASAP titles follow a regular pattern derived from directory paths:
- `Waltz Op. 64 No. 2` → opus_number=64, piece_number=2, catalogue_type="op"
- `Etudes Op. 10 No. 3` → opus_number=10, piece_number=3, catalogue_type="op"
- `Ballades No. 1` → opus_number=null, piece_number=1, catalogue_type=null
- `WTC I - Prelude - 1` → opus_number=null, piece_number=1, catalogue_type="wtc"
- `Arabesques` → all null (no structured identifier in title)

Regex patterns are sufficient: `Op\.\s*(\d+)` for opus, `No\.\s*(\d+)` for piece number, trailing `[-–]\s*(\d+)\s*$` for WTC piece numbers.

### Why not vector search or pg_trgm

**Vector search:** Encodes semantic similarity. "Waltz Op. 64 No. 2" and "Waltz Op. 64 No. 3" are semantically identical (same composer, same opus, same genre). Their cosine similarity will be >0.97. The discriminating feature ("2" vs "3") carries essentially zero weight in embedding space. Vector search solves language variant problems (Waltz/Valse) but cannot solve within-collection disambiguation.

**pg_trgm:** Trigram similarity for "No. 2" vs "No. 3" would produce nearly identical scores (the strings are 87% similar). Still cannot discriminate.

**Conclusion:** The discriminating features are integers. Use integer equality.

## Modules

### `catalog-parse.ts` — Pure title field extractor
- **Interface:** `parseTitleFields(title: string): TitleFields` where `TitleFields = { opusNumber: number | null, pieceNumber: number | null, catalogueType: string | null }`
- **Hides:** Regex matching for Op./No./WTC/BWV patterns, edge cases (WTC trailing dash-number, titles with no identifiers), parseInt with base 10
- **Tested through:** Direct unit tests on `parseTitleFields()` with representative ASAP title strings
- **Depth verdict:** DEEP — simple one-function interface hides all pattern-matching complexity

### `search_catalog` tool in `tool-processor.ts`
- **Interface:** `TOOL_REGISTRY.search_catalog` — same as existing, updated schema and process fn
- **Hides:** Condition building (integer eq vs ILIKE), token-split for title_keywords, fallback query logic, empty-result messaging, Drizzle query assembly
- **Tested through:** Zod schema validation tests + processToolUse integration tests with mock DB
- **Depth verdict:** DEEP — simple registry interface hides multi-branch condition builder and query strategy

### Backfill script `backfill-piece-fields.ts`
- **Interface:** Standalone runnable script: `bun run apps/api/src/scripts/backfill-piece-fields.ts`
- **Hides:** DB connection setup, batch update logic, uses `parseTitleFields` internally
- **Tested through:** `parseTitleFields` unit tests cover the core transformation; script is verified by running against dev DB
- **Depth verdict:** SHALLOW by design — it is one-time infrastructure, not a reused module

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/services/catalog-parse.ts` | New module: `parseTitleFields()` pure function | New |
| `apps/api/src/services/catalog-parse.test.ts` | Unit tests for parseTitleFields across ASAP title patterns | New |
| `apps/api/src/db/schema/catalog.ts` | Add `opusNumber`, `pieceNumber`, `catalogueType` columns | Modify |
| `apps/api/src/db/migrations/` | Generated migration SQL for three new columns | New (generated) |
| `apps/api/src/scripts/backfill-piece-fields.ts` | One-time script to populate new columns from existing titles | New |
| `apps/api/src/services/tool-processor.ts` | Replace `searchCatalogSchema` and `processSearchCatalog` with structured-retrieval versions; update `searchCatalogAnthropicSchema` description | Modify |
| `apps/api/src/services/tool-processor.test.ts` | Update search_catalog schema validation tests and behavior tests to match new schema | Modify |

## Open Questions

- Q: Should `catalogue_type = "wtc"` pieces have `opus_number` set to the WTC book number (1 or 2)?  Default: No — WTC Book I and II are not opus numbers. Leave `opus_number` null for WTC; use `catalogue_type = "wtc"` as the discriminator. If the teacher needs to search for "WTC I Prelude 1", it calls `{ title_keywords: "WTC", piece_number: 1 }`.
- Q: Should the backfill script be idempotent (skip pieces where fields are already set)?  Default: Yes — skip any piece where `opus_number IS NOT NULL OR piece_number IS NOT NULL`. Safe to run multiple times.
