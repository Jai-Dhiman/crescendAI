# Tutor Tasks (End-to-End)

This document tracks the tasks to deliver the full Tutor retrieval + ACE feedback system with Cloudflare Workers AI, Vectorize, KV, and R2.

## Decisions (locked)

- Embeddings: Cloudflare Workers AI `@cf/google/embeddinggemma-300m` only (no fallbacks).
- Storage:
  - Vectorize: vectors + minimal chunk metadata for retrieval.
  - KV: canonical chunk JSON and small caches (TutorFeedback, ACE playbook, ACE events recent).
  - R2: distribution PDFs, ingestion manifests (NDJSON), ACE event logs (NDJSON), large assets.
  - D1: deferred (add later for auth/assignments/relational analytics if needed).
- Content format: `.txt` for chunking/embeddings; PDFs in R2 for students.

## Milestones

- M1: Retrieval backbone working (ingest -> vectorize/kv/r2; worker returns top-k chunks + citations).
- M2: ACE event capture + shared playbook evolution in KV; basic personalization hooks.
- M3: Tutor generation integrates Evaluator scores + retrieved chunks; returns actionable JSON feedback.
- M4: Ops/docs/QA solid; remove dead code; CI green.

## Tasks

### A) Repo structure + hygiene

- [ ] Create ingest input folder: `tools/ingest-kb/piano_pedagogy/` (move pedagogy here)
- [ ] Update ingest preview default to point to new folder
- [ ] Remove dead or duplicate code related to outdated ingestion paths
- [ ] Keep a single ingestion path (tools/ingest-kb), evolve into full CLI later

### B) Remove all fallbacks

- [ ] Remove OpenAI chat fallback in `server/src/tutor.rs` (fail fast if CF not configured)
- [ ] Remove OpenAI embedding fallback in `server/src/knowledge_base.rs` (fail fast)
- [ ] Delete unused env vars and references (e.g., `OPENAI_API_KEY`, `TUTOR_OPENAI_MODEL`)

### C) Cloudflare resources

- [ ] Determine embedding dimension for `@cf/google/embeddinggemma-300m` (via `wrangler ai run` sample)
- [ ] Create Vectorize index with that dimension and `cosine`
- [ ] Create KV namespace (Tutor KV)
- [ ] Create R2 bucket for PDFs/manifests (e.g., `crescendai-practice`)
- [ ] Configure `wrangler.toml` bindings (AI, KV, Vectorize, R2)

### D) Ingestion (local, cargo run)

- [ ] Parse `.txt` with simple header (id/title/tags/source/url) + body
- [ ] Optional sidecar `.meta.json` per doc (title, level, tags, pdf filename)
- [ ] Chunk text (configurable char-based, default ~1000 chars, ~200 overlap)
- [ ] Embed each chunk with CF AI (validate returned vector length)
- [ ] Upsert into Vectorize (vector_id = `doc_id::c{idx}`) with metadata
- [ ] Write chunk JSON into KV (`doc:{doc_id}:chunk:{idx}`)
- [ ] Upload PDF to R2 at `docs/{doc_id}/{filename}.pdf`
- [ ] Write ingestion manifest NDJSON to R2 (`ingest/manifests/{timestamp}-{doc_id}.ndjson`)
- [ ] `validate` command: check index dims, sample retrieval, and connectivity
- [ ] `purge` command: remove document vectors/KV/PDF

### E) Tutor Worker (retrieval API)

- [ ] POST `/tutor/retrieve` → embed query; query Vectorize; fetch chunk JSON from KV; return results + citations (include R2 pdf key)
- [ ] Optional GET `/tutor/doc/:doc_id/pdf` → stream from R2 (or use signed URLs)
- [ ] Strict errors (no silent fallback). Empty results → 200 with empty array

### F) ACE feedback loop

- [ ] Persist per-user ACE events: KV keys `ace:event:{user_id}:{ts}:{rand}` with TTL; append to R2 NDJSON daily
- [ ] GET `/tutor/ace/user/:user_id/recent?limit=50` → list recent events (best-effort from KV)
- [ ] Keep shared ACE playbook in KV (`ace_playbook`); already integrated with pipeline

### G) Security + secrets

- [ ] Use wrangler secrets for Worker; local `.env` for ingest CLI (gitignored)
- [ ] Minimal scopes for ingest token: Vectorize write, KV write, R2 write, AI run

### H) Observability + QA

- [ ] Structured logs (doc_id, chunk_id, vector_id, request_id, timings)
- [ ] Health endpoint checks AI/KV/Vectorize availability (non-invasive)
- [ ] QA checklist: dims match, sample retrieval relevance, R2 PDF availability, KV consistency

### I) Docs (concise)

- [ ] `docs/tutor/INGEST.md` (CLI usage, .env, sidecar schema)
- [ ] `docs/tutor/RETRIEVAL.md` (API shapes, citations, R2 path)
- [ ] `docs/tutor/ACE.md` (event schema, retention, playbook)
- [ ] `docs/tutor/OPERATIONS.md` (index recreate, tokens, backfill, troubleshooting)

### J) Web hooks

- [ ] Minimal client in web app to call `/tutor/retrieve`
- [ ] Send ACE events after user interactions

## Cloudflare setup (reference)

- Determine dims:
  - Run a sample embedding with Workers AI and inspect the vector length; use that for Vectorize index creation.
- Create resources:
  - Vectorize: create index with derived dims + cosine
  - KV: create namespace (Tutor KV)
  - R2: create bucket (e.g., `crescendai-practice`)
  - Wrangler: bind AI, KV, Vectorize, R2 in `wrangler.toml`

## Definition of Done

- Ingest CLI runs locally and populates Vectorize/KV/R2
- Retrieval endpoint returns relevant chunks with citations and PDF links
- ACE events persist and playbook updates in KV
- No provider fallbacks present; strict errors; docs concise; CI green
