# Citable RAG System for Piano Pedagogy Feedback

## Overview

This document outlines the system design for CrescendAI's natural language feedback generation with Retrieval-Augmented Generation (RAG). The system provides personalized, citable piano performance feedback grounded in authoritative pedagogy sources.

### Goals

1. Generate natural language feedback based on 19-dimension performance predictions
2. Ground all advice in authoritative piano pedagogy sources
3. Provide NotebookLM-style citations that link directly to source material
4. Support both historical texts (public domain) and modern masterclass content (YouTube)
5. Resist semantic collapse through hybrid retrieval (BM25 + dense vectors)

### Non-Goals

- Real-time transcription of user-uploaded audio (handled by ML inference pipeline)
- Licensed/copyrighted content ingestion (future phase)
- Multi-language support (English only for v1)

---

## Architecture Overview

```
                                 USER REQUEST
                                      |
                                      v
                    +----------------------------------+
                    |         ANALYSIS REQUEST         |
                    |  - Performance audio URL         |
                    |  - Performer/Piece metadata      |
                    +----------------------------------+
                                      |
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
        +-------------------+               +-------------------+
        |   HF INFERENCE    |               |  HYBRID RETRIEVAL |
        |   (ML Predictions)|               |   (RAG Context)   |
        +-------------------+               +-------------------+
                    |                                   |
                    |   19-dimension scores             |   Top-5 pedagogy chunks
                    |   (timing, legato, etc.)          |   with citation metadata
                    |                                   |
                    +-----------------+-----------------+
                                      |
                                      v
                    +----------------------------------+
                    |       WORKERS AI (LLM)           |
                    |  - Llama 3.3 70B Instruct        |
                    |  - Prompt: predictions + chunks  |
                    |  - Output: cited feedback        |
                    +----------------------------------+
                                      |
                                      v
                    +----------------------------------+
                    |       CITATION PARSING           |
                    |  - Extract [1], [2] references   |
                    |  - Map to source metadata        |
                    |  - Generate clickable links      |
                    +----------------------------------+
                                      |
                                      v
                    +----------------------------------+
                    |         LEPTOS FRONTEND          |
                    |  - Rendered feedback with links  |
                    |  - Citation panel on hover       |
                    |  - Sources footer                |
                    +----------------------------------+
```

---

## Data Model

### Pedagogy Chunk Schema

Each chunk represents a discrete piece of pedagogical content with full citation metadata.

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | UUID, primary key across both indexes |
| `text` | string | Raw chunk text (400-512 tokens) |
| `text_with_context` | string | Text with injected context header (used for embedding) |
| `source_type` | enum | `book`, `letter`, `masterclass`, `journal` |
| `source_title` | string | Title of source work |
| `source_author` | string | Author or speaker name |
| `source_url` | string | Original URL (Gutenberg, YouTube, etc.) |
| `page_number` | int? | For books/PDFs |
| `section_title` | string? | Chapter or section heading |
| `paragraph_index` | int? | 0-indexed paragraph within page/section |
| `char_start` | int? | Character offset for precise highlighting |
| `char_end` | int? | End character offset |
| `timestamp_start` | float? | Seconds (for video/audio) |
| `timestamp_end` | float? | End timestamp |
| `speaker` | string? | For masterclasses with multiple speakers |
| `composers` | string[] | Composers mentioned/relevant |
| `pieces` | string[] | Specific pieces mentioned |
| `techniques` | string[] | Technique keywords (legato, pedaling, etc.) |
| `ingested_at` | timestamp | When chunk was added |
| `source_hash` | string | SHA256 of content for deduplication |

### Context Header Injection

Before embedding, each chunk receives a context header to improve semantic retrieval:

```
[Source: Piano Mastery by Harriette Brower, p.45]
[Context: Chopin - Nocturne Op. 9 No. 2 - Legato, Rubato]

The secret of Chopin's legato lies not in finger pressure alone...
```

This ensures embeddings capture source attribution and musical context.

---

## Storage Architecture

### Dual-Index Strategy

Two parallel indexes serve different retrieval needs:

| Storage | Purpose | Query Type |
|---------|---------|------------|
| **D1 (SQLite)** | BM25 full-text search + metadata | Keyword matches, exact terms |
| **Vectorize** | Dense semantic search (BGE-768) | Conceptual similarity |

### D1 Schema

**Main Table**: `pedagogy_chunks`

- Stores all chunk content and metadata
- Primary key: `chunk_id`
- Unique constraint: `source_hash` (prevents duplicates)

**FTS5 Virtual Table**: `pedagogy_chunks_fts`

- Indexes: `text`, `source_title`, `source_author`, `composers`, `pieces`, `techniques`
- Enables BM25 ranking for keyword queries
- Kept in sync via INSERT/DELETE triggers

**Indexes**:

- `idx_chunks_composers` - Filter by composer
- `idx_chunks_techniques` - Filter by technique
- `idx_chunks_source_type` - Filter by source type

### Vectorize Configuration

- **Index Name**: `crescendai-piano-pedagogy`
- **Embedding Model**: `@cf/baai/bge-base-en-v1.5` (768 dimensions)
- **Metric**: Cosine similarity
- **Metadata Fields**: `composer` (filterable), `source_type` (filterable)

---

## Hybrid Retrieval Strategy

### Why Hybrid?

Pure vector search has limitations:

- **Semantic collapse**: At scale, similar documents cluster together, degrading precision
- **Keyword blindness**: Misses exact matches (e.g., "Chopin's rubato" might retrieve Liszt content)
- **BGE length collapse**: Longer texts cluster together regardless of content

Hybrid retrieval combines:

- **BM25**: Catches exact keyword matches, composer names, piece titles
- **Dense vectors**: Captures semantic meaning, related concepts

### Reciprocal Rank Fusion (RRF)

RRF merges rankings without score normalization:

```
RRF_score(doc) = sum over rankings R of: 1 / (k + rank_R(doc))
```

Where `k = 60` (standard constant that dampens high-rank dominance).

**Process**:

1. Query D1 FTS5 for top-20 BM25 results
2. Query Vectorize for top-20 semantic results
3. Compute RRF score for each unique chunk
4. Sort by combined RRF score
5. Return top-5 with full metadata

### Retrieval Query Construction

Given a performance analysis request, construct retrieval query from:

1. **Composer/Piece context**: "Chopin Nocturne Op. 9 No. 2"
2. **Weak dimensions**: dimensions scoring below 0.5
3. **Technique keywords**: mapped from dimension names

Example query for a Chopin performance with weak legato (0.35) and pedaling (0.42):

```
Chopin Nocturne legato pedaling technique singing tone
```

Optional filters:

- `composer_filter`: Prioritize composer-specific content
- `technique_filter`: Focus on specific weak areas

---

## Content Sources

### Public Domain (Phase 1)

| Source | Type | Content | URL |
|--------|------|---------|-----|
| Piano Mastery (Brower) | Book | Interviews with Paderewski, Hofmann, Lhévinne, etc. | Project Gutenberg |
| Beethoven's Letters | Letters | Primary source on interpretation | Project Gutenberg |
| C.P.E. Bach Essay | Book | Foundational keyboard technique (J.S. Bach's method via his son) | Internet Archive |
| Czerny on Beethoven | Book | Performance practice from Beethoven's student | Public domain |
| IMSLP Pedagogy Texts | Books | Various historical methods | imslp.org |

### YouTube Masterclasses (Phase 1)

Curated list of high-quality, transcribable masterclasses:

| Channel/Source | Content Type | Composers Covered |
|----------------|--------------|-------------------|
| Archived Horowitz classes | Historical masterclass | Chopin, Liszt, Rachmaninoff |
| Josh Wright | Modern pedagogy | General technique, Chopin |
| Tiffany Poon | Practice insights | Bach, Chopin |
| Pianist Magazine | Competition masterclasses | Various |

**Selection Criteria**:

- Clear audio (accurate auto-transcription)
- Focused pedagogical content (not just performance)
- Specific technique discussions
- Named piece/composer context

### Licensed Content (Future Phase)

| Source | Content | Licensing Approach |
|--------|---------|-------------------|
| Eigeldinger "Chopin: Pianist and Teacher" | Definitive Chopin pedagogy | Publisher license |
| Liszt Masterclass Notes (Göllerich) | 100+ documented classes | Academic use |
| Competition judging criteria | Evaluation frameworks | Direct partnership |

---

## Ingestion Pipeline

### Overview

```
Source Documents
      |
      v
+------------------+
|  EXTRACTION      |
|  - PDF parsing   |
|  - HTML scraping |
|  - YouTube API   |
+------------------+
      |
      v
+------------------+
|  PREPROCESSING   |
|  - Caption correction (musical terms)
|  - Composer/piece detection
|  - Technique tagging
+------------------+
      |
      v
+------------------+
|  CHUNKING        |
|  - 400-512 tokens
|  - 15% overlap
|  - Context header injection
+------------------+
      |
      v
+------------------+
|  DUAL INDEXING   |
|  - D1 insert + FTS trigger
|  - Vectorize upsert
+------------------+
```

### YouTube Transcript Processing

1. **Fetch transcript**: Use `youtube-transcript-api` Python library
2. **Prefer manual captions**: Fall back to auto-generated if unavailable
3. **Correct musical terms**: Map common auto-caption errors

| Correct Term | Common Mistakes |
|--------------|-----------------|
| rubato | "robot o", "rub auto", "rue bato" |
| legato | "leg auto", "legate o" |
| staccato | "stack auto", "stock otto" |
| Chopin | "show pan", "shopping" |
| Rachmaninoff | "rock man in off" |

1. **Preserve timestamps**: Each chunk stores `timestamp_start` and `timestamp_end`
2. **Detect context**: Extract composer/piece mentions from surrounding text

### Book/PDF Processing

1. **Parse structure**: Extract chapter > section > paragraph hierarchy
2. **Preserve page numbers**: Map paragraphs to source pages
3. **Detect musical context**: Regex patterns for composer names, opus numbers
4. **Handle historical language**: Flag archaic terms for LLM context

### Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Target size | 400-512 tokens | Balances context vs. precision |
| Overlap | 15% (~60-75 tokens) | Preserves cross-boundary concepts |
| Separator hierarchy | `\n\n` > `\n` > `.` > ` ` | Respects paragraph boundaries |
| Min chunk size | 100 tokens | Avoids fragment chunks |

### Deduplication

- Compute SHA256 hash of chunk text
- Store as `source_hash` with unique constraint
- Skip insertion if hash exists (allows re-running ingestion)

---

## Feedback Generation

### LLM Integration

**Model**: Cloudflare Workers AI - `@cf/meta/llama-3.3-70b-instruct-fp8-fast`

**Prompt Structure**:

```
You are an expert piano teacher providing feedback on a student's performance.

## Performance Analysis
Performer: {performer_name}
Piece: {piece_title} by {composer}

Dimension Scores (0.0-1.0 scale):
- Timing: {score}
- Articulation (Legato): {score}
- Pedal Clarity: {score}
... (all 19 dimensions)

## Reference Sources
[1] {source_title} by {author}, p.{page}
"{chunk_text}"

[2] {source_title} by {author} ({timestamp})
"{chunk_text}"

... (top 5 retrieved chunks)

## Instructions
Write 2-3 paragraphs of personalized feedback.

CRITICAL: Include inline citations using [1], [2], etc. when referencing
advice from the sources. Every specific piece of advice should be grounded
in a source.

Focus on:
1. One specific strength to celebrate (with citation if applicable)
2. One or two areas for improvement with actionable practice suggestions
3. A composer-specific insight relevant to this piece

Write in an encouraging but specific tone. Be concrete about what you
observed in the performance data.
```

### Response Parsing

1. **Extract citation markers**: Regex for `\[(\d+)\]` patterns
2. **Map to source metadata**: Link each `[N]` to the Nth retrieved chunk
3. **Generate clickable elements**: Replace markers with interactive buttons
4. **Build sources footer**: List all cited sources with full attribution

### Feedback Structure

**Output Schema**:

| Field | Type | Description |
|-------|------|-------------|
| `html` | string | Rendered HTML with citation buttons |
| `plain_text` | string | Raw text (for accessibility, copying) |
| `citations` | Citation[] | Array of citation metadata |

**Citation Schema**:

| Field | Type | Description |
|-------|------|-------------|
| `number` | int | Citation number (1-indexed) |
| `source_type` | string | book, masterclass, letter, journal |
| `title` | string | Source title |
| `author` | string | Author/speaker |
| `url` | string? | Link to source (YouTube, Gutenberg, etc.) |
| `page_number` | int? | For books |
| `timestamp_start` | float? | For video (seconds) |

---

## UI/UX Design

### Feedback Display

**Main Feedback Area**:

- Rendered prose with inline `[1]`, `[2]` citation markers
- Citations styled as subtle buttons (not disruptive to reading flow)
- Hover: Show tooltip with source preview
- Click: Open citation panel or jump to source

**Citation Panel** (on click):

- Slides in from right or bottom
- Shows: source title, author, page/timestamp
- "View Source" button links to original (YouTube timestamp, Gutenberg page)
- Close button returns to feedback

**Sources Footer**:

- Listed below main feedback
- Format varies by source type:
  - Book: `[1] Piano Mastery by Harriette Brower, p.45`
  - Video: `[2] Horowitz Masterclass - Chopin Ballade (14:32)`
- Each source is a clickable link

### Video Citation UX

For YouTube sources, link includes timestamp parameter:

```
https://youtube.com/watch?v={video_id}&t={timestamp_seconds}
```

Clicking jumps directly to the relevant moment in the masterclass.

### Accessibility

- `plain_text` field available for screen readers
- Citations have `aria-label` describing the source
- Keyboard navigation for citation panel
- High contrast for citation markers

---

## Semantic Collapse Mitigation

### The Problem

Single-vector embeddings have fundamental limits:

- 768-dim embeddings break down at ~1-2M documents
- Inter-document similarity > 0.65 causes retrieval quality collapse
- BGE specifically has length-induced clustering issues

### Mitigation Strategies

1. **Hybrid Retrieval**: BM25 catches exact matches vectors miss
2. **Context Headers**: Injected headers differentiate similar content
3. **Metadata Filtering**: Pre-filter by composer reduces similarity clustering
4. **Corpus Monitoring**: Track inter-document similarity, alert if > 0.60

### Monitoring Metrics

| Metric | Threshold | Action |
|--------|-----------|--------|
| Avg inter-document similarity | > 0.60 | Review chunking strategy |
| Retrieval recall@5 | < 0.7 | Add more diverse sources |
| Citation usage rate | < 80% | Improve prompt instructions |
| User feedback rating | < 4.0 | Review content quality |

### Future Scaling Options

If corpus grows beyond safe thresholds:

- **ColBERT**: Multi-vector embeddings with late interaction
- **Cross-encoder reranking**: Score query-doc pairs directly
- **Dimension increase**: Upgrade to 1024-dim model

---

## Caching Strategy

### KV Cache Layers

| Cache | TTL | Key Pattern | Content |
|-------|-----|-------------|---------|
| Analysis results | 24h | `analysis:{performance_id}` | Full predictions + feedback |
| Retrieval results | 1h | `retrieval:{query_hash}` | Top-5 chunks (no metadata) |
| Embeddings | 7d | `embed:{text_hash}` | Query embeddings |

### Cache Invalidation

- New content ingestion: Clear retrieval cache
- Model update: Clear all caches
- Manual purge: Admin endpoint

---

## Quality Assurance

### Content Quality Gates

Before ingestion:

- [ ] Source is authoritative (published, recognized expert)
- [ ] Content is pedagogical (teaching, not just performance)
- [ ] Transcript quality > 90% accuracy (for YouTube)
- [ ] No duplicate content in corpus

### Feedback Quality Metrics

| Metric | Measurement | Target |
|--------|-------------|--------|
| Citation grounding | % of advice with citation | > 80% |
| Citation accuracy | Manual review of source match | > 95% |
| Feedback relevance | User rating (1-5) | > 4.0 |
| Hallucination rate | Claims not in sources | < 5% |

### Testing Strategy

1. **Retrieval tests**: Known queries should return expected chunks
2. **Citation tests**: LLM output should correctly reference sources
3. **E2E tests**: Full analysis flow returns valid cited feedback
4. **Regression tests**: New content doesn't degrade existing quality

---

## Implementation Phases

### Phase 1: Infrastructure

- Create D1 schema with FTS5 tables
- Configure Vectorize index with metadata fields
- Update wrangler.toml bindings
- Create migration scripts

### Phase 2: Ingestion Pipeline

- Build Python ingestion scripts
- Implement YouTube transcript extraction
- Implement Gutenberg HTML parsing
- Curate initial source list (10-15 sources)
- Run initial ingestion, validate chunk quality

### Phase 3: Retrieval Layer

- Implement BM25 search function
- Implement vector search function
- Implement RRF merge algorithm
- Create hybrid_retrieve endpoint
- Add retrieval caching

### Phase 4: Feedback Generation

- Build LLM prompt template
- Implement citation parsing
- Create feedback response schema
- Integrate with existing analysis flow
- Add feedback caching

### Phase 5: UI Integration

- Create CitedFeedback component
- Implement citation hover/click behavior
- Style sources footer
- Add video timestamp links
- Accessibility review

### Phase 6: Quality & Monitoring

- Implement retrieval quality metrics
- Add citation coverage tracking
- Create admin dashboard for corpus stats
- Set up alerting for quality degradation

---

## Open Questions

1. **Rate limiting**: How to handle Workers AI rate limits for high-traffic periods?
2. **Fallback behavior**: What to show if RAG retrieval fails? (Current templates?)
3. **User feedback loop**: How to collect ratings to improve retrieval quality?
4. **Multi-performer comparison**: How to cite when comparing two performers?
5. **Practice plan generation**: Separate RAG query for practice tips vs. feedback?

---

## References

### Semantic Collapse Research

- Google DeepMind: "On the Theoretical Limitations of Embedding-Based Retrieval" (2025)
- "Length-Induced Embedding Collapse in Transformer-based Models" (ACL 2025)

### RAG Best Practices

- RAGVue: Diagnostic evaluation framework
- ColBERT: Multi-vector retrieval

### Piano Pedagogy

- Eigeldinger: "Chopin: Pianist and Teacher" (Cambridge)
- C.P.E. Bach: "Essay on the True Art of Playing Keyboard Instruments"
